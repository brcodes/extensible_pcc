"""Train and evaluate predictive-coding models from key=value text configs.

CLI entry point for the pypredcoding pipeline: load a config from config/
(``python train_model.py --config config_active.txt``), build the params dict,
create a StaticPCC or RecurrentPCC model, train it (optionally resuming from a
checkpoint), and write logs/models under experiment-named (exp_YYMMDD_HHMMSS)
directories. Passing a config/ subdirectory runs every config*.txt in it as a
grid search.
"""

from __future__ import annotations

from ast import literal_eval
import argparse
from model import (
    create_model,
    build_model_output_name,
    model_name_stem_without_epoch,
    epoch_from_name,
    validate_config_key_policy,
)
from os.path import join, exists, getmtime, isdir, isfile
from os import listdir, makedirs
from pickle import load
from numbers import Integral
import numpy as np
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Any, Callable


def _install_numpy_pickle_compatibility_aliases() -> None:
    """Alias numpy._core module names to numpy.core so old-numpy pickles load under new numpy."""
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np.core

    compatible_submodules = (
        'multiarray',
        'numeric',
        'numerictypes',
        'fromnumeric',
        'arrayprint',
        'defchararray',
        'records',
        'memmap',
        'shape_base',
        'umath',
        'einsumfunc',
        'getlimits',
        'function_base',
        'overrides',
        '_multiarray_umath',
    )
    for submodule_name in compatible_submodules:
        old_name = f'numpy._core.{submodule_name}'
        if old_name not in sys.modules:
            try:
                sys.modules[old_name] = __import__(f'numpy.core.{submodule_name}', fromlist=['*'])
            except Exception:
                continue


def _is_experiment_name(value: Any) -> bool | re.Match[str] | None:
    """Check whether value is an experiment-name string like 'exp_YYMMDD_HHMMSS' (truthy on match)."""
    return isinstance(value, str) and re.match(r'^exp_\d{6}_\d{6}$', value)


def _parse_named_prior_dist_tuple(value_str: str) -> tuple[str, float, float] | tuple[str, float]:
    """Parse a named prior-dist tuple-string into (name, *numeric_args).

    Accepts ('gaussian'|'sparse_kurtotic', mean=K, scale=N) or
    ('shifted_uniform_random', shift=S).

    Raises:
        ValueError: If value_str matches neither supported format.
    """
    gauss_sparse_pattern = (
        r"^\(\s*['\"](?P<name>gaussian|sparse_kurtotic)['\"]\s*,\s*"
        r"mean\s*=\s*(?P<mean>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"scale\s*=\s*(?P<scale>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)$"
    )
    shifted_uniform_pattern = (
        r"^\(\s*['\"](?P<name>shifted_uniform_random)['\"]\s*,\s*"
        r"shift\s*=\s*(?P<shift>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)$"
    )

    match = re.match(gauss_sparse_pattern, value_str)
    if match:
        return (
            match.group('name'),
            float(match.group('mean')),
            float(match.group('scale')),
        )

    match = re.match(shifted_uniform_pattern, value_str)
    if match:
        return (
            match.group('name'),
            float(match.group('shift')),
        )

    raise ValueError(
        f"Unsupported prior dist tuple-string format: {value_str}. "
        "Use ('gaussian', mean=K, scale=N), ('sparse_kurtotic', mean=K, scale=N), "
        "or ('shifted_uniform_random', shift=S)."
    )


def _parse_prior_dist_value(value_str: str) -> Any:
    """Parse a prior-dist config value: try literal_eval, then the named-tuple grammar."""
    try:
        return literal_eval(value_str)
    except (ValueError, SyntaxError):
        return _parse_named_prior_dist_tuple(value_str)


def _normalize_load_checkpoint_value(value: Any) -> int | tuple[int, str] | None:
    """Validate and normalize a load_checkpoint config value.

    Returns:
        None, an int epoch (-1 means latest), or an (epoch, experiment_name) tuple.

    Raises:
        ValueError: If value is not one of the accepted forms.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                "load_checkpoint must be None, -1, a non-negative integer epoch, or "
                "a tuple like (epoch, 'exp_YYMMDD_HHMMSS')."
            )

        epoch_value, experiment_name = value
        if isinstance(epoch_value, bool) or not isinstance(epoch_value, Integral):
            raise ValueError(
                "load_checkpoint tuple epoch must be -1 or a positive integer."
            )
        normalized_epoch = int(epoch_value)
        if normalized_epoch != -1 and normalized_epoch <= 0:
            raise ValueError(
                "load_checkpoint tuple epoch must be -1 or a positive integer."
            )
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ValueError(
                "load_checkpoint tuple experiment name must be a non-empty string like 'exp_YYMMDD_HHMMSS'."
            )
        return normalized_epoch, experiment_name

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            "load_checkpoint must be None, -1, a non-negative integer epoch, or "
            "a tuple like (epoch, 'exp_YYMMDD_HHMMSS')."
        )

    normalized_value = int(value)
    if normalized_value < -1:
        raise ValueError(
            "load_checkpoint must be None, -1, a non-negative integer epoch, or "
            "a tuple like (epoch, 'exp_YYMMDD_HHMMSS')."
        )

    return normalized_value


def _normalize_seed_value(value: Any) -> int | None:
    """Validate that a seed is None or an integer (bools rejected) and return it as int."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("seed must be None or an integer.")
    return int(value)


def _experiment_models_dir(experiment_name: str) -> str:
    """Return the models directory path for an experiment."""
    return join('models', f'{experiment_name}_models')


def _experiment_checkpoints_dir(experiment_name: str) -> str:
    """Return the checkpoints directory path for an experiment."""
    return join(_experiment_models_dir(experiment_name), 'checkpoints')


def _resolve_final_model_path(model_name: str) -> str:
    """Find a saved final model by name under models/ and return the newest match.

    Searches the legacy flat models/ location and every models/*_models
    experiment directory.

    Raises:
        ValueError: If no saved model with that name exists.
    """
    legacy_path = join('models', model_name)
    candidates = []

    if exists(legacy_path):
        candidates.append(legacy_path)

    models_root = 'models'
    if exists(models_root):
        for child in listdir(models_root):
            child_path = join(models_root, child)
            candidate = join(child_path, model_name)
            if child.endswith('_models') and exists(candidate):
                candidates.append(candidate)

    if not candidates:
        raise ValueError(f"No saved final model found for {model_name}")

    return max(candidates, key=getmtime)


def _resolve_data_path(dataset_name: str) -> Path:
    """Resolve a dataset name to an existing pickle path in pypredcoding/data/.

    Names without a suffix are tried as-is, then with .pydb and .pkl.

    Raises:
        ValueError: If no candidate file exists.
    """
    data_dir = Path(__file__).resolve().parent / 'data'
    dataset_path = Path(dataset_name)

    if dataset_path.suffix:
        candidate_paths = [data_dir / dataset_path.name]
    else:
        candidate_paths = [
            data_dir / dataset_path.name,
            data_dir / f'{dataset_path.name}.pydb',
            data_dir / f'{dataset_path.name}.pkl',
        ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    attempted_names = ', '.join(str(path.name) for path in candidate_paths)
    raise ValueError(
        f"Dataset file not found: {dataset_name}. Expected a premade pickle named one of: {attempted_names}."
    )

def load_params(file_path: str) -> dict[str, Any]:
    """Parse a key=value config file into a params dict.

    Ignore blank and '#' comment lines; literal_eval each value (prior-dist
    keys get the named-tuple grammar fallback). Attach _config_source_path,
    _config_key_lines, and _config_entries metadata, then run
    validate_config_key_policy.

    Raises:
        ValueError: If a non-comment line lacks '=' or the key policy fails.
    """
    params = {}
    key_lines = {}
    entries = []
    prior_dist_keys = {'r_prior_dist', 'U_prior_dist'}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            # Skip comments and blank lines
            if stripped_line and not stripped_line.startswith('#'):
                if '=' in stripped_line:
                    variable, value = stripped_line.split('=', 1)
                    key = variable.strip()
                    value_str = value.strip()
                    if key in prior_dist_keys:
                        evaluated_value = _parse_prior_dist_value(value_str)
                    else:
                        evaluated_value = literal_eval(value_str)
                    if key == 'load_checkpoint':
                        evaluated_value = _normalize_load_checkpoint_value(evaluated_value)
                    params[key] = evaluated_value
                    key_lines.setdefault(key, []).append(line_number)
                    entries.append((key, evaluated_value, line_number))
                else:
                    raise ValueError(f"Invalid line in config file: {line}")
    params['_config_source_path'] = file_path
    params['_config_key_lines'] = key_lines
    params['_config_entries'] = tuple(entries)
    validate_config_key_policy(
        params,
        config_source_path=file_path,
        config_key_lines=key_lines,
        config_entries=entries,
    )
    return params

def model_name_from_params(params: dict[str, Any], experiment_name: str | None = None) -> str:
    """Build the canonical model filename from params and an experiment name.

    Canonical format: prefix.modeltype.exp_YYMMDD_HHMMSS.numepochs.extension,
    e.g. mod.recurrent.exp_260706_095415.100.pydb. If experiment_name is None,
    a valid params['exp_name'] is required.

    Raises:
        ValueError: If no experiment name in exp_YYMMDD_HHMMSS format is available.
    """
    if experiment_name is None:
        configured_name = params.get('exp_name')
        if _is_experiment_name(configured_name):
            experiment_name = configured_name
        else:
            raise ValueError(
                'model_name_from_params requires experiment_name in format exp_YYMMDD_HHMMSS.'
            )

    prefix = params.get('model_prefix', 'mod')
    model_type = params['model_type']
    num_epochs = int(params['epoch_n'])
    return build_model_output_name(
        prefix=prefix,
        model_type=model_type,
        experiment_name=experiment_name,
        epoch=num_epochs,
        extension='pydb',
    )

def load_model(model_path: str) -> Any:
    """Unpickle and return a saved model (StaticPCC or RecurrentPCC) from model_path.

    Raises:
        ValueError: If model_path does not exist.
    """
    if not exists(model_path):
        raise ValueError(f"The model path {model_path} does not exist.")
    
    _install_numpy_pickle_compatibility_aliases()
    with open(model_path, 'rb') as file:
        model = load(file)
        

    return model

def load_checkpoint(model_name: str, params: dict[str, Any]) -> tuple[Any, int]:
    """Load the checkpoint matching model_name per params['load_checkpoint'].

    Resolves the checkpoint directory (current experiment's, or the one named
    in an (epoch, experiment_name) tuple) and picks the latest epoch when the
    requested epoch is -1, else the exact epoch.

    Returns:
        Tuple of the loaded model and its checkpoint epoch.

    Raises:
        ValueError: If no matching or resolvable checkpoint file is found.
    """
    requested_epoch = _normalize_load_checkpoint_value(params['load_checkpoint'])
    if requested_epoch is None:
        raise ValueError('load_checkpoint cannot be None when loading a checkpoint.')

    if isinstance(requested_epoch, tuple):
        requested_epoch_value, experiment_name = requested_epoch
        checkpoint_dir = _experiment_checkpoints_dir(experiment_name)
    else:
        requested_epoch_value = requested_epoch
        checkpoint_dir = join('models', 'checkpoints')
    
    # Candidate checkpoints share the model name stem up to the epoch field.
    model_name_no_epoch_no_pydb = model_name_stem_without_epoch(model_name)
    
    chk_dir_names = listdir(checkpoint_dir)
    matching_fns = [f for f in chk_dir_names if f.startswith(model_name_no_epoch_no_pydb)]
    if not matching_fns:
        raise ValueError('No matching checkpoint found')
    
    if requested_epoch_value == -1:
        
        print('Loading latest checkpoint')
        # Latest checkpoint: keep the highest parseable epoch
        max_epoch = -1
        chk_file_path = None
        file_name_valid = False
        
        for filename in matching_fns:
            try:
                epoch = epoch_from_name(filename)
                # Epoch parsed, so the filename format is valid
                file_name_valid = True
                
                if epoch > max_epoch:
                    max_epoch = chk_epoch = epoch
                    chk_file_path = join(checkpoint_dir, filename)
            except ValueError:
                continue
            
    else:
        
        print(f'Loading checkpoint at epoch {requested_epoch_value}')
        desired_epoch = requested_epoch_value
        chk_file_path = None
        file_name_valid = False
        
        for filename in matching_fns:
            try:
                epoch = epoch_from_name(filename)
                # Epoch parsed, so the filename format is valid
                file_name_valid = True
                
                if epoch == desired_epoch:
                    chk_file_path = join(checkpoint_dir, filename)
                    chk_epoch = desired_epoch
            except ValueError:
                continue
            
    if not file_name_valid:
        raise ValueError('No valid checkpoint filenames found')
    if chk_file_path is None:
        if requested_epoch_value == -1:
            raise ValueError('No checkpoint epochs could be resolved from matching checkpoint filenames')
        raise ValueError(f'No checkpoint found for requested epoch {requested_epoch_value}')
    
    checkpoint = load_model(chk_file_path)
    
    print(f'Loaded checkpoint: {chk_file_path}')
    
    return checkpoint, chk_epoch


def load_configured_data(config_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the (X, Y) dataset named by a config's dataset_train, independent of caller cwd."""
    params = load_params(str(config_path))
    data_path = _resolve_data_path(params['dataset_train'])
    _install_numpy_pickle_compatibility_aliases()
    with data_path.open('rb') as data_file:
        X, Y = load(data_file)
    return np.asarray(X), np.asarray(Y)


def run_bounded_experiment(
    config_path: str | Path,
    *,
    seed: int,
    epochs: int,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    num_ts: int | None = None,
    run_name: str = 'parity_test',
    before_train: Callable[[Any, np.ndarray, np.ndarray], Any] | None = None,
    train_seed: int | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, Path, Any]:
    """Run the current extensible training path with in-memory test controls.

    Override epoch count, seed(s), data, and checkpointing on a config-built
    model for bounded parity/smoke tests, then train without plotting.

    Args:
        before_train: Optional hook called as before_train(model, X, Y) before
            training; its return value is passed through.

    Returns:
        Tuple of (model, X, Y, resolved config_path, before_train result).
    """
    config_path = Path(config_path).resolve()
    params = dict(load_params(str(config_path)))
    if X is None or Y is None:
        X, Y = load_configured_data(config_path)
    X = np.asarray(X)
    Y = np.asarray(Y)

    params['epoch_n'] = int(epochs)
    params['num_inps'] = int(X.shape[0])
    params['save_checkpoint'] = {}
    params['load_checkpoint'] = None
    params['seed'] = int(seed)
    if num_ts is not None:
        params['num_ts'] = int(num_ts)

    np.random.seed(params['seed'])
    model = create_model(params)
    log_dir = Path(__file__).resolve().parent / 'log'
    log_dir.mkdir(exist_ok=True)
    log_name = f'{run_name}_log.txt'
    (log_dir / log_name).write_text('', encoding='utf-8')
    model.exp_log_name = log_name
    model.exp_log_path = log_dir / log_name
    model.mod_name = f'{run_name}.pydb'
    model.config_epoch_n = None
    before_train_result = None
    if before_train is not None:
        before_train_result = before_train(model, X, Y)
    if train_seed is not None:
        np.random.seed(int(train_seed))
    model.train(X, Y, save_checkpoint={}, load_checkpoint=None, plot=False)
    return model, X, Y, config_path, before_train_result


def inspect_bounded_inference(model: Any, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Run the model's active inference operations and return raw final states.

    Runs label-free inference per input (per-timestep for recurrent models)
    and evaluates classification accuracy.

    Returns:
        Tuple of stacked final top-layer states and the accuracy as float.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    update_method_name = next(iter(model.update_method))
    update_method_number = model.update_method[update_method_name]

    if model.model_type == 'recurrent':
        prior_dist = model.recurrent_reset_prior_sampler()
        states = []
        for input_value, label in zip(X, Y):
            model.reset_rs_gteq1(all_lyr_sizes=model.all_lyr_sizes, prior_dist=prior_dist)
            model.fill_UsVs_with_last_timestep()
            valid_num_ts = model.valid_timesteps_until_nan(input_value)
            for timestep in range(valid_num_ts):
                model.r[0] = input_value[:, timestep]
                model.update_method_no_weight_dict[update_method_name](
                    update_method_number, ts=timestep, label=np.zeros_like(label)
                )
            states.append(np.array(model.rhat[model.num_layers][:, valid_num_ts - 1], dtype=np.float64))
    else:
        model.hard_set_r_prior_dist()
        prior_dist = model.load_hard_r_prior_dist
        states = []
        for input_value, label in zip(X, Y):
            model.reset_rs_gteq1(all_lyr_sizes=model.all_lyr_sizes, prior_dist=prior_dist)
            model.r[0] = input_value
            model.update_method_no_weight_dict[update_method_name](
                update_method_number, label=np.zeros_like(label)
            )
            states.append(np.array(model.r[model.num_layers], dtype=np.float64))

    accuracy = model.evaluate(
        X,
        Y,
        update_method_name=update_method_name,
        update_method_number=update_method_number,
        plot=None,
    )
    return np.stack(states), float(accuracy)

def load_data(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a premade (X, Y) dataset pickle by name from pypredcoding/data/."""
    data_path = _resolve_data_path(dataset_name)

    _install_numpy_pickle_compatibility_aliases()
    with data_path.open('rb') as file:
        X, Y = load(file)

    return X, Y

def print_params(params: dict[str, Any]) -> None:
    """Print each params key-value pair to stdout."""
    for key, value in params.items():
        print(f'{key}: {value}')
        
def initiate_log(params: dict[str, Any], experiment_name: str) -> str:
    """Create log/<experiment_name>_log.txt seeded with all params and return its filename."""
    log_dir = 'log'
    makedirs(log_dir, exist_ok=True)
    log_file_name = f'{experiment_name}_log.txt'
    log_file_path = join(log_dir, log_file_name)
    with open(log_file_path, 'w') as log_file:
        log_file.write(f'Experiment log begin: {log_file_path}\n')
        for key, value in params.items():
            log_file.write(f'{key}: {value}\n')
    return log_file_name

def init_log_print_params(params: dict[str, Any], experiment_name: str) -> tuple[str, bool]:
    """Initiate the experiment log, print params, and return (log filename, True)."""
    log_file_name = initiate_log(params, experiment_name)
    print_params(params)
    initiated_log = True
    return log_file_name, initiated_log

def run_experiment(config_file_path: str) -> dict[str, Any]:
    """Run one full experiment from a config file: build/load model, log, train.

    Normalizes checkpoint/seed params, stamps a new exp_YYMMDD_HHMMSS name,
    creates the model (or resumes from a checkpoint), initializes the log,
    loads the training data, and trains.

    Returns:
        Dict with experiment_name, experiment_log_path, and final_kpis.
    """
    params = load_params(config_file_path)
    params['load_checkpoint'] = _normalize_load_checkpoint_value(params.get('load_checkpoint'))
    params['seed'] = _normalize_seed_value(params.get('seed'))
    params['seed_init'] = _normalize_seed_value(params.get('seed_init', params['seed']))
    params['seed_shuffle'] = _normalize_seed_value(params.get('seed_shuffle', params['seed']))

    if params['seed_init'] is not None:
        np.random.seed(params['seed_init'])
        print(f"Initialization seed set from config: {params['seed_init']}")

    run_experiment_name = datetime.now().strftime('exp_%y%m%d_%H%M%S')
    params['exp_name'] = run_experiment_name
    model_name = model_name_from_params(
        params,
        experiment_name=run_experiment_name,
    )
    
    initiated_log = False
    log_file_name = None
    final_kpis = None
    
    # Model
    if params['load_checkpoint'] is not None:
        model, epoch = load_checkpoint(model_name, params)
        load_name_ep_params = {'load_name': model_name, 'load_epoch': epoch, 'config_epoch_n': params['epoch_n']}
        model.set_model_attributes(load_name_ep_params)
        print(f'Desired final state: {model_name}\n')
    else:
        model = create_model(params)
        model_name_ep_param = {'mod_name': model_name, 'config_epoch_n': None}
        model.set_model_attributes(model_name_ep_param)
        print(f'Instantiated model for desired final state: {model_name}\n')

    # Print and log params (base log for appending)
    log_file_name, initiated_log = init_log_print_params(params, run_experiment_name)
    log_file_name_param = {'exp_log_name': log_file_name}
    model.set_model_attributes(log_file_name_param)
    
    # Data
    X_train, Y_train = load_data(params['dataset_train'])
    print(f'Loaded data: {params["dataset_train"]}')
    
    # Train: will shuffle data automatically
    if params['seed_shuffle'] is not None:
        np.random.seed(params['seed_shuffle'])
        print(f"Training shuffle seed set from config: {params['seed_shuffle']}")
    model.train(X_train, Y_train, save_checkpoint=params['save_checkpoint'], load_checkpoint=params['load_checkpoint'], plot=params['plot_train'])
    final_kpis = getattr(model, 'final_kpis', None)

    experiment_log_path = join('log', log_file_name) if log_file_name else None
    return {
        'experiment_name': run_experiment_name,
        'experiment_log_path': experiment_log_path,
        'final_kpis': final_kpis,
    }

def run_experiment_grid(config_dir_path: str) -> None:
    """Run every config*.txt in a directory sequentially and write a grid summary log.

    Failures are recorded and the grid continues; KeyboardInterrupt writes the
    summary before re-raising.

    Raises:
        ValueError: If the directory contains no runnable config*.txt files.
    """
    config_file_paths = [
        join(config_dir_path, file_name)
        for file_name in sorted(listdir(config_dir_path))
        if file_name.startswith('config') and file_name.endswith('.txt') and isfile(join(config_dir_path, file_name))
    ]

    if not config_file_paths:
        raise ValueError(f'No runnable config*.txt files found in grid search directory: {config_dir_path}')

    dir_name = config_dir_path.replace('\\', '/').rstrip('/').split('/')[-1]
    # Reuse the config-generation timestamp from the dir name when present
    dir_timestamp_match = re.match(r'^configs_grid_search_(\d{6}_\d{6})$', dir_name)
    if dir_timestamp_match:
        grid_config_timestamp = dir_timestamp_match.group(1)
    else:
        grid_config_timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    grid_runtime_timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    grid_run_name = f'grid_search_{grid_config_timestamp}_run_{grid_runtime_timestamp}'
    log_dir = 'log'
    makedirs(log_dir, exist_ok=True)
    grid_log_name = f'{grid_run_name}_log.txt'
    grid_log_path = join(log_dir, grid_log_name)

    grid_start_time = datetime.now()
    status_rows = []

    def config_summary(config_file_path: str) -> dict[str, Any]:
        """Summarize key params of a config for the grid log; report parse errors inline."""
        try:
            params = load_params(config_file_path)
        except Exception as error:
            return {'metadata_error': f'{type(error).__name__}: {error}'}

        summary_keys = (
            'notes', 'epoch_n', 'seed', 'seed_init', 'seed_shuffle',
            'hidden_lyr_sizes', 'top_lyr_size', 'kr', 'kU', 'rate_schedule',
        )
        return {key: params[key] for key in summary_keys if key in params}

    def write_grid_summary() -> None:
        """Write (or overwrite) the grid summary log from the status rows so far."""
        total_configs = len(config_file_paths)
        num_passed = sum(1 for row in status_rows if row['status'] == 'PASS')
        num_failed = sum(1 for row in status_rows if row['status'] == 'FAIL')
        num_interrupted = sum(1 for row in status_rows if row['status'] == 'INTERRUPTED')
        grid_end_time = datetime.now()
        total_time = grid_end_time - grid_start_time

        with open(grid_log_path, 'w') as log_file:
            print(f'Grid search log begin: {grid_log_path}', file=log_file)
            print(f'Grid search name: {grid_run_name}', file=log_file)
            print(f'Grid search config dir: {config_dir_path}', file=log_file)
            print(f'Start time: {grid_start_time.strftime("%Y-%m-%d %H:%M:%S")}', file=log_file)
            print(f'End time: {grid_end_time.strftime("%Y-%m-%d %H:%M:%S")}', file=log_file)
            print(f'Total time: {total_time}', file=log_file)
            print(f'Total configs discovered: {total_configs}', file=log_file)
            print(f'Configs attempted: {len(status_rows)}', file=log_file)
            print(f'Passed: {num_passed}', file=log_file)
            print(f'Failed: {num_failed}', file=log_file)
            print(f'Interrupted: {num_interrupted}', file=log_file)
            print('', file=log_file)

            for row in status_rows:
                print(f'Run {row["index"]}/{total_configs}: {row["config"]}', file=log_file)
                print(f'Status: {row["status"]}', file=log_file)
                metadata = row.get('config_summary')
                if metadata:
                    print('Config summary:', file=log_file)
                    for key, value in metadata.items():
                        print(f'  {key}: {value}', file=log_file)
                if row.get('status') == 'PASS':
                    kpis = row.get('kpis')
                    if kpis:
                        print('KPIs:', file=log_file)
                        print(f'  {kpis.get("header_line", "Final diagnostics")}', file=log_file)
                        for kpi_line in kpis.get('lines', []):
                            print(f'  {kpi_line}', file=log_file)
                    else:
                        print('KPIs: unavailable', file=log_file)
                if row.get('error_type'):
                    print(f'Error type: {row["error_type"]}', file=log_file)
                if row.get('error_message'):
                    print(f'Error message: {row["error_message"]}', file=log_file)
                print('', file=log_file)

    print(f'Running grid search directory: {config_dir_path}')
    print(f'Found {len(config_file_paths)} config files.')
    print(f'Grid search summary log: {grid_log_path}')
    for index, config_file_path in enumerate(config_file_paths, start=1):
        print(f'\nGrid search run {index}/{len(config_file_paths)}: {config_file_path}')
        metadata = config_summary(config_file_path)
        try:
            run_result = run_experiment(config_file_path)
            exp_log_path = None
            kpis = None
            if isinstance(run_result, dict):
                exp_log_path = run_result.get('experiment_log_path')
                kpis = run_result.get('final_kpis')
            status_rows.append(
                {
                    'index': index,
                    'config': config_file_path,
                    'status': 'PASS',
                    'config_summary': metadata,
                    'experiment_log_path': exp_log_path,
                    'kpis': kpis,
                }
            )
        except KeyboardInterrupt:
            status_rows.append(
                {'index': index, 'config': config_file_path, 'status': 'INTERRUPTED', 'config_summary': metadata}
            )
            write_grid_summary()
            print(f'Grid search interrupted by user. Wrote summary log to: {grid_log_path}')
            raise
        except Exception as error:
            status_rows.append(
                {
                    'index': index,
                    'config': config_file_path,
                    'status': 'FAIL',
                    'config_summary': metadata,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                }
            )
            print(f'Grid search run failed and will continue: {config_file_path}')
            print(f'  {type(error).__name__}: {error}')

    write_grid_summary()
    print(f'Grid search complete. Wrote summary log to: {grid_log_path}')

def main() -> None:
    """Run one experiment config, or every config in a grid-search directory, inside config/.

    Raises:
        ValueError: If the requested path is missing, escapes config/, or is malformed.
    """
    config_folder = 'config'
    if not exists(config_folder):
        raise ValueError(f"The config folder {config_folder} does not exist.")

    parser = argparse.ArgumentParser(
        description='Run an experiment config, or all .txt configs in a config/ subdirectory.'
    )
    parser.add_argument(
        '--config',
        dest='config_path_flag',
        help='Path to config file or config directory relative to config/ (example: config_rPCC.txt).',
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        help='Legacy positional config path (deprecated): use --config instead.',
    )
    args = parser.parse_args()

    requested_path_arg = args.config_path_flag if args.config_path_flag is not None else args.config_path
    if requested_path_arg is None:
        parser.error('Please provide a config path with --config (or legacy positional config_path).')

    # Prevent paths from escaping config/ while still allowing nested paths.
    requested_path = requested_path_arg.strip()
    if not requested_path or requested_path.startswith('/'):
        raise ValueError('config_path must be a non-empty path relative to config/.')

    normalized_requested_path = requested_path.replace('\\', '/').lstrip('./')
    if normalized_requested_path == 'config':
        raise ValueError('config_path must point to a config file, not the config directory.')

    if normalized_requested_path.startswith('config/'):
        normalized_requested_path = normalized_requested_path[len('config/'):]

    if not normalized_requested_path:
        raise ValueError('config_path must point to a file inside config/.')

    if '..' in normalized_requested_path.split('/'):
        raise ValueError('config_path cannot contain parent-directory traversal (..).')

    config_file_path = join(config_folder, normalized_requested_path)

    if not exists(config_file_path):
        raise ValueError(f"The config path {config_file_path} does not exist.")

    if isdir(config_file_path):
        run_experiment_grid(config_file_path)
        return

    print(f'Running experiment with config file: {config_file_path}')
    run_experiment(config_file_path)

if __name__ == '__main__':
    main()