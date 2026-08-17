"""Predictive-coding classifier models (Rao & Ballard 1999 style, extended per Rogers 2026).

Provide the abstract PredictiveCodingClassifier base class, the StaticPCC and
RecurrentPCC concrete models, the create_model factory, and helpers for model
output naming, config-policy validation, receptive-field rendering, and
training diagnostics. Models are built from a params dict loaded from the
repo's text configs and delegate their update/cost math to cost_functions.
"""
from __future__ import annotations

from cost_functions import StaticCostFunction, RecurrentCostFunction
from abc import ABC, abstractmethod
from ast import literal_eval
import numpy as np
from functools import partial
from copy import copy
from datetime import datetime
import pickle
import re
import sys
from os import makedirs
from os.path import join, exists, dirname, isfile
from pathlib import Path
from typing import Any, Callable, Iterable

# Agg keeps plotting headless-safe (e.g. cluster runs).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODEL_OUTPUT_NAME_RE = re.compile(
    r'^(?P<prefix>[A-Za-z0-9_-]+)\.(?P<model_type>[A-Za-z0-9_-]+)\.'
    r'(?P<experiment_name>exp_\d{6}_\d{6})\.(?P<epoch>\d+)\.(?P<extension>[A-Za-z0-9]+)$'
)

LEGACY_MODEL_OUTPUT_NAME_RE = re.compile(
    r'^(?P<stem>[A-Za-z0-9._-]+?)(?:\.(?P<epoch>\d+))?\.(?P<extension>[A-Za-z0-9]+)$'
)


def install_numpy_pickle_compatibility_aliases() -> None:
    """Alias numpy._core submodule names to numpy.core so newer-NumPy pickles unpickle here."""
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


def build_model_output_name(prefix: str, model_type: str, experiment_name: str, epoch: int, extension: str = 'pydb') -> str:
    """Build a canonical model output name: prefix.model_type.exp_YYMMDD_HHMMSS.epoch.extension."""
    if not isinstance(experiment_name, str) or not re.match(r'^exp_\d{6}_\d{6}$', experiment_name):
        raise ValueError(
            'experiment_name must match exp_YYMMDD_HHMMSS to build a model output name.'
        )
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('epoch must be a non-negative integer.')
    if not isinstance(extension, str) or not extension:
        raise ValueError('extension must be a non-empty string.')
    return f'{prefix}.{model_type}.{experiment_name}.{epoch}.{extension}'


def parse_model_output_name(output_name: str) -> dict[str, Any]:
    """Parse a model output name into its component fields.

    Returns:
        Dict with 'format' ('new' or 'legacy'), 'prefix', 'model_type',
        'experiment_name', 'epoch', 'extension', and 'stem' (name without
        epoch/extension); legacy names leave unavailable fields as None.

    Raises:
        ValueError: If the name matches neither the canonical nor legacy format.
    """
    if not isinstance(output_name, str):
        raise ValueError('output_name must be a string.')

    match = MODEL_OUTPUT_NAME_RE.fullmatch(output_name)
    if match:
        groups = match.groupdict()
        return {
            'format': 'new',
            'prefix': groups['prefix'],
            'model_type': groups['model_type'],
            'experiment_name': groups['experiment_name'],
            'epoch': int(groups['epoch']),
            'extension': groups['extension'],
            'stem': f"{groups['prefix']}.{groups['model_type']}.{groups['experiment_name']}",
        }

    legacy_match = LEGACY_MODEL_OUTPUT_NAME_RE.fullmatch(output_name)
    if legacy_match:
        legacy_groups = legacy_match.groupdict()
        legacy_epoch = legacy_groups['epoch']
        return {
            'format': 'legacy',
            'prefix': None,
            'model_type': None,
            'experiment_name': None,
            'epoch': int(legacy_epoch) if legacy_epoch is not None else None,
            'extension': legacy_groups['extension'],
            'stem': legacy_groups['stem'],
        }

    raise ValueError(
        f'Output name {output_name} must match canonical model naming format: '
        'prefix.modeltype.exp_YYMMDD_HHMMSS.numepochs.extension, '
        'or legacy format stem[.epoch].extension.'
    )


def model_name_stem_without_epoch(output_name: str) -> str:
    """Return the model output name minus its epoch and extension."""
    return parse_model_output_name(output_name)['stem']


def epoch_from_name(output_name: str) -> int:
    """Return the epoch number embedded in a model output name.

    Raises:
        ValueError: If the name carries no epoch number.
    """
    epoch = parse_model_output_name(output_name)['epoch']
    if epoch is None:
        raise ValueError(f'Output name {output_name} does not contain an epoch number.')
    return epoch


def with_epoch(output_name: str, epoch: int) -> str:
    """Return output_name rebuilt with the given epoch number."""
    parsed = parse_model_output_name(output_name)
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('epoch must be a non-negative integer.')

    if parsed['format'] == 'legacy':
        return f"{parsed['stem']}.{epoch}.{parsed['extension']}"

    return build_model_output_name(
        prefix=parsed['prefix'],
        model_type=parsed['model_type'],
        experiment_name=parsed['experiment_name'],
        epoch=epoch,
        extension=parsed['extension'],
    )


def normalize_plot_receptive_fields_setting(setting: bool | dict[str, Any]) -> dict[str, Any]:
    """Normalize the plot_receptive_fields config value to a canonical spec dict.

    Returns:
        {'mode': None}, {'mode': 'all'}, or {'mode': 'random_sample', 'random_sample': N}.

    Raises:
        ValueError: For any other form of setting.
    """
    if isinstance(setting, bool):
        return {'mode': 'all' if setting else None}

    if not isinstance(setting, dict):
        raise ValueError(
            'plot_receptive_fields must be set to a bool or a dict like '
            "{'mode': None}, {'mode': 'all'}, or {'mode': {'random_sample': N}}."
        )

    if set(setting.keys()) != {'mode'}:
        raise ValueError(
            "plot_receptive_fields dict form must contain only the 'mode' key. "
            "Use {'mode': None}, {'mode': 'all'}, or {'mode': {'random_sample': N}}."
        )

    mode = setting['mode']
    if mode is None:
        return {'mode': None}

    if mode == 'all':
        return {'mode': 'all'}

    if isinstance(mode, dict) and set(mode.keys()) == {'random_sample'}:
        random_sample = mode['random_sample']
        if isinstance(random_sample, bool) or not isinstance(random_sample, int) or random_sample < 1:
            raise ValueError(
                "plot_receptive_fields random_sample must be a positive integer. "
                "Use {'mode': {'random_sample': N}} with N >= 1."
            )
        return {'mode': 'random_sample', 'random_sample': int(random_sample)}

    raise ValueError(
        "plot_receptive_fields dict form must be one of: {'mode': None}, {'mode': 'all'}, "
        "or {'mode': {'random_sample': N}}."
    )


def validate_config_key_policy(
    params: dict[str, Any],
    *,
    config_source_path: str | None = None,
    config_key_lines: dict[str, list[int]] | None = None,
    config_entries: Iterable[tuple[str, Any, int | None]] | None = None,
) -> dict[str, Any]:
    """Validate cross-key config policy ('o'-key usage, schedules, plotting flags).

    Args:
        params: Full config dict (or model __dict__) to check.
        config_source_path: Config file path used to point error messages at lines.
        config_key_lines: Map of key name to source-config line numbers.
        config_entries: Optional (key, value, line_no) triples to check instead of params.

    Returns:
        The params dict unchanged.

    Raises:
        ValueError: With all accumulated policy violations joined into one message.
    """
    classif_method = params.get('classif_method')
    model_type = params.get('model_type')
    is_c2 = classif_method == 'c2'
    rate_schedule = params.get('rate_schedule')

    def format_location(param_name: str, line_no: int | None = None) -> str:
        """Format a config file location for an error message."""
        if line_no is None and config_key_lines is not None:
            line_numbers = config_key_lines.get(param_name)
            if line_numbers:
                line_no = line_numbers[-1]

        if config_source_path is not None and line_no is not None:
            return f'{config_source_path}:{line_no}'
        if line_no is not None:
            return f'line {line_no}'
        if config_source_path is not None:
            return config_source_path
        return 'this config'

    issues = []
    seen_issues = set()

    def add_issue(message: str) -> None:
        """Record a violation once, preserving first-seen order."""
        if message not in seen_issues:
            seen_issues.add(message)
            issues.append(message)

    if classif_method not in ('c1', 'c2', None):
        add_issue(
            f"classif_method must be 'c1', 'c2', or None. Got {classif_method!r}. Edit {format_location('classif_method')}."
        )

    if config_entries is None:
        config_entries = [
            (key, params.get(key), None)
            for key in ('kr', 'kU', 'kV', 'alph', 'lam', 'ssq', 'ssqr', 'ssqV', 'rate_schedule')
        ]

    for param_name, value, line_no in config_entries:
        if param_name in {'kr', 'kU', 'kV', 'alph', 'lam', 'ssq', 'ssqr', 'ssqV'}:
            if not isinstance(value, dict) or 'o' not in value:
                continue

            location = format_location(param_name, line_no)
            if param_name in {'kr', 'kV', 'alph', 'ssq', 'ssqr', 'ssqV'}:
                add_issue(
                    f"{param_name} must not contain an 'o' key. Edit {location}."
                )
            elif param_name in {'kU', 'lam'} and not is_c2:
                add_issue(
                    f"{param_name} may contain an 'o' key only when classif_method='c2'. Edit {location}."
                )
        elif param_name == 'rate_schedule' and value is not None:
            schedules = value if isinstance(value, (list, tuple)) else [value]
            for schedule in schedules:
                if not isinstance(schedule, dict):
                    raise ValueError('rate_schedule must be None, a schedule dictionary, or a list/tuple of schedule dictionaries.')

                schedule_type = schedule.get('type')
                schedule_location = format_location(param_name, line_no)
                if schedule_type not in {'piecewise_linear', 'two_phase_linear_decay'}:
                    add_issue(
                        f"Unsupported rate_schedule type: {schedule_type!r}. Edit {schedule_location}."
                    )

                schedule_key = schedule.get('key')
                if schedule_key == 'o':
                    if not is_c2:
                        add_issue(
                            f"rate_schedule key 'o' is only allowed when classif_method='c2'. Edit {schedule_location}."
                        )
                    elif schedule.get('param') != 'kU':
                        add_issue(
                            f"rate_schedule key 'o' is only valid for kU schedules. Edit {schedule_location}."
                        )

    if classif_method == 'c1' and params.get('num_classes') != params.get('top_lyr_size'):
        add_issue(
            f"num_classes must equal top_lyr_size when classif_method='c1'. Edit {format_location('num_classes')} and {format_location('top_lyr_size')}."
        )

    plot_receptive_fields = params.get('plot_receptive_fields')
    try:
        plot_receptive_fields_spec = normalize_plot_receptive_fields_setting(plot_receptive_fields)
    except ValueError as error:
        add_issue(f"{error} Edit {format_location('plot_receptive_fields')}.")
        plot_receptive_fields_spec = None
    else:
        if model_type == 'recurrent' and plot_receptive_fields_spec['mode'] is not None:
            add_issue(
                f"plot_receptive_fields is only supported for static models. Edit {format_location('plot_receptive_fields')} to set it False or {{'mode': None}}."
            )
        elif plot_receptive_fields_spec['mode'] is not None and not params.get('plot_train', False):
            add_issue(
                f"plot_receptive_fields requires plot_train=True. Edit {format_location('plot_receptive_fields')} or {format_location('plot_train')} so receptive-field plotting can run."
            )

    plot_softmaxed_activations = params.get('plot_softmaxed_activations')
    if not isinstance(plot_softmaxed_activations, bool):
        add_issue(
            f"plot_softmaxed_activations must be explicitly set to True or False. Edit {format_location('plot_softmaxed_activations')}."
        )
    elif plot_softmaxed_activations and not params.get('plot_train', False):
        add_issue(
            f"plot_softmaxed_activations requires plot_train=True. Edit {format_location('plot_softmaxed_activations')} or {format_location('plot_train')} so the activation diagnostics plot can run."
        )
    elif model_type == 'static' and plot_softmaxed_activations:
        add_issue(
            f"plot_softmaxed_activations is not supported for static models. Edit {format_location('plot_softmaxed_activations')} to set it False. Activation calculations are supported in plotting, but are skipped internally because no SPCC word-processing model datasets currently provide the *_tcru_index_map lookup tables required by calculate_average_softmaxed_activations_over_correct_inputs(). Without those maps, the code cannot map inputs to cohort, rhyme, and unrelated identities."
        )

    softmax_k_eval = params.get('softmax_k_eval')
    if model_type == 'static':
        if softmax_k_eval is not None:
            add_issue(
                f"softmax_k_eval is recurrent-only; set it to None for static models. Edit {format_location('softmax_k_eval')}."
            )
    elif model_type == 'recurrent':
        if isinstance(softmax_k_eval, bool) or not isinstance(softmax_k_eval, (int, float)) or softmax_k_eval <= 0:
            add_issue(
                f"softmax_k_eval must be a positive number for recurrent models. Edit {format_location('softmax_k_eval')}."
            )

    if is_c2:
        required_c2_keys = ['kU']
        if model_type == 'static':
            required_c2_keys.append('lam')

        for param_name in required_c2_keys:
            value = params.get(param_name)
            if isinstance(value, dict) and 'o' in value:
                continue

            location = format_location(param_name)
            add_issue(
                f"{param_name} must include an 'o' key when classif_method='c2'. Edit {location} to add {param_name}['o']."
            )

    if issues:
        raise ValueError(' '.join(issues))

    return params

class PredictiveCodingClassifier(ABC):
    """Abstract base for predictive-coding classifiers (sPCC/rPCC).

    Hold shared config validation, prior sampling, activation and softmax
    choices, r/weight update drivers, and diagnostics/plotting used by both
    StaticPCC and RecurrentPCC. r[i] are layer representations (r[0] is the
    current input); U[i] are generative top-down weights (U['o'] classifies
    under the c2 method).
    """

    REQUIRED_CONFIG_KEYS_SPCC = (
        'notes',
        'input_shape', 'ntiles_per_input', 'num_layers', 'hidden_lyr_sizes', 'top_lyr_size',
        'architecture', 'first_to_second_lyr_connection_architecture',
        'classif_method', 'r_prior_dist', 'U_prior_dist',
        'r_prior_cost', 'U_prior_cost', 'r_prior_cost_denominator', 'U_prior_cost_denominator', 'update_method',
        'softmax_type', 'softmax_k', 'model_type', 'flat_input',
        'activ_func', 'num_inps', 'num_classes', 'dataset_train', 'epoch_n',
        'kr', 'kU', 'alph', 'lam', 'ssq', 'plot_train', 'plot_receptive_fields', 'plot_softmaxed_activations', 'save_checkpoint',
        'load_checkpoint', 'batch_size'
    )

    REQUIRED_CONFIG_KEYS_RPCC = (
        'notes',
        'input_shape', 'num_ts', 'num_layers', 'hidden_lyr_sizes', 'top_lyr_size',
        'architecture', 'first_to_second_lyr_connection_architecture',
        'classif_method',
        'rc_topdown_cost_denominator', 'r_prior_dist', 'U_prior_dist', 'r_prior_cost', 'U_prior_cost',
        'r_prior_cost_denominator', 'U_prior_cost_denominator',
        'update_method', 'softmax_type', 'softmax_k', 'softmax_k_eval', 'model_type',
        'activ_func', 'num_inps', 'num_classes', 'dataset_train', 'epoch_n',
        'kr', 'kU', 'kV', 'ssqr', 'ssqV', 'plot_train', 'plot_receptive_fields', 'plot_softmaxed_activations', 'save_checkpoint',
        'load_checkpoint', 'batch_size'
    )

    REQUIRED_CONFIG_KEYS = ()

    def __init__(self, params: dict[str, Any]) -> None:
        """Register activation, prior, and softmax function choices, then set attributes from params."""
        
        # Choices for transformation functions and prior costs.
        # Each entry: (f, F_mult) — activation and its diag(f')-times-error operator.
        self.act_fxn_dict = {'linear': (self.linear_activation, self.linear_error_gradient),
                                'tanh': (self.tanh_activation, self.tanh_error_gradient)}

        self.prior_dist_dict = {'gaussian': self.normal_using_mean_and_scale,
                    'sparse_kurtotic': self.laplace_using_mean_and_scale,
                    'pseudo_sparse_kurtotic': self.pseudo_sparse_kurtotic,
                    'uniform_random': self.uniform_using_optional_shift,
        }
        
        self.prior_cost_dict = {'gaussian': self.gaussian_prior_costs,
                                'sparse_kurtotic': self.sparse_kurtotic_prior_costs}
        
        self.softmax_dict = {'normal': self.softmax,
                            'stable': self.stable_softmax}

        self.set_model_attributes(params)

    @abstractmethod
    def _configure_model(self) -> None:
        """Validate parameters and initialize concrete model state."""
        
    def set_model_attributes(self, params: dict[str, Any]) -> None:
        """Set every params entry as an attribute and derive experiment artifact paths.

        Also sets bookkeeping attributes (e.g. name, train, notes) that only
        recount the last experiment run on the model.
        """
        for key, value in params.items():
            setattr(self, key, value)

        # Derive per-experiment artifact directories from the log name.
        if hasattr(self, 'exp_log_name'):
            artifact_root = dirname(__file__)
            run_name = self.exp_log_name.replace('_log.txt', '')

            mod_name = getattr(self, 'mod_name', None)
            if isinstance(mod_name, str) and mod_name.startswith('tmp_'):
                try:
                    run_name = parse_model_output_name(mod_name)['stem']
                except ValueError:
                    pass

            self.exp_log_path = join(artifact_root, 'log', self.exp_log_name)
            self.experiment_name = run_name
            self.models_dir = join(artifact_root, 'models', f'{self.experiment_name}_models')
            self.checkpoints_dir = join(self.models_dir, 'checkpoints')
            self.diagnostics_dir = join(artifact_root, 'results', 'diagnostics', f'{self.experiment_name}_diagnostics')
            self.plots_dir = join(artifact_root, 'results', 'plots', f'{self.experiment_name}_plots')

            makedirs(self.models_dir, exist_ok=True)
            makedirs(self.checkpoints_dir, exist_ok=True)
            makedirs(self.diagnostics_dir, exist_ok=True)
            makedirs(self.plots_dir, exist_ok=True)

    def verify_required_config_keys_present(self) -> None:
        """Raise if any REQUIRED_CONFIG_KEYS attribute is missing."""
        missing_keys = [key for key in self.REQUIRED_CONFIG_KEYS if not hasattr(self, key)]
        if missing_keys:
            raise ValueError(
                "Config missing required keys: " + ', '.join(missing_keys)
            )

    def validate_recurrent_stub_controls(self) -> None:
        """Reject prior-cost settings rPCC does not implement (prior costs are sPCC-only per Rogers 2026)."""
        stub_message = (
            'rPCC has no prior cost drivers in its update terms. Currently sPCC only, per Rogers 2026. '
            'Prior distributions do determine inits of r, U, V but there are no extant prior costs in rPCC'
        )

        if self.r_prior_cost is not None:
            raise ValueError(stub_message + '.')
        if self.U_prior_cost is not None:
            raise ValueError(stub_message + '.')
        if self.r_prior_cost_denominator is not None:
            raise ValueError(stub_message + ', so no denominators necessary.')
        if self.U_prior_cost_denominator is not None:
            raise ValueError(stub_message + ', so no denominators necessary.')

    def load_prior_dist_settings(self) -> None:
        """Parse the r/U prior-distribution config tuples into names, labels, and param dicts."""
        def read_prior_dist_settings(prior_setting: tuple[Any, ...]) -> tuple[str, str, dict[str, float]]:
            """Return (name, human-readable label, sampler params) for one prior setting."""
            prior_name = prior_setting[0]
            if prior_name == 'pseudo_sparse_kurtotic':
                label = prior_name
                params = {}
            elif prior_name == 'uniform_random':
                shift = float(prior_setting[1])
                label = f"uniform_random(shift={shift})"
                params = {'shift': shift}
            else:
                mean = float(prior_setting[1])
                scale = float(prior_setting[2])
                label = f"{prior_name}(mean={mean}, scale={scale})"
                params = {'mean': mean, 'scale': scale}
            return prior_name, label, params
        self.r_prior_dist_name, self.r_prior_dist_label, self.r_prior_dist_params = read_prior_dist_settings(self.r_prior_dist)
        self.U_prior_dist_name, self.U_prior_dist_label, self.U_prior_dist_params = read_prior_dist_settings(self.U_prior_dist)
        self.r_prior_cost_name = self.r_prior_cost
        self.U_prior_cost_name = self.U_prior_cost

    def uses_reshaped_first_to_second_connection(self) -> bool:
        """Return True when the Li-style reshaped (flattened-r1) first-to-second connection is configured."""
        return self.first_to_second_lyr_connection_architecture == 'reshaped'

    def is_static_model(self) -> bool:
        """Return True for sPCC models."""
        return self.model_type == 'static'

    def static_input_is_tiled(self) -> bool:
        """Return True when static inputs are split into tiles (ntiles_per_input is set)."""
        return getattr(self, 'ntiles_per_input', None) is not None

    def validate_static_input_controls(self) -> None:
        """Check flat_input and ntiles_per_input settings for static models."""
        if not self.is_static_model():
            return

        if not isinstance(self.flat_input, bool):
            raise ValueError(
                'flat_input must be explicitly set to True or False for static models.'
            )

        ntiles_per_input = self.ntiles_per_input
        if ntiles_per_input is None:
            return

        if isinstance(ntiles_per_input, bool) or not isinstance(ntiles_per_input, int) or ntiles_per_input < 1:
            raise ValueError(
                'ntiles_per_input must be None or a positive integer for static models.'
            )

    def receptive_fields_enabled(self) -> bool:
        """Return True when receptive-field plotting is configured on."""
        return self.receptive_fields_plot_spec()['mode'] is not None

    def receptive_fields_plot_spec(self) -> dict[str, Any]:
        """Return the normalized plot_receptive_fields spec dict."""
        return normalize_plot_receptive_fields_setting(getattr(self, 'plot_receptive_fields', False))

    def softmaxed_activation_plots_enabled(self) -> bool:
        """Return True when softmaxed-activation diagnostics plotting is configured on."""
        return bool(getattr(self, 'plot_softmaxed_activations', False))

    def validate_architecture_controls(self) -> None:
        """Check architecture / first-to-second-connection settings for consistency."""
        self.validate_static_input_controls()

        if self.architecture == 'expand_first_lyr_Li':
            raise ValueError(
                "'expand_first_lyr_Li' is deprecated. Use architecture='expand_first_lyr' "
                "with first_to_second_lyr_connection_architecture='reshaped' for Li-equivalent behavior."
            )

        if self.architecture not in ('flat_hidden_lyrs', 'expand_first_lyr'):
            raise ValueError(f'Unsupported architecture: {self.architecture}')

        if not self.is_static_model() and self.architecture != 'flat_hidden_lyrs':
            raise ValueError("rPCC requires architecture='flat_hidden_lyrs'.")

        valid_connections = {'structured', 'reshaped'}
        if self.first_to_second_lyr_connection_architecture not in valid_connections:
            raise ValueError(
                "first_to_second_lyr_connection_architecture must be one of "
                "{'structured', 'reshaped'}"
            )

        if self.architecture == 'flat_hidden_lyrs':
            if self.first_to_second_lyr_connection_architecture != 'structured':
                raise ValueError(
                    "flat_hidden_lyrs requires first_to_second_lyr_connection_architecture='structured'."
                )

        if self.uses_reshaped_first_to_second_connection() and not (self.static_input_is_tiled() and self.flat_input):
            raise ValueError(
                "first_to_second_lyr_connection_architecture='reshaped' requires "
                "ntiles_per_input to be set and flat_input=True."
            )

    def layer_sizes(self) -> list[int]:
        """Return hidden layer sizes plus the top layer size."""
        return list(self.hidden_lyr_sizes) + [self.top_lyr_size]

    def validate_layer_config(self) -> None:
        """Check layer counts/sizes and reject unsupported layer-architecture combinations."""
        if self.num_layers < 1:
            raise ValueError('num_layers must be at least 1.')

        expected_hidden_layers = self.num_layers - 1
        if len(self.hidden_lyr_sizes) != expected_hidden_layers:
            raise ValueError(
                f'hidden_lyr_sizes must contain {expected_hidden_layers} entries for '
                f'num_layers={self.num_layers}, got {len(self.hidden_lyr_sizes)}.'
            )

        if (
            self.is_static_model()
            and self.num_layers == 1
            and self.architecture == 'expand_first_lyr'
            and self.static_input_is_tiled()
            and self.classif_method in ('c1', 'c2')
        ):
            raise ValueError(
                '`n == 1` + tiled input + `expand_first_lyr` makes the top layer non-vectorial. '
                'C1 fails because the top layer is no longer a single class-logit vector. '
                'C2 also fails as currently written because `Uo` is a vector-top-layer classifier, '
                'not a tiled/spatial classifier. Use n >= 2, untile the input, or make architecture=\'flat_hidden_layers\'.'
            )

    def require_dict_keys(self, dict_name: str, required_keys: Iterable[Any]) -> None:
        """Raise if the named dict attribute is missing any required key."""
        values = getattr(self, dict_name)
        missing = [key for key in required_keys if key not in values]
        if missing:
            raise ValueError(f'{dict_name} missing required key(s): {missing}')

    def validate_config_key_policy(self) -> None:
        """Run the module-level config policy check against this model's attributes."""
        validate_config_key_policy(
            self.__dict__,
            config_source_path=getattr(self, '_config_source_path', None),
            config_key_lines=getattr(self, '_config_key_lines', None),
            config_entries=getattr(self, '_config_entries', None),
        )

    def configure_shared_model(self) -> None:
        """Run all shared validation, then bind activation (f, F_mult) and prior settings."""
        self.verify_required_config_keys_present()
        self.validate_architecture_controls()
        self.validate_layer_config()
        self.validate_config_key_policy()
        self.validate_model_layer_config()
        self.f, self.F_mult = self.act_fxn_dict[self.activ_func]
        self.load_prior_dist_settings()

    @abstractmethod
    def validate_model_layer_config(self) -> None:
        """Validate dictionaries owned by the concrete model type."""
            
    def set_Idiff_tdot_dims(self, input_size: tuple[int, ...]) -> None:
        """Set tensordot dims (all input dims) for the Idiff^2 term in rep_cost(); all architectures."""
        ndims_input = len(input_size)
        range_ndims_input = range(ndims_input)
        self.Idiff_tdot_dims = list(range_ndims_input)
        
    def set_L1diff_tdot_dims(self, L1_size: tuple[int, ...]) -> None:
        """Set tensordot dims (all r1 dims) for the L1diff^2 term in rep_cost(); all architectures."""
        ndims_input = len(L1_size)
        range_ndims_input = range(ndims_input)
        self.L1diff_tdot_dims = list(range_ndims_input)
            
    def einsum_letters(self, count: int, start: int = 0) -> str:
        """Return `count` einsum subscript letters starting at alphabet offset `start`."""
        return 'abcdefghijklmnopqrstuvwxyz'[start:start + count]

    def first_lyr_shared_tile_ndims(self) -> int:
        """Return the number of leading tile-grid dims shared between the input and r1.

        expand_first_lyr keeps r1's leading tile dims; a vector r1 shares none.
        """
        return self.r[1].ndim - 1

    def set_U1_einsum_arg(self) -> None:
        """Set the einsum spec for the generative first-layer product U1.r1; 'expand_first_lyr' only.

        U1 = input_dims + (layer,), r1 = shared_tile_dims + (layer,) -> input_dims.
        Built generically from operand dimensionality, e.g. tiled flat
        'ab c, a c -> ab', tiled nonflat 'abcd e, ab e -> abcd',
        untiled flat 'a b, b -> a', untiled nonflat 'ab c, c -> ab'.
        """
        input_ndims = len(self.input_shape)
        shared_ndims = self.first_lyr_shared_tile_ndims()
        shared = self.einsum_letters(shared_ndims)
        patch = self.einsum_letters(input_ndims - shared_ndims, start=shared_ndims)
        layer = self.einsum_letters(1, start=input_ndims)
        self.U1_einsum_arg = f'{shared}{patch}{layer},{shared}{layer}->{shared}{patch}'
            
    def set_U1T_dims(self, U1_size: tuple[int, ...]) -> None:
        """Set transpose dims that move U1's layer (last) axis first; reshaped connection uses a fixed 3-axis permutation."""
        if self.architecture == 'expand_first_lyr' and self.uses_reshaped_first_to_second_connection():
            self.U1T_dims = (0, 2, 1)
            return

        ndims_U1 = len(U1_size)
        range_ndims_U1 = range(ndims_U1)
        last_dim_id_U1 = range_ndims_U1[-1]
        nonlast_dim_ids_U1 = range_ndims_U1[:-1]
        transpose_dims_U1 = tuple([last_dim_id_U1] + list(nonlast_dim_ids_U1))
        self.U1T_dims = transpose_dims_U1

    def set_U1T_tdot_dims(self, U1_size: tuple[int, ...]) -> None:
        """Set tensordot dims (all non-first U1 axes) for the U1T.Idiff product; 'flat_hidden_lyrs' only."""
        ndims_U1 = len(U1_size)
        range_ndims_U1 = range(ndims_U1)
        nonfirst_dim_ids_U1 = range_ndims_U1[1:]
        self.U1T_tdot_dims = list(nonfirst_dim_ids_U1)
            
    def set_U1T_einsum_arg(self) -> None:
        """Set the einsum spec for the U1T.Idiff product; 'expand_first_lyr' only.

        U1T = (layer,) + input_dims (transposed U1), Idiff = input_dims -> r1 dims.
        """
        input_ndims = len(self.input_shape)
        shared_ndims = self.first_lyr_shared_tile_ndims()
        shared = self.einsum_letters(shared_ndims)
        patch = self.einsum_letters(input_ndims - shared_ndims, start=shared_ndims)
        layer = self.einsum_letters(1, start=input_ndims)
        self.U1T_einsum_arg = f'{layer}{shared}{patch},{shared}{patch}->{shared}{layer}'
            
    def set_U2T_dims(self, U2_size: tuple[int, ...]) -> None:
        """Set transpose dims that move U2's last axis first (same protocol as U1T); all architectures."""
        ndims_U2 = len(U2_size)
        range_ndims_U2 = range(ndims_U2)
        last_dim_id_U2 = range_ndims_U2[-1]
        nonlast_dim_ids_U2 = range_ndims_U2[:-1]
        transpose_dims_U2 = tuple([last_dim_id_U2] + list(nonlast_dim_ids_U2))
        self.U2T_dims = transpose_dims_U2
        
    def set_U2T_einsum_arg(self) -> None:
        """Set the einsum spec for U2T.L1diff: U2T = (r2,) + r1_dims, L1diff = r1_dims -> (r2,); 'expand_first_lyr' only."""
        r1_sub = self.einsum_letters(self.r[1].ndim, start=1)
        self.U2T_einsum_arg = f'a{r1_sub},{r1_sub}->a'
            
    def set_Idiff_einsum_arg(self) -> None:
        """Set the einsum spec for the Idiff ('outer') r1 product used in U1 updates.

        Idiff = input_dims, r1 = shared_tile_dims + (layer,) -> input_dims + (layer,).
        flat_hidden_lyrs has a vector r1, so shared_tile_dims is empty and this
        reduces to a plain outer product for any input dimensionality.
        """
        input_ndims = len(self.input_shape)
        shared_ndims = self.first_lyr_shared_tile_ndims()
        shared = self.einsum_letters(shared_ndims)
        patch = self.einsum_letters(input_ndims - shared_ndims, start=shared_ndims)
        layer = self.einsum_letters(1, start=input_ndims)
        self.Idiff_einsum_arg = f'{shared}{patch},{shared}{layer}->{shared}{patch}{layer}'
            
    def set_L1diff_einsum_arg(self) -> None:
        """Set the einsum spec for the L1diff ('outer') r2 product: L1diff = r1_dims, r2 = (layer2,); 'expand_first_lyr' only."""
        r1_sub = self.einsum_letters(self.r[1].ndim)
        r2_sub = self.einsum_letters(1, start=self.r[1].ndim)
        self.L1diff_einsum_arg = f'{r1_sub},{r2_sub}->{r1_sub}{r2_sub}'
    
    def configure_static_model(self) -> None:
        """Validate config, then initialize static r/U state, einsum plumbing, and diagnostics.

        Weight shapes: U1 is input dims plus the r1 layer size; U2 is r1 dims
        plus r2 dims (the reshaped Li-style connection flattens r1 first);
        U3..Un are (r_{i-1}, r_i) matrices.
        """
        self.configure_shared_model()

        self.g = self.prior_cost_dict[self.r_prior_cost_name]
        self.h = self.prior_cost_dict[self.U_prior_cost_name]

        tiled_input = self.static_input_is_tiled()
        flat_input = self.flat_input
        architecture = self.architecture
        input_shape = self.input_shape
        n = self.num_layers

        self.r = {}
        self.U = {}
        self.r[0] = np.zeros(input_shape)

        U1_size = None
        U2_size = None

        # Build r and U layer by layer; layer 1 shape depends on architecture and tiling.
        for i in range(1, n + 1):
            if i == 1:
                layer_1_size = self.top_lyr_size if n == 1 else self.hidden_lyr_sizes[0]
                if architecture == 'flat_hidden_lyrs':
                    self.r[1] = self.initialize_r_state(size=(layer_1_size))
                elif architecture == 'expand_first_lyr':
                    if tiled_input and flat_input:
                        self.r[1] = self.initialize_r_state(size=(self.ntiles_per_input, layer_1_size))
                    elif tiled_input and not flat_input:
                        self.r[1] = self.initialize_r_state(size=(self.input_shape[0], self.input_shape[1], layer_1_size))
                    elif not tiled_input:
                        self.r[1] = self.initialize_r_state(size=(layer_1_size))
                else:
                    raise ValueError(f'Unsupported architecture: {architecture}')

                input_shape_list = list(input_shape)
                input_shape_list.append(layer_1_size)
                U1_size = tuple(input_shape_list)
                self.U[1] = self.sample_U_prior(size=U1_size)

            elif i > 1 and i < n:
                self.r[i] = self.initialize_r_state(size=(self.hidden_lyr_sizes[i - 1]))

                if i == 2:
                    if architecture == 'expand_first_lyr' and self.uses_reshaped_first_to_second_connection():
                        U2_size = (self.r[1].shape[0] * self.r[1].shape[1], self.hidden_lyr_sizes[i - 1])
                        self.U[2] = self.sample_U_prior(size=U2_size)
                    else:
                        U2_size = (*self.r[1].shape, self.r[2].shape[0])
                        self.U[2] = self.sample_U_prior(size=U2_size)
                else:
                    Ui_size = (self.r[i - 1].shape[0], self.r[i].shape[0])
                    self.U[i] = self.sample_U_prior(size=Ui_size)

            elif i == n:
                self.r[n] = self.initialize_r_state(size=(self.top_lyr_size))
                if i == 2:
                    if architecture == 'expand_first_lyr' and self.uses_reshaped_first_to_second_connection():
                        Un_size = (np.prod(self.r[1].shape), self.r[n].shape[0])
                    else:
                        # Same shape rule as the i < n branch: U2 maps all of r1's
                        # dims (vector r1 reduces this to (r1, top) as before).
                        Un_size = (*self.r[1].shape, self.r[n].shape[0])
                    U2_size = Un_size
                else:
                    Un_size = (self.r[n - 1].shape[0], self.r[n].shape[0])
                self.U[n] = self.sample_U_prior(size=Un_size)

        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.top_lyr_size)
            self.U['o'] = self.sample_U_prior(size=Uo_size)

        self.set_Idiff_tdot_dims(input_shape)
        self.set_L1diff_tdot_dims(self.r[1].shape)
        self.set_U1_einsum_arg()
        self.set_U1T_dims(U1_size)
        if U2_size is not None:
            self.set_U2T_dims(U2_size)
        self.set_U1T_tdot_dims(U1_size)
        self.set_U1T_einsum_arg()
        if U2_size is not None:
            self.set_U2T_einsum_arg()
        self.set_Idiff_einsum_arg()
        self.set_L1diff_einsum_arg()

        self.all_lyr_sizes = self.hidden_lyr_sizes.copy()
        self.all_lyr_sizes.append(self.top_lyr_size)
        if architecture == 'expand_first_lyr':
            self.all_lyr_sizes[0] = self.r[1].shape

        self.softmax_func = partial(self.set_softmax_func, softmax_type=self.softmax_type, k=self.softmax_k)

        epoch_n = self.epoch_n
        self.Jr = [0] * (epoch_n + 1)
        self.Jc = [0] * (epoch_n + 1) if self.classification_enabled() else None
        self.accuracy = [0] * (epoch_n + 1) if self.classification_enabled() else None

        # Static code paths still reference num_imgs.
        if hasattr(self, 'num_inps') and not hasattr(self, 'num_imgs'):
            self.num_imgs = self.num_inps
        
    # Activation functions: f applies to generative U.r predictions only (never V.r).
    # F_mult(f_x, error) computes F @ error with F = diag(f'(x)), elementwise so it
    # works for any prediction shape (vector or tiled tensor architectures).
    def linear_activation(self, U_dot_r: np.ndarray) -> np.ndarray:
        """Apply the identity activation."""
        return U_dot_r

    def linear_error_gradient(self, f_x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Return the error unchanged (f' = 1 for linear f)."""
        return error

    def tanh_activation(self, U_dot_r: np.ndarray) -> np.ndarray:
        """Apply the tanh activation."""
        return np.tanh(U_dot_r)

    def tanh_error_gradient(self, f_x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Apply diag(1 - f(x)^2) to the error, reusing the already-computed f(x)."""
        return (1 - np.square(f_x)) * error
    
    # r, U or V prior functions
    def gaussian_prior_costs(self, r_or_U: np.ndarray | None = None, alph_or_lam: float | None = None) -> tuple[float, np.ndarray] | None:
        """Evaluate the Gaussian prior cost and gradient for r (with alpha) or U (with lambda).

        Returns:
            (g(x), g'(x)) — or (h(x), h'(x)) for weights — or None on overflow,
            which is trapped and logged here.
        """
        printlog = self.print_and_log
        
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            func_eval = alph_or_lam * np.square(r_or_U).sum()
            func_deriv_eval = 2 * alph_or_lam * r_or_U
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (func_eval, func_deriv_eval)
        
        except FloatingPointError as e:
            printlog(f"FloatingPointError: {e}")
            printlog(f'Overflow is checked in prior cost evaluation, and has been encountered.')
            
            if r_or_U is not None:
                printlog("r or U shape:", r_or_U.shape)
                # Create a slice object that slices the first 5 elements in each dimension
                slice_obj = tuple(slice(0, 3) for _ in range(r_or_U.ndim))
                printlog("A few elements of r_or_U:\n", r_or_U[slice_obj])
            
            return None
    
    def sparse_kurtotic_prior_costs(self, r_or_U: np.ndarray | None = None, alph_or_lam: float | None = None) -> tuple[float, np.ndarray] | None:
        """Evaluate the sparse kurtotic prior cost and gradient for r (with alpha) or U (with lambda).

        Returns:
            (g(x), g'(x)) — or (h(x), h'(x)) for weights — or None on overflow,
            which is trapped and logged here.
        """
        printlog = self.print_and_log
        
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            func_eval = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
            func_deriv_eval = (alph_or_lam * 2 *  r_or_U) / (1 + np.square(r_or_U))
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (func_eval, func_deriv_eval)
        
        except FloatingPointError as e:
            printlog(f"FloatingPointError: {e}")
            printlog(f'Overflow is checked in prior cost evaluation, and has been encountered.')
            
            if r_or_U is not None:
                printlog("r or U shape:", r_or_U.shape)
                # Create a slice object that slices the first 5 elements in each dimension
                slice_obj = tuple(slice(0, 3) for _ in range(r_or_U.ndim))
                printlog("A few elements of r_or_U:\n", r_or_U[slice_obj])
            
            return None

    def prior_dist_params_dict(self) -> dict[str, dict[str, float]]:
        """Return the sampler parameter dicts for the 'r' and 'U' prior distributions."""
        return {'r': self.r_prior_dist_params,
            'U': self.U_prior_dist_params}
        
    def pseudo_sparse_kurtotic(self, size: int | tuple[int, ...], component: str) -> np.ndarray:
        """Return zeros; `component` is unused by design to keep a uniform sampler interface."""
        return np.zeros(size)
    
    def normal_using_mean_and_scale(self, size: int | tuple[int, ...], component: str) -> np.ndarray:
        """Sample a normal distribution using the component's configured mean and scale."""
        params = self.prior_dist_params_dict()[component]
        return np.random.normal(loc=params['mean'], scale=params['scale'], size=size)

    def laplace_using_mean_and_scale(self, size: int | tuple[int, ...], component: str) -> np.ndarray:
        """Sample a Laplace distribution using the component's configured mean and scale."""
        params = self.prior_dist_params_dict()[component]
        return np.random.laplace(loc=params['mean'], scale=params['scale'], size=size)

    def uniform_using_optional_shift(self, size: tuple[int, ...], component: str) -> np.ndarray:
        """Sample uniform [0, 1) values plus the component's configured shift."""
        params = self.prior_dist_params_dict()[component]
        return np.random.rand(*size) + params['shift']

    def sample_r_prior(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw an r-state sample from the configured r prior distribution."""
        sampler = self.prior_dist_dict[self.r_prior_dist_name]
        return sampler(size=size, component='r')

    def initialize_r_state(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Initialize a layer representation from the r prior."""
        return self.sample_r_prior(size=size)

    def sample_U_prior(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw a weight sample from the configured U prior distribution."""
        sampler = self.prior_dist_dict[self.U_prior_dist_name]
        return sampler(size=size, component='U')

    def sample_V_prior(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw a V sample from the U prior distribution (inext: normal(0, 0.1) for all weights)."""
        sampler = self.prior_dist_dict[self.U_prior_dist_name]
        return sampler(size=size, component='U')
    
    def hard_set_r_prior_dist(self) -> None:
        """Cache one prior sample per layer size so per-input resets reuse it.

        Timesaver: reused samples are not truly independent random draws.
        """
        self.r_dists_hard = {}
        n = self.num_layers
        for i in range (1, n + 1):
            lyr_size = self.all_lyr_sizes[i - 1]
            self.r_dists_hard[lyr_size] = self.sample_r_prior(size=lyr_size)

    def configured_r_reset_mode(self) -> str:
        """Return the validated r_reset_mode setting."""
        mode = getattr(self, 'r_reset_mode', 'reset_from_single_sample')
        valid_modes = {'reset_from_single_sample', 'reset_with_continual_resampling'}
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported r_reset_mode '{mode}'. Expected one of {sorted(valid_modes)}."
            )
        return mode

    def static_reset_prior_sampler(self) -> Callable[..., np.ndarray]:
        """Return the r-reset sampler for the configured r_reset_mode."""
        mode = self.configured_r_reset_mode()
        if mode == 'reset_from_single_sample':
            self.hard_set_r_prior_dist()
            return self.load_hard_r_prior_dist
        return self.sample_r_prior
    
    def load_hard_r_prior_dist(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Return a copy of the cached prior sample so per-sample resets do not alias the cache."""
        return np.array(self.r_dists_hard[size], copy=True)
    
    def softmax(self, vector: np.ndarray, k: float = 1) -> np.ndarray:
        """Compute softmax(k * vector)."""
        exp_vector = np.exp(k * vector)
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def stable_softmax(self, vector: np.ndarray, k: float = 1) -> np.ndarray:
        """Compute softmax(k * vector) after max-shifting for numerical stability."""
        shift_vector = vector - np.max(vector)
        exp_vector = np.exp(k * shift_vector)
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def set_softmax_func(self, softmax_type: str, vector: np.ndarray, k: float) -> np.ndarray:
        """Dispatch to the configured ('normal' or 'stable') softmax implementation."""
        return self.softmax_dict[softmax_type](vector=vector, k=k)
    
    def reset_rs_gteq1(self, all_lyr_sizes: list[int | tuple[int, ...]], prior_dist: Callable[..., np.ndarray]) -> None:
        """Re-sample r[1..n] from the prior; r[0] (the input) is untouched."""
        n = self.num_layers
        for i in range(1, n + 1):
            self.r[i] = prior_dist(size=all_lyr_sizes[i - 1])

    def update_method_rW_instdeltas_niters(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Apply the combined r+weight instantaneous-delta update niters times."""
        rW_instdeltas_updates = component_updates[0]
        
        for _ in range(niters):
            rW_instdeltas_updates(label)
        
    def update_method_r_niters_W(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Update r for niters iterations, then apply each weight update once.

        Rogers/Brown default niters: 100.
        """
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        for _ in range(niters):
            r_updates(label)
        for w in range_num_weight_updates:
            # For as many weight sets are there are to update, update them.
            weight_updates[w](label)

    def update_method_r_eq_W(self, stop_criterion: float, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Update r until each layer's percent change falls below stop_criterion, then update weights."""
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        num_layers = self.num_layers
        n = num_layers
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, n + 1)]
        diffs = [float('inf')] * n  # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, n + 1)}  # Copy all vectors to avoid reference issues
            r_updates(label)
            
            for i in range(1, n + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change
        
        for w in range_num_weight_updates:
            # For as many weight sets are there are to update, update them.
            weight_updates[w](label) 
            
    def update_method_r_instdeltas_niters(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Apply the r-only instantaneous-delta update niters times."""
        r_updates = component_updates[0]
        
        for _ in range(niters):
            r_updates(label)

    def update_method_r_niters(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Update r only, for niters iterations (Li default: 30; Rogers/Brown default: 100)."""
        r_updates = component_updates[0]
        
        for _ in range(niters):
            r_updates(label)

    def update_method_r_eq(self, stop_criterion: float, label: np.ndarray, component_updates: list[Callable[..., None]]) -> None:
        """Update r until each layer's percent change falls below stop_criterion (Rogers/Brown default: 0.05)."""
        r_updates = component_updates[0]
        
        num_layers = self.num_layers
        n = num_layers
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, n + 1)]
        diffs = [float('inf')] * n # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, n + 1)}  # Copy all vectors to avoid reference issues
            r_updates(label)
            
            for i in range(1, n + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change

    # Prints and sends to log file
    def print_and_log(self, *args: Any, **kwargs: Any) -> None:
        """Print to the terminal and append the same line to the experiment log file."""
        print(*args, **kwargs)
        exp_log_path = getattr(self, 'exp_log_path', None)
        if exp_log_path is None:
            exp_log_name = getattr(self, 'exp_log_name', None)
            if exp_log_name is None:
                return
            exp_log_path = join(dirname(__file__), 'log', exp_log_name)
        if not exists(exp_log_path):
            raise FileNotFoundError(f"Log file {exp_log_path} not found.")
        with open(exp_log_path, "a") as f:
            print(*args, **kwargs, file=f)

    def format_elapsed(self, start_dt: datetime, end_dt: datetime | None = None) -> str:
        """Format the elapsed time between two datetimes (end defaults to now)."""
        if end_dt is None:
            end_dt = datetime.now()
        return str(end_dt - start_dt)

    def num_metric_samples(self) -> int | None:
        """Return the sample count used in metric denominators, or None if unknown."""
        if hasattr(self, 'num_inps'):
            return int(self.num_inps)
        if hasattr(self, 'num_imgs'):
            return int(self.num_imgs)
        return None

    def classification_enabled(self) -> bool:
        """Return True when a classification method (c1/c2) is configured."""
        return self.classif_method in ('c1', 'c2')

    def format_epoch_metrics(self, Jr: float, Jc: float | None = None, accuracy: float | None = None) -> str:
        """Format the per-epoch metric line, omitting Jc/accuracy when classification is disabled."""
        if self.classification_enabled():
            return f'Jr: {Jr}, Jc: {Jc}, Accuracy: {accuracy}'
        return f'Jr: {Jr}, Classification: disabled'

    def gain_percent(self, start_value: float, end_value: float, epsilon: float = 1e-6) -> float:
        """Return the percent decrease from start to end (positive = improvement)."""
        denominator = abs(start_value) + epsilon
        return ((start_value - end_value) / denominator) * 100

    def late_window_slope(self, values: list[float], epoch: int, window: int) -> tuple[float, int]:
        """Return (per-epoch slope over the trailing window, actual window length used)."""
        if epoch <= 0:
            return 0.0, 0
        start_epoch = max(0, epoch - window)
        actual_window = epoch - start_epoch
        if actual_window == 0:
            return 0.0, 0
        return (values[epoch] - values[start_epoch]) / actual_window, actual_window

    def format_kpi_value(self, value: float, decimals: int = 6) -> str:
        """Format a KPI value to a fixed number of decimals."""
        return f'{float(value):.{decimals}f}'

    def compute_final_kpis(self, epoch: int) -> dict[str, Any]:
        """Compute end-of-training KPI summary lines from the Jr/Jc/accuracy histories.

        Returns:
            Dict with 'epoch', 'header_line', and 'lines' (list of formatted strings).
        """
        jr_values = self.Jr
        jc_values = self.Jc
        accuracy_values = self.accuracy
        num_samples = self.num_metric_samples()
        late_window = min(max(5, epoch // 10), 20) if epoch > 0 else 0

        # Start/final/best values and epochs.
        jr0 = jr_values[0]
        jc0 = jc_values[0]
        acc0 = accuracy_values[0]
        jr_final = jr_values[epoch]
        jc_final = jc_values[epoch]
        acc_final = accuracy_values[epoch]

        jr_best_epoch = int(np.argmin(jr_values[:epoch + 1]))
        jc_best_epoch = int(np.argmin(jc_values[:epoch + 1]))
        acc_best_epoch = int(np.argmax(accuracy_values[:epoch + 1]))

        jr_best = jr_values[jr_best_epoch]
        jc_best = jc_values[jc_best_epoch]
        acc_best = accuracy_values[acc_best_epoch]
        jc_max = float(np.max(jc_values[:epoch + 1]))

        # Gains, rebounds, and late-window slopes.
        jr_gain_final = self.gain_percent(jr0, jr_final)
        jc_gain_final = self.gain_percent(jc0, jc_final)
        jr_best_gain = self.gain_percent(jr0, jr_best)
        jc_best_gain = self.gain_percent(jc0, jc_best)
        jc_rebound = ((jc_final - jc_best) / (abs(jc0) + 1e-6)) * 100
        jc_max_blowup = ((jc_max - jc0) / (abs(jc0) + 1e-6)) * 100
        acc_delta_points = (acc_final - acc0) * 100
        acc_best_delta_points = (acc_best - acc0) * 100

        jr_late_slope, late_window_used = self.late_window_slope(jr_values, epoch, late_window)
        jc_late_slope, _ = self.late_window_slope(jc_values, epoch, late_window)
        acc_late_slope, _ = self.late_window_slope(accuracy_values, epoch, late_window)

        if num_samples is None:
            start_line = (
                f'Start: Jr={self.format_kpi_value(jr0)}, Jc={self.format_kpi_value(jc0)}, '
                f'Accuracy={self.format_kpi_value(acc0)}'
            )
            final_line = (
                f'Final: Jr={self.format_kpi_value(jr_final)}, Jc={self.format_kpi_value(jc_final)}, '
                f'Accuracy={self.format_kpi_value(acc_final)}'
            )
            best_accuracy_line = f'Best Accuracy: {self.format_kpi_value(acc_best)} at epoch {acc_best_epoch}'
            accuracy_delta_line = f'Accuracy delta: {self.format_kpi_value(acc_delta_points)} points'
            best_accuracy_delta_line = f'Best Accuracy delta: {self.format_kpi_value(acc_best_delta_points)} points'
        else:
            start_correct = int(round(acc0 * num_samples))
            final_correct = int(round(acc_final * num_samples))
            best_correct = int(round(acc_best * num_samples))
            start_line = (
                f'Start: Jr={self.format_kpi_value(jr0)}, Jc={self.format_kpi_value(jc0)}, '
                f'Accuracy={self.format_kpi_value(acc0)} '
                f'({start_correct}/{num_samples})'
            )
            final_line = (
                f'Final: Jr={self.format_kpi_value(jr_final)}, Jc={self.format_kpi_value(jc_final)}, '
                f'Accuracy={self.format_kpi_value(acc_final)} '
                f'({final_correct}/{num_samples})'
            )
            best_accuracy_line = (
                f'Best Accuracy: {self.format_kpi_value(acc_best)} ({best_correct}/{num_samples}) '
                f'at epoch {acc_best_epoch}'
            )
            accuracy_delta_line = (
                f'Accuracy delta: {self.format_kpi_value(acc_delta_points)} points '
                f'({final_correct - start_correct:+d} correct)'
            )
            best_accuracy_delta_line = (
                f'Best Accuracy delta: {self.format_kpi_value(acc_best_delta_points)} points '
                f'({best_correct - start_correct:+d} correct)'
            )

        final_jr_gain_line = f'Final Jr gain: {self.format_kpi_value(jr_gain_final)}%'
        final_jc_gain_line = f'Final Jc gain: {self.format_kpi_value(jc_gain_final)}%'
        best_jr_line = (
            f'Best Jr: {self.format_kpi_value(jr_best)} at epoch {jr_best_epoch} '
            f'(gain {self.format_kpi_value(jr_best_gain)}%)'
        )
        best_jc_line = (
            f'Best Jc: {self.format_kpi_value(jc_best)} at epoch {jc_best_epoch} '
            f'(gain {self.format_kpi_value(jc_best_gain)}%)'
        )
        jc_rebound_line = f'Jc rebound from best to final: {self.format_kpi_value(jc_rebound)}% of Jc0'
        jc_max_blowup_line = f'Jc max blowup vs start: {self.format_kpi_value(jc_max_blowup)}% of Jc0'
        late_window_slopes_line = (
            f'Late-window slopes ({late_window_used} epochs): '
            f'Jr={self.format_kpi_value(jr_late_slope)}, '
            f'Jc={self.format_kpi_value(jc_late_slope)}, '
            f'Accuracy={self.format_kpi_value(acc_late_slope)}'
        )

        prediction_diversity_line = None
        if hasattr(self, 'prediction_unique_count') and epoch < len(self.prediction_unique_count):
            unique_count = self.prediction_unique_count[epoch]
            dominant_class = self.prediction_dominant_class[epoch]
            dominant_count = self.prediction_dominant_count[epoch]
            if dominant_class is None:
                prediction_diversity_line = 'Prediction diversity: unavailable'
            else:
                prediction_diversity_line = (
                    f'Prediction diversity: {unique_count}/{self.num_classes} classes; '
                    f'dominant={dominant_class} ({dominant_count}/{num_samples})'
                )

        lines = [
            start_line,
            final_line,
            best_accuracy_line,
            accuracy_delta_line,
            best_accuracy_delta_line,
            final_jr_gain_line,
            final_jc_gain_line,
            best_jr_line,
            best_jc_line,
            jc_rebound_line,
            jc_max_blowup_line,
            late_window_slopes_line,
        ]
        if prediction_diversity_line is not None:
            lines.append(prediction_diversity_line)

        return {
            'epoch': epoch,
            'header_line': f'Final diagnostics over {epoch} epochs:',
            'lines': lines,
        }

    def log_final_kpis(self, epoch: int) -> None:
        """Log the final KPI summary (static models only; skips detail when classification is off)."""
        if self.model_type == 'recurrent':
            return
        if not self.classification_enabled():
            self.print_and_log(
                f'Final diagnostics over {epoch} epochs: Jr={self.format_kpi_value(self.Jr[epoch])}; '
                'classification disabled, so Jc and accuracy were not calculated.'
            )
            return
        printlog = self.print_and_log

        kpis = self.compute_final_kpis(epoch)
        printlog(kpis['header_line'])
        for line in kpis['lines']:
            printlog(line)

    def log_training_event(self, message: str) -> None:
        """Log a one-line training event."""
        self.print_and_log(message)
    
    def train(self, X: np.ndarray, Y: np.ndarray, save_checkpoint: dict[str, Any] | None = None, load_checkpoint: dict[str, Any] | None = None, plot: bool = False) -> None:
        """Train the static model over epoch_n epochs with per-epoch evaluation and checkpointing.

        Args:
            X: Training inputs (reshaped to (num_inps, ntiles_per_input, patch) when tiled).
            Y: One-hot labels.
            save_checkpoint: {'save_every': N} or {'fraction': frac or (num, den)} checkpoint policy.
            load_checkpoint: Resume marker; when set, training resumes from self.load_epoch.
            plot: Whether to render diagnostics plots at each checkpoint save.
        """
        printlog = self.print_and_log
        if not self.softmaxed_activation_plots_enabled():
            self.softmaxed_activations_over_all_correct_inputs = None
            self.average_softmaxed_activations_over_correct_inputs = None

        # Log initial shapes and prior previews.
        printlog('\n\n')
        printlog('shapes')
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} shape: {self.r[i].shape}')
            if i > 0:
                # Check if the array is 3D or 5D
                printlog(f'U{i} shape: {self.U[i].shape}')
        if self.classif_method == 'c2':
            printlog(f'Uo shape: {self.U["o"].shape}')
        printlog('\n')
            
        printlog(
            f'r_prior_dist: {self.r_prior_dist_label}, r_prior_cost: {self.r_prior_cost_name}, '
            f'U_prior_dist: {self.U_prior_dist_label}, U_prior_cost: {self.U_prior_cost_name} init preview:'
        )
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} first 3: {self.r[i][:3]}')
            if i > 0:
                # Check if the array is 3D or 5D
                if self.U[i].ndim == 3:
                    printlog(f'U{i} first 3x3x3: {self.U[i][:3, :3, :3]}')
                elif self.U[i].ndim == 5:
                    printlog(f'U{i} first 3x3x3x3x3: {self.U[i][:3, :3, :3, :3, :3]}')
                else:
                    printlog(f'U{i} first 3x3: {self.U[i][:3, :3]}')
        if self.classif_method == 'c2':
            printlog(f'Uo first 3x3: {self.U["o"][:3, :3]}')
        printlog('\n')

        epoch_n = self.epoch_n
        num_inps = self.num_imgs
        ntiles_per_input = self.ntiles_per_input
        # Reshape flat tiled X to (num_inps, ntiles_per_input, patch).
        printlog('original X shape:', X.shape)
        if ntiles_per_input is not None:
            X = X.reshape(num_inps, ntiles_per_input, -1)
            printlog('test: reshaping X into num inps, num tiles per image, flattened tile')
        else:
            printlog('test: keeping X unreshaped because ntiles_per_input=None')

        printlog('Train init:')
        printlog('X shape:', X.shape)
        printlog('Y shape:', Y.shape)
        
        
        prior_dist = self.static_reset_prior_sampler()
        printlog(f'r_reset_mode: {self.configured_r_reset_mode()}')

        # Methods
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_all_components = partial(self.update_method_dict[update_method_name], update_method_number)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        rep_cost = self.rep_cost
        
        classification_enabled = self.classification_enabled()
        classif_cost = self.classif_cost if classification_enabled else None
        
        evaluate = partial(self.evaluate, update_method_name=update_method_name, update_method_number=update_method_number, plot=None) if classification_enabled else None
        
        # Checkpointing
        if 'save_every' in save_checkpoint:
            # If N
            checkpoint = save_checkpoint['save_every'] 
        elif 'fraction' in save_checkpoint:
            fraction_value = save_checkpoint['fraction']
            # If frac tuple
            if isinstance(fraction_value, tuple):
                numerator, denominator = fraction_value
                checkpoint = (numerator/denominator) * epoch_n
            # If it's a decimal
            else:
                checkpoint = fraction_value * epoch_n
            # Round up to the nearest whole number
            checkpoint = np.ceil(checkpoint)
        else:
            checkpoint = None  # Default case if neither key is found
        if checkpoint is not None:
            printlog(f'\nSaving checkpoints.')
            printlog(f'Checkpoint method: {save_checkpoint}', '\n', 'saving every', checkpoint, '\n')
            
        # If loading
        if load_checkpoint is not None:
            printlog('Loaded a checkpoint for training.')
            printlog(f'Load checkpoint method: {load_checkpoint}')
            start_epoch = self.load_epoch
            printlog('Starting epoch is:', start_epoch,'\n')
        else:
            start_epoch = 0
            
        # if new max epoch (loaded, new goal)
        config_epoch_n = getattr(self, 'config_epoch_n', None)
        if config_epoch_n:
            # this will only exist in the loaded checkpoint case.
            # it will either be the same as the checkpoint epoch_n
            if config_epoch_n == epoch_n:
                pass
            # or it will be greater (pushing the number of epochs now past the original experiment)
            elif config_epoch_n > epoch_n:
                printlog(f"Requested config epoch_n (max epochs): {config_epoch_n} greater than \nyour checkpoint's epoch_n: {epoch_n}.\nRaising model.epoch_n (max epochs) to: {config_epoch_n}")
                # Calculate the number of zeros to add to evaluation lists
                num_zeros_to_add = config_epoch_n - epoch_n
                # Extend each list with the calculated number of zeros
                self.Jr.extend([0] * num_zeros_to_add)
                if classification_enabled:
                    self.Jc.extend([0] * num_zeros_to_add)
                    self.accuracy.extend([0] * num_zeros_to_add)
                self.epoch_n = config_epoch_n
                epoch_n = self.epoch_n
                
            # or less, probably accidentally
            else:
                raise ValueError(f'You have loaded a model checkpoint originally set to train to a greater number of epochs: {epoch_n}\n'+ \
                    f'than has now been requested in your config.txt: {config_epoch_n}.\nBoost epoch_n in config greater than {epoch_n} before training.')
                
        model_type = self.model_type
        self.num_ts = 98
        num_ts = self.num_ts

        # Epoch '0' evaluation (pre-training, or if checkpoint has been loaded, pre-additional-training)
        Jr0 = 0
        Jc0 = 0 if classification_enabled else None
        accuracy = 0 if classification_enabled else None
        printlog('\n')
        printlog(f'Epoch: {start_epoch}')
        for inp in range(num_inps):
            reset_rs_gteq1(prior_dist=prior_dist)
            input = X[inp]
            label = Y[inp]
            self.r[0] = input
            eval_label = np.zeros_like(label)
            update_non_weight_components(label=eval_label)
            Jr0 += rep_cost()
            if classification_enabled:
                Jc0 += classif_cost(label)
        if classification_enabled:
            accuracy += evaluate(X, Y)
        printlog(self.format_epoch_metrics(Jr0, Jc0, accuracy))
        self.Jr[start_epoch] = Jr0
        if classification_enabled:
            self.Jc[start_epoch] = Jc0
            self.accuracy[start_epoch] = accuracy
        
        # Training
        t_start_train = datetime.now()
        printlog('Training...')
        self.log_training_event(
            f'Training started at {t_start_train.strftime("%Y-%m-%d %H:%M:%S")}'
        )
        # Resume-aware: train only the epochs remaining until epoch_n
        # (fresh runs have start_epoch=0, so behavior there is unchanged).
        for e in range(epoch_n - start_epoch):
            epoch = e + 1 + start_epoch
            printlog(f'Epoch {epoch}')
            t_start_epoch = datetime.now()
            Jre = 0
            Jce = 0 if classification_enabled else None
            accuracy = 0 if classification_enabled else None
            # Shuffle X, Y
            shuffle_indices = np.random.permutation(num_inps)
            X_shuff = X[shuffle_indices]
            Y_shuff = Y[shuffle_indices]
            for inp in range(num_inps):
                reset_rs_gteq1(prior_dist=prior_dist)
                input = X_shuff[inp]
                label = Y_shuff[inp]
                self.r[0] = input
                update_all_components(label=label)
                Jre += rep_cost()
                if classification_enabled:
                    Jce += classif_cost(label)
            printlog(f'eval {epoch}')
            if classification_enabled:
                accuracy += evaluate(X, Y)
            self.Jr[epoch] = Jre
            if classification_enabled:
                self.Jc[epoch] = Jce
                self.accuracy[epoch] = accuracy
            
            # Checkpointing
            if checkpoint is not None and epoch % checkpoint == 0:
                checkpoint_time = datetime.now()
                self.log_training_event(
                    f'Checkpoint {epoch} reached at {checkpoint_time.strftime("%Y-%m-%d %H:%M:%S")} '
                    f'(elapsed {self.format_elapsed(t_start_train, checkpoint_time)})'
                )
                # Save checkpoint model
                checkpoint_name = self.generate_output_name(self.mod_name, epoch)
                self.save_model(output_dir=self.checkpoints_dir, output_name=checkpoint_name)
                # Save diagnostics only when a checkpoint is saved.
                self.save_diagnostics(output_dir=self.diagnostics_dir, output_name=checkpoint_name)
                if plot:
                    self.plot_safely(input_dir=self.diagnostics_dir, input_name=checkpoint_name)
                checkpoint_path = join(self.checkpoints_dir, checkpoint_name)
                printlog(f'Checkpoint model saved at epoch {epoch}:\n{checkpoint_path}')
            
            printlog(self.format_epoch_metrics(Jre, Jce, accuracy))
            t_end_epoch = datetime.now()
            printlog(f'Est. time remaining: {(t_end_epoch - t_start_epoch) * (epoch_n - epoch)}.')
            if epoch == 1:
                printlog(f'Est. tot time: {(t_end_epoch - t_start_epoch) * epoch_n}.')
            
        printlog('Training complete.')
        tot_time = t_end_epoch - t_start_train
        printlog(f'Tot time: {tot_time}.')
        if checkpoint is not None:
            final_save_time = datetime.now()
            self.log_training_event(
                f'Final save at {final_save_time.strftime("%Y-%m-%d %H:%M:%S")} '
                f'(elapsed {self.format_elapsed(t_start_train, final_save_time)})'
            )
            printlog('Saving final checkpoint model and diagnostics...')
            final_name = self.generate_output_name(self.mod_name, epoch)
            self.save_model(output_dir=self.models_dir, output_name=final_name)
            self.save_diagnostics(output_dir=self.diagnostics_dir, output_name=final_name)
            if plot:
                self.plot_safely(input_dir=self.diagnostics_dir, input_name=final_name)
        else:
            printlog('Checkpoint saving disabled; skipping model checkpoint and diagnostics saves.')
        training_end_time = datetime.now()
        self.log_training_event(
            f'Training finished at {training_end_time.strftime("%Y-%m-%d %H:%M:%S")} '
            f'(elapsed {self.format_elapsed(t_start_train, training_end_time)})'
        )
        
        self.log_final_kpis(epoch)

    def evaluate(self, X: np.ndarray, Y: np.ndarray, update_method_name: str, update_method_number: int | float, plot: bool | None = None) -> float | None:
        """Score classification accuracy over X with weights frozen (r-only updates)."""
        if not self.classification_enabled():
            return None
        
        
        prior_dist = self.static_reset_prior_sampler()
        
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        classify = self.classify
        num_inps = self.num_imgs
        
        accuracy = 0
        for inp in range(num_inps):
            reset_rs_gteq1(prior_dist=prior_dist)
            input = X[inp]
            label = Y[inp]
            self.r[0] = input
            eval_label = np.zeros_like(label)
            update_non_weight_components(label=eval_label)
            guess = classify(label)
            accuracy += guess
        accuracy /= num_inps
    
        return accuracy
    
    def predict(self, X: np.ndarray, plot: bool | None = None) -> None:
        """Placeholder; prediction is not implemented and returns None."""
        return None
                
    def generate_output_name(self, base_name: str, epoch: int) -> str:
        """Return the model output name for a given epoch."""
        return with_epoch(base_name, epoch)
    
    def save_model(self, output_dir: str, output_name: str) -> None:
        """Pickle this model to output_dir/output_name, creating the directory as needed."""
        makedirs(output_dir, exist_ok=True)
        output_path = join(output_dir, output_name)
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)

    def epoch_from_output_name(self, output_name: str) -> int:
        """Return the epoch parsed from a model output name.

        Raises:
            ValueError: If no epoch number can be parsed.
        """
        try:
            return epoch_from_name(output_name)
        except ValueError as exc:
            raise ValueError(
                f'Could not parse epoch number from output name {output_name}.'
            ) from exc

    def diagnostics_file_name(self, output_name: str) -> str:
        """Return the diagnostics artifact name for a model output name."""
        return 'diagnostics.' + output_name

    def activations_file_name(self, output_name: str) -> str:
        """Return the activations artifact name for a model output name."""
        return 'activations.' + output_name

    def epoch_artifact_dir(self, base_dir: str, output_name: str) -> str:
        """Return (and create) the per-epoch artifact subdirectory under base_dir."""
        epoch_dir = join(base_dir, str(self.epoch_from_output_name(output_name)))
        makedirs(epoch_dir, exist_ok=True)
        return epoch_dir

    def log_correct_guess_count(self, num_correct: int, num_inputs: int) -> None:
        """Log the correct-guess count, falling back to plain print when no log is configured."""
        message = f'correct guesses / num inputs: {num_correct}/{num_inputs}'
        if hasattr(self, 'exp_log_name') or hasattr(self, 'exp_log_path'):
            self.print_and_log(message)
        else:
            print(message)

    def receptive_fields_dir(self, output_name: str) -> str:
        """Return (and create) the receptive-fields plot directory for a model output name."""
        results_folder = self.epoch_artifact_dir(getattr(self, 'plots_dir', 'results/plots'), output_name)
        receptive_fields_dir = join(results_folder, 'receptive_fields')
        makedirs(receptive_fields_dir, exist_ok=True)
        return receptive_fields_dir

    def receptive_field_file_name(self, output_name: str, input_index: int) -> str:
        """Return the receptive-field plot filename for one input."""
        return f'receptive-fields-image{input_index}.{output_name}.png'

    def _resolve_dataset_pickle_path(self) -> Path:
        """Resolve dataset_train to an existing pickle path, searching the package data dir.

        Raises:
            ValueError: If no candidate path exists.
        """
        dataset_path = Path(self.dataset_train)
        if dataset_path.is_absolute() and dataset_path.exists():
            return dataset_path

        data_dir = Path(__file__).resolve().parent / 'data'
        if dataset_path.suffix:
            candidate_paths = [dataset_path, data_dir / dataset_path.name]
        else:
            candidate_paths = [
                dataset_path,
                data_dir / dataset_path.name,
                data_dir / f'{dataset_path.name}.pydb',
                data_dir / f'{dataset_path.name}.pkl',
            ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path
        attempted_names = ', '.join(path.name for path in candidate_paths)
        raise ValueError(
            f'Dataset file not found for receptive-field plotting: {self.dataset_train}. '
            f'Expected one of: {attempted_names}.'
        )

    def load_configured_dataset(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Load the configured dataset pickle; return (X, Y) or (X, None) for unlabeled payloads."""
        dataset_path = self._resolve_dataset_pickle_path()
        install_numpy_pickle_compatibility_aliases()
        with dataset_path.open('rb') as data_file:
            payload = pickle.load(data_file)

        if isinstance(payload, tuple) and len(payload) == 2:
            X, Y = payload
            return np.asarray(X), np.asarray(Y)

        return np.asarray(payload), None

    def normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Min-max scale an image to uint8 [0, 255]; constant images become zeros."""
        image = np.asarray(image, dtype=np.float32)
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        if np.isclose(vmax, vmin):
            return np.zeros_like(image, dtype=np.uint8)
        scaled = (image - vmin) / (vmax - vmin)
        return np.round(scaled * 255.0).astype(np.uint8)

    def nearest_upsample(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upsample an image by integer nearest-neighbor repetition."""
        image = np.asarray(image)
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    def factor_pair(self, value: int) -> tuple[int, int]:
        """Return the most-square (rows, cols) integer factorization of value.

        Raises:
            ValueError: For non-positive values.
        """
        value = int(value)
        if value < 1:
            raise ValueError(f'Cannot factor non-positive value {value}.')
        root = int(np.sqrt(value))
        for factor in range(root, 0, -1):
            if value % factor == 0:
                return factor, value // factor
        return 1, value

    def reshape_flat_vector_for_display(self, vector: np.ndarray) -> np.ndarray:
        """Reshape a flat vector into its most-square 2D display matrix."""
        vector = np.asarray(vector)
        rows, cols = self.factor_pair(vector.size)
        return vector.reshape(rows, cols)

    def dataset_train_key(self) -> str:
        """Return dataset_train lowercased with non-alphanumerics stripped, for dataset matching."""
        return re.sub(r'[^a-z0-9]', '', str(getattr(self, 'dataset_train', '')).lower())

    def receptive_field_metadata_path(self) -> Path:
        """Return the expected path of the dataset's receptive-field geometry metadata file."""
        dataset_path = self._resolve_dataset_pickle_path()
        metadata_dir = Path(__file__).resolve().parent / 'data' / 'metadata'
        return metadata_dir / f'{dataset_path.stem}_rf_metadata.txt'

    def load_tiled_receptive_field_geometry(self, warn_if_missing: bool = False) -> dict[str, int] | None:
        """Load tile-overlap geometry from metadata; return None (optionally warning) when the file is absent.

        Raises:
            ValueError: For untiled inputs, malformed metadata, or a tile-count mismatch.
        """
        if not self.static_input_is_tiled():
            raise ValueError('Tiled receptive-field geometry is only defined for tiled static inputs.')

        metadata_path = self.receptive_field_metadata_path()
        if not metadata_path.exists():
            if warn_if_missing:
                self.print_and_log(
                    'Warning: tiled receptive-field overlap metadata not found at '
                    f'{metadata_path}. Only tiled mosaic reconstructions will be returned.'
                )
            return None

        metadata = {}
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            for raw_line in metadata_file:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    raise ValueError(f'Invalid receptive-field metadata line in {metadata_path}: {raw_line.rstrip()}')
                key, value = line.split('=', 1)
                metadata[key.strip()] = literal_eval(value.strip())

        geometry = {
            'rf1_x': int(metadata['rf1_x']),
            'rf1_y': int(metadata['rf1_y']),
            'rf1_offset_x': int(metadata['rf1_offset_x']),
            'rf1_offset_y': int(metadata['rf1_offset_y']),
            'rf1_layout_x': int(metadata['rf1_layout_x']),
            'rf1_layout_y': int(metadata['rf1_layout_y']),
        }
        expected_tiles = geometry['rf1_layout_x'] * geometry['rf1_layout_y']
        if expected_tiles != int(self.ntiles_per_input):
            raise ValueError(
                'Resolved tiled receptive-field metadata does not match ntiles_per_input: '
                f'{expected_tiles} vs {self.ntiles_per_input}.'
            )
        return geometry

    def tiled_receptive_field_geometry(self) -> dict[str, int]:
        """Return tile-overlap geometry, raising when metadata is unavailable."""
        geometry = self.load_tiled_receptive_field_geometry(warn_if_missing=False)
        if geometry is None:
            raise ValueError(
                'Tiled receptive-field overlap metadata is unavailable. '
                'Only tiled mosaic reconstructions can be returned without metadata.'
            )
        return geometry

    def patch_to_display_matrix(self, patch: np.ndarray) -> np.ndarray:
        """Coerce a patch of any dimensionality to a 2D display matrix."""
        patch = np.asarray(patch)
        if patch.ndim == 0:
            return patch.reshape(1, 1)
        if patch.ndim == 1:
            return self.reshape_flat_vector_for_display(patch)
        if patch.ndim == 2:
            return patch
        return patch.reshape(patch.shape[0], -1)

    def tile_patch_sequence(self, patches: np.ndarray) -> np.ndarray:
        """Arrange a (num_patches, ...) sequence into a most-square 2D mosaic."""
        patches = np.asarray(patches)
        if patches.ndim == 2:
            patch_iter = [patches[index] for index in range(patches.shape[0])]
        elif patches.ndim >= 3:
            patch_iter = [patches[index] for index in range(patches.shape[0])]
        else:
            raise ValueError(f'Expected at least 2D patch sequence, got shape {patches.shape}.')

        grid_rows, grid_cols = self.factor_pair(patches.shape[0])
        patch_matrices = [self.patch_to_display_matrix(patch) for patch in patch_iter]
        patch_height, patch_width = patch_matrices[0].shape
        tiled = np.zeros((grid_rows * patch_height, grid_cols * patch_width), dtype=np.float32)

        for patch_index, patch_matrix in enumerate(patch_matrices):
            if patch_matrix.shape != (patch_height, patch_width):
                raise ValueError(
                    'All patches in a tiled sequence must share the same display shape, '
                    f'got {(patch_height, patch_width)} and {patch_matrix.shape}.'
                )
            patch_row = patch_index // grid_cols
            patch_col = patch_index % grid_cols
            row_start = patch_row * patch_height
            col_start = patch_col * patch_width
            tiled[row_start:row_start + patch_height, col_start:col_start + patch_width] = patch_matrix

        return tiled

    def tile_patch_grid(self, patch_grid: np.ndarray) -> np.ndarray:
        """Arrange a (tile_rows, tile_cols, ...) grid into a 2D mosaic."""
        patch_grid = np.asarray(patch_grid)
        if patch_grid.ndim not in (3, 4):
            raise ValueError(f'Expected 3D or 4D patch grid, got shape {patch_grid.shape}.')

        grid_rows, grid_cols = patch_grid.shape[:2]
        patch_height, patch_width = self.patch_to_display_matrix(patch_grid[0, 0]).shape
        tiled = np.zeros((grid_rows * patch_height, grid_cols * patch_width), dtype=np.float32)

        for patch_row in range(grid_rows):
            for patch_col in range(grid_cols):
                patch_matrix = self.patch_to_display_matrix(patch_grid[patch_row, patch_col])
                if patch_matrix.shape != (patch_height, patch_width):
                    raise ValueError(
                        'All patches in a tiled grid must share the same display shape, '
                        f'got {(patch_height, patch_width)} and {patch_matrix.shape}.'
                    )
                row_start = patch_row * patch_height
                col_start = patch_col * patch_width
                tiled[row_start:row_start + patch_height, col_start:col_start + patch_width] = patch_matrix

        return tiled

    def render_static_array_to_2d(self, array: np.ndarray) -> np.ndarray:
        """Render a static-model array to 2D per the tiled/flat display mode.

        Raises:
            ValueError: When the array shape does not match the configured mode.
        """
        array = np.asarray(array)
        if self.static_input_is_tiled() and self.flat_input:
            if array.ndim != 2 or array.shape[0] != self.ntiles_per_input:
                raise ValueError(
                    'Static tiled-flat rendering expects shape '
                    f'({self.ntiles_per_input}, patch_values), got {array.shape}.'
                )
            return self.tile_patch_sequence(array)
        if self.static_input_is_tiled() and not self.flat_input:
            if array.ndim != 3:
                raise ValueError(
                    'Static tiled-expanded rendering expects shape '
                    '(tile_rows, tile_cols, patch_values), '
                    f'got {array.shape}.'
                )
            return self.tile_patch_grid(array)
        if not self.static_input_is_tiled() and self.flat_input:
            if array.ndim != 1:
                raise ValueError(
                    'Static untiled-flat rendering expects shape '
                    '(input_values,), '
                    f'got {array.shape}.'
                )
            return self.reshape_flat_vector_for_display(array)
        if array.ndim == 2:
            return array
        raise ValueError(
            'Static untiled-expanded rendering expects shape (height, width). '
            f'Got {array.shape}. Supported display modes are tiled flat, tiled expanded, '
            'untiled flat, and untiled expanded.'
        )

    def render_static_array_to_overlapped_2d(self, array: np.ndarray, geometry: dict[str, int] | None = None) -> np.ndarray:
        """Sum tile patches into a single overlapped image using tile-offset geometry.

        Falls back to the plain mosaic rendering for untiled inputs or when no
        geometry is provided.
        """
        array = np.asarray(array)
        if not self.static_input_is_tiled():
            return self.render_static_array_to_2d(array)

        if geometry is None:
            return self.render_static_array_to_2d(array)

        rf1_x = geometry['rf1_x']
        rf1_y = geometry['rf1_y']
        rf1_offset_x = geometry['rf1_offset_x']
        rf1_offset_y = geometry['rf1_offset_y']
        rf1_layout_x = geometry['rf1_layout_x']
        rf1_layout_y = geometry['rf1_layout_y']
        patch_size = rf1_x * rf1_y

        if self.flat_input:
            if array.ndim != 2 or array.shape != (self.ntiles_per_input, patch_size):
                raise ValueError(
                    'Static tiled-flat overlap rendering expects shape '
                    f'({self.ntiles_per_input}, {patch_size}), got {array.shape}.'
                )
        else:
            if array.ndim != 3 or array.shape[:2] != (rf1_layout_y, rf1_layout_x) or array.shape[2] != patch_size:
                raise ValueError(
                    'Static tiled-expanded overlap rendering expects shape '
                    f'({rf1_layout_y}, {rf1_layout_x}, {patch_size}), got {array.shape}.'
                )

        overlapped = np.zeros(
            (
                rf1_y + (rf1_offset_y * (rf1_layout_y - 1)),
                rf1_x + (rf1_offset_x * (rf1_layout_x - 1)),
            ),
            dtype=np.float32,
        )

        for module_x in range(rf1_layout_x):
            for module_y in range(rf1_layout_y):
                if self.flat_input:
                    patch_index = module_x * rf1_layout_y + module_y
                    patch = array[patch_index]
                else:
                    patch = array[module_y, module_x]
                patch_image = patch.reshape(rf1_y, rf1_x)
                y_start = rf1_offset_y * module_y
                x_start = rf1_offset_x * module_x
                overlapped[y_start:y_start + rf1_y, x_start:x_start + rf1_x] += patch_image

        return overlapped

    def static_display_image(self, image: np.ndarray) -> np.ndarray:
        """Render an array to an upsampled uint8 display image."""
        display_matrix = self.render_static_array_to_2d(image)
        return self.normalize_to_uint8(self.nearest_upsample(display_matrix, scale=4))

    def static_overlapped_display_image(self, image: np.ndarray, geometry: dict[str, int] | None = None) -> np.ndarray:
        """Render an array to an upsampled uint8 overlapped display image."""
        display_matrix = self.render_static_array_to_overlapped_2d(image, geometry=geometry)
        return self.normalize_to_uint8(self.nearest_upsample(display_matrix, scale=4))

    def static_input_display_image(self, input_sample: np.ndarray) -> np.ndarray:
        """Render an input sample to a display image."""
        return self.static_display_image(input_sample)

    def static_cost_functions(self) -> StaticCostFunction:
        """Return the StaticCostFunction instance bound to this model's update methods."""
        return self.r_updates.__self__

    def project_static_layer_to_previous(self, layer_index: int, layer_state: np.ndarray) -> np.ndarray:
        """Project a layer's state down one layer via f(U_i . r_i)."""
        static_cost_func = self.static_cost_functions()
        if layer_index == 2:
            return self.f(static_cost_func.U2mat_mult_r2vec(self.U[2], layer_state))
        return self.f(static_cost_func.U_gteq3_mat_mult_r_gteq3_vec(self.U[layer_index], layer_state))

    def reconstruct_static_layer_image(self, layer_state: np.ndarray, level: int = 1) -> np.ndarray:
        """Reconstruct the input-space image generated by a layer's state.

        Projects the state down layer by layer to r1, then through U1.

        Raises:
            ValueError: For non-static models, out-of-range levels, or shape mismatches.
        """
        if not self.is_static_model():
            raise ValueError('Receptive-field reconstruction is only supported for static models.')

        if level < 1 or level > self.num_layers:
            raise ValueError(f'Layer level must be between 1 and {self.num_layers}, got {level}.')

        projected_state = np.array(layer_state, copy=True)
        for layer_index in range(level, 1, -1):
            projected_state = self.project_static_layer_to_previous(layer_index, projected_state)

        first_layer_state = np.asarray(projected_state)
        if first_layer_state.shape != self.r[1].shape:
            raise ValueError(
                f'Reconstructed first-layer state shape {first_layer_state.shape} does not match r[1] shape {self.r[1].shape}.'
            )

        rf2_patch = self.f(self.static_cost_functions().U1mat_mult_r1vecormat(self.U[1], first_layer_state))
        return np.asarray(rf2_patch)

    def static_receptive_field_state_by_layer(self, input_sample: np.ndarray, label: np.ndarray | None, update_method_name: str, update_method_number: int | float) -> dict[int, np.ndarray]:
        """Run an r-only settle on one input and return copies of each layer's final state."""
        prior_dist = self.static_reset_prior_sampler()
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)

        reset_rs_gteq1(prior_dist=prior_dist)
        self.r[0] = input_sample
        if label is None:
            eval_label = np.zeros(self.num_classes, dtype=np.float64)
        else:
            eval_label = np.zeros_like(label)
        update_non_weight_components(label=eval_label)
        return {layer_index: np.array(self.r[layer_index], copy=True) for layer_index in range(1, self.num_layers + 1)}

    def save_receptive_field_plots(self, input_name: str) -> None:
        """Render and save per-input receptive-field reconstruction figures.

        For each selected input, settles r, reconstructs each layer's generative
        image, and saves a panel figure (raw input, optional overlapped input,
        one panel per layer).

        Raises:
            ValueError: For recurrent models.
        """
        spec = self.receptive_fields_plot_spec()
        mode = spec['mode']
        if mode is None:
            return
        if self.model_type == 'recurrent':
            raise ValueError('plot_receptive_fields is only supported for static models.')

        X, Y = self.load_configured_dataset()
        X = np.asarray(X)
        if self.static_input_is_tiled() and self.flat_input and X.ndim == 2 and X.shape[0] == self.num_imgs * self.ntiles_per_input:
            X = X.reshape(self.num_imgs, self.ntiles_per_input, -1)

        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        receptive_fields_dir = self.receptive_fields_dir(input_name)
        tiled_geometry = None
        use_overlapped_tiled_panels = False
        if self.static_input_is_tiled():
            tiled_geometry = self.load_tiled_receptive_field_geometry(warn_if_missing=True)
            use_overlapped_tiled_panels = tiled_geometry is not None

        if mode == 'all':
            input_indices = list(range(self.num_imgs))
        else:
            requested_sample_count = spec['random_sample']
            available_input_count = int(self.num_imgs)
            if requested_sample_count > available_input_count:
                self.print_and_log(
                    f"Warning: plot_receptive_fields random_sample={requested_sample_count} exceeds available inputs ({available_input_count}); plotting only the {available_input_count} available inputs."
                )
                requested_sample_count = available_input_count
            input_indices = list(np.random.choice(available_input_count, size=requested_sample_count, replace=False))

        for input_index in input_indices:
            input_sample = X[input_index]
            label = None if Y is None else Y[input_index]
            layer_states = self.static_receptive_field_state_by_layer(
                input_sample,
                label,
                update_method_name,
                update_method_number,
            )

            panels = [(f'input raw: image_{input_index}', self.static_input_display_image(input_sample))]
            if use_overlapped_tiled_panels:
                panels.append(('input overlapped', self.static_overlapped_display_image(input_sample, geometry=tiled_geometry)))
            for layer_index in range(1, self.num_layers + 1):
                reconstructed = self.reconstruct_static_layer_image(layer_states[layer_index], level=layer_index)
                if use_overlapped_tiled_panels:
                    layer_image = self.static_overlapped_display_image(reconstructed, geometry=tiled_geometry)
                else:
                    layer_image = self.static_display_image(reconstructed)
                panels.append((f'layer {layer_index}', layer_image))

            fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3), dpi=80)
            if len(panels) == 1:
                axes = [axes]
            for ax, (title, image) in zip(axes, panels):
                ax.set_title(title)
                ax.imshow(image, cmap='gray')
                ax.axis('off')
            fig.tight_layout()

            output_path = join(
                receptive_fields_dir,
                self.receptive_field_file_name(input_name, input_index),
            )
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        self.print_and_log(f'Saved receptive-field plots:\n{receptive_fields_dir}\n')

    def summarize_softmaxed_activation_diagnostics(self, softmaxed_activations: np.ndarray | None) -> str:
        """Return a one-line summary of the saved softmaxed-activation diagnostics."""
        if softmaxed_activations is None:
            return 'softmaxed activations unavailable in diagnostics'

        softmaxed_activations = np.asarray(softmaxed_activations)
        if softmaxed_activations.shape != (4, self.num_ts):
            return f'softmaxed activations saved with unexpected shape: {softmaxed_activations.shape}'
        if np.isnan(softmaxed_activations).all():
            return 'softmaxed activations contain no correct-classification averages'

        return f'softmaxed activations shape: {softmaxed_activations.shape}'

    def summarize_prediction_diversity_diagnostics(self, prediction_diversity: dict[str, Any] | None) -> str:
        """Return a one-line summary of the saved prediction-diversity diagnostics."""
        if prediction_diversity is None:
            return 'prediction diversity unavailable in diagnostics'
        dominant_class = prediction_diversity['dominant_prediction_class']
        if dominant_class is None:
            return 'Prediction diversity: unavailable'
        return (
            f"Prediction diversity: {prediction_diversity['unique_prediction_count']}/{self.num_classes} classes; "
            f"dominant={dominant_class} ({prediction_diversity['dominant_prediction_count']}/{self.num_inps})"
        )

    def current_prediction_diversity_diagnostics(self, epoch: int) -> dict[str, Any] | None:
        """Return this epoch's prediction-diversity stats, or None when untracked."""
        if not hasattr(self, 'prediction_unique_count') or epoch >= len(self.prediction_unique_count):
            return None
        return {
            'epoch': epoch,
            'unique_prediction_count': self.prediction_unique_count[epoch],
            'dominant_prediction_class': self.prediction_dominant_class[epoch],
            'dominant_prediction_count': self.prediction_dominant_count[epoch],
        }

    def learning_rate_trace_specs(self) -> list[tuple[str, int | str]]:
        """Return (param_name, key) pairs for every learning rate worth tracing."""
        specs = []
        for param_name in ('kr', 'kU', 'kV'):
            rates = getattr(self, f'base_{param_name}', getattr(self, param_name, None))
            if rates is None:
                continue
            for param_key in sorted(rates, key=lambda key: (str(type(key)), str(key))):
                if param_key == 'o' and rates[param_key] == 0:
                    continue
                specs.append((param_name, param_key))
        return specs

    def learning_rate_value_at_epoch(self, param_name: str, param_key: int | str, epoch: int) -> float:
        """Return the (possibly scheduled) learning-rate value at an epoch."""
        base_rates = getattr(self, f'base_{param_name}', getattr(self, param_name))
        value = base_rates[param_key]
        for schedule in getattr(self, 'rate_schedules', []):
            if schedule.get('param') == param_name and schedule.get('key') == param_key:
                value = self.scheduled_rate_value(schedule, epoch)
        return value

    def learning_rate_label_epochs(self, param_name: str, param_key: int | str, max_epoch: int) -> list[int]:
        """Return the epochs at which a learning-rate trace should be labeled."""
        label_epochs = {0, max_epoch}
        for schedule in getattr(self, 'rate_schedules', []):
            if schedule.get('param') != param_name or schedule.get('key') != param_key:
                continue
            if schedule.get('type') == 'piecewise_linear':
                label_epochs.update(epoch for epoch, _ in schedule['points'] if 0 <= epoch <= max_epoch)
            elif schedule.get('type') == 'two_phase_linear_decay':
                label_epochs.update(
                    epoch for epoch in (schedule['start_epoch'], schedule['end_epoch'])
                    if 0 <= epoch <= max_epoch
                )
        return sorted(label_epochs)

    def add_learning_rate_traces_to_plot(self, ax: plt.Axes, epochs: list[int]) -> None:
        """Overlay normalized learning-rate traces (one lane per rate) on a metrics axis."""
        specs = self.learning_rate_trace_specs()
        if not specs or not epochs:
            return

        rate_ax = ax.twinx()
        rate_ax.set_axis_off()
        rate_ax.set_ylim(-0.05, len(specs) + 0.65)
        max_epoch = epochs[-1]
        x_values = list(epochs)

        for spec_index, (param_name, param_key) in enumerate(specs):
            baseline = spec_index + 0.15
            trace_values = [self.learning_rate_value_at_epoch(param_name, param_key, epoch) for epoch in x_values]
            min_value = min(trace_values)
            max_value = max(trace_values)
            value_span = max(max_value - min_value, 1e-12)
            y_values = [baseline + 0.5 * ((value - min_value) / value_span) for value in trace_values]
            rate_ax.plot(x_values, y_values, color='g', linewidth=0.5, alpha=0.75)

            for label_epoch in self.learning_rate_label_epochs(param_name, param_key, max_epoch):
                value = self.learning_rate_value_at_epoch(param_name, param_key, label_epoch)
                y_value = baseline + 0.5 * ((value - min_value) / value_span)
                component_label = f'{param_name}{param_key}'
                rate_ax.plot([label_epoch], [y_value], marker='o', markersize=2, color='g', alpha=0.85)
                rate_ax.annotate(
                    f'{component_label}, {value:.4g}',
                    xy=(label_epoch, y_value),
                    xytext=(3, 2),
                    textcoords='offset points',
                    color='g',
                    fontsize=6,
                )
            
    def calculate_average_softmaxed_activations_over_correct_inputs(self) -> np.ndarray:
        """Average correct inputs' softmaxed activations into (4, num_ts) TCRU rows.

        Groups each correct input's class-by-timestep activation matrix into
        Target/Cohort/Rhyme/Unrelated rows via dataset-specific index maps,
        then NaN-aware-averages across inputs.

        Returns:
            The (4, num_ts) TCRU average matrix (also stored on the model).

        Raises:
            ValueError: For static models, unrecognized datasets, class-count
                mismatches, or missing per-input activation data.
        """
        if self.model_type == 'static':
            raise ValueError(
                'Activation calculations are supported in plotting, but are skipped internally for static models because no SPCC word-processing model datasets currently provide the *_tcru_index_map lookup tables required by calculate_average_softmaxed_activations_over_correct_inputs(). '
                'Without those maps, the code cannot map inputs to cohort, rhyme, and unrelated identities.'
            )

        # Other lexica have different Target, Cohort, Rhyme characteristics.
        cvcv12_tcru_index_map = {
            0: (0, 1, 2),
            1: (1, 0, 3),
            2: (2, 3, 0),
            3: (3, 2, 1),
            4: (4, 5, 6),
            5: (5, 4, 7),
            6: (6, 7, 4),
            7: (7, 6, 5),
            8: (8, 9, 10),
            9: (9, 8, 11),
            10: (10, 11, 8),
            11: (11, 10, 9),
        }

        tom3phon3let_tcru_index_map = {
            0: (0, (1, 13), (7,)),
            1: (1, (0, 13), (2, 10, 11)),
            2: (2, (3, 6), (1, 10, 11)),
            3: (3, (2, 6), (4,)),
            4: (4, (5,), (3,)),
            5: (5, (4,), (6, 12, 15)),
            6: (6, (2, 3), (5, 12, 15)),
            7: (7, (8,), (0,)),
            8: (8, (7,), (9,)),
            9: (9, (10,), (8,)),
            10: (10, (9,), (1, 2, 11)),
            11: (11, (12,), (1, 2, 10)),
            12: (12, (11,), (5, 6, 15)),
            13: (13, (0, 1), (14,)),
            14: (14, (15,), (13,)),
            15: (15, (14,), (5, 6, 12)),
        }

        # 4-word subset order expected from data.py TOM3PHON3LET_4SET_WORDS:
        # ['LOT', 'LOP', 'TOP', 'TOT']. every word has 1 cohort, 1 rhyme, and 1 'unrelated' (though all share a center V in CVC).
        # Target/Cohort/Rhyme/Unrelated index tuples by class index.
        # LOT: 0, 1, 3
        # LOP: 1, 0, 2
        # TOP: 2, 3, 1
        # TOT: 3, 2, 0
        tom3phon3let_4set_tcru_index_map = {
            0: (0, (1,), (3,)),
            1: (1, (0,), (2,)),
            2: (2, (3,), (1,)),
            3: (3, (2,), (0,)),
        }

        dataset_train = getattr(self, 'dataset_train', '')
        dataset_key = re.sub(r'[^a-z0-9]', '', str(dataset_train).lower())
        if 'cvcv12' in dataset_key:
            tcru_index_map = cvcv12_tcru_index_map
            dataset_label = 'CVCV_12'
        elif 'tom3phon3let' in dataset_key:
            if self.num_classes == len(tom3phon3let_tcru_index_map):
                tcru_index_map = tom3phon3let_tcru_index_map
            elif self.num_classes == len(tom3phon3let_4set_tcru_index_map):
                tcru_index_map = tom3phon3let_4set_tcru_index_map
            else:
                raise ValueError(
                    'tom3phon3let TCRU activation averaging currently supports '
                    f'num_classes == {len(tom3phon3let_4set_tcru_index_map)} or '
                    f'num_classes == {len(tom3phon3let_tcru_index_map)}, '
                    f'got {self.num_classes}.'
                )
            dataset_label = 'tom3phon3let'
        else:
            raise ValueError(
                'TCRU activation averaging requires dataset_train to identify '
                f'cvcv12 or tom3phon3let, got {dataset_train}.'
            )

        if self.num_classes != len(tcru_index_map):
            raise ValueError(
                f'{dataset_label} TCRU activation averaging requires '
                f'num_classes == {len(tcru_index_map)}, '
                f'got {self.num_classes}.'
            )

        softmaxed_activations_over_all_correct_inputs = getattr(
            self,
            'softmaxed_activations_over_all_correct_inputs',
            None,
        )
        if softmaxed_activations_over_all_correct_inputs is None:
            raise ValueError(
                'Cannot calculate TCRU activation averages before '
                'self.softmaxed_activations_over_all_correct_inputs is set.'
            )

        softmaxed_activations_correct_inputs_TCRU = {}
        tcru_matrices = []
        for label_key, softmaxed_activation_matrix in softmaxed_activations_over_all_correct_inputs.items():
            if label_key not in tcru_index_map:
                raise ValueError(f'{dataset_label} TCRU mapping has no entry for label index {label_key}.')

            softmaxed_activation_matrix = np.asarray(softmaxed_activation_matrix)
            if softmaxed_activation_matrix.shape == (self.num_classes, self.num_ts):
                class_by_timestep_matrix = softmaxed_activation_matrix
            elif softmaxed_activation_matrix.shape == (self.num_ts, self.num_classes):
                class_by_timestep_matrix = softmaxed_activation_matrix.T
            else:
                raise ValueError(
                    'Expected softmaxed activation matrix shape '
                    f'({self.num_classes}, {self.num_ts}) or ({self.num_ts}, {self.num_classes}) '
                    f'for label {label_key}, got {softmaxed_activation_matrix.shape}.'
                )

            target_index, cohort_indices, rhyme_indices = tcru_index_map[label_key]
            if not isinstance(cohort_indices, (tuple, list)):
                cohort_indices = (cohort_indices,)
            if not isinstance(rhyme_indices, (tuple, list)):
                rhyme_indices = (rhyme_indices,)
            related_indices = {target_index, *cohort_indices, *rhyme_indices}
            unrelated_indices = [
                class_index
                for class_index in range(self.num_classes)
                if class_index not in related_indices
            ]
            tcru_matrix = np.vstack([
                class_by_timestep_matrix[target_index],
                class_by_timestep_matrix[list(cohort_indices)].mean(axis=0),
                class_by_timestep_matrix[list(rhyme_indices)].mean(axis=0),
                class_by_timestep_matrix[unrelated_indices].mean(axis=0),
            ])
            softmaxed_activations_correct_inputs_TCRU[label_key] = tcru_matrix
            tcru_matrices.append(tcru_matrix)

        self.softmaxed_activations_correct_inputs_TCRU = softmaxed_activations_correct_inputs_TCRU
        if tcru_matrices:
            stacked_tcru_matrices = np.stack(tcru_matrices, axis=0)
            valid_counts = np.sum(~np.isnan(stacked_tcru_matrices), axis=0)
            summed_tcru_matrices = np.nansum(stacked_tcru_matrices, axis=0)
            self.average_softmaxed_activations_over_correct_inputs = np.divide(
                summed_tcru_matrices,
                valid_counts,
                out=np.full_like(summed_tcru_matrices, np.nan),
                where=valid_counts > 0,
            )
        else:
            self.average_softmaxed_activations_over_correct_inputs = np.full((4, self.num_ts), np.nan)

        return self.average_softmaxed_activations_over_correct_inputs
        
            
    def save_diagnostics(self, output_dir: str, output_name: str) -> None:
        """Pickle the Jr/Jc/accuracy histories and activation/diversity diagnostics for one epoch."""
        epoch_dir = self.epoch_artifact_dir(output_dir, output_name)
        epoch = self.epoch_from_output_name(output_name)
        output_path = join(epoch_dir, self.diagnostics_file_name(output_name))
        softmaxed_activations = getattr(self, 'average_softmaxed_activations_over_correct_inputs', None)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'Jr': self.Jr,
                'Jc': self.Jc,
                'accuracy': self.accuracy,
                'softmaxed_activations': softmaxed_activations,
                'prediction_diversity': self.current_prediction_diversity_diagnostics(epoch),
            }, f)

    def finite_plot_values(self, values: Iterable[float]) -> np.ndarray:
        """Replace non-finite values with NaN so matplotlib skips them."""
        values = np.asarray(values, dtype=float)
        return np.where(np.isfinite(values), values, np.nan)

    def finite_values(self, values: Iterable[float]) -> np.ndarray:
        """Return only the finite values."""
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    def set_finite_ylim(self, ax: plt.Axes, values: Iterable[float], default: tuple[float, float] = (0, 1)) -> bool:
        """Set y-limits from finite values; fall back to default and return False when none exist."""
        finite_values = self.finite_values(values)
        if finite_values.size == 0:
            ax.set_ylim(*default)
            return False

        y_min = min(0, float(np.min(finite_values)))
        y_max = float(np.max(finite_values))
        if y_min == y_max:
            margin = abs(y_max) * 0.05 or 0.05
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(y_min, y_max * 1.05)
        return True

    def finite_percent_change(self, start_value: float, end_value: float, epsilon: float = 1e-6) -> float:
        """Return percent change from start to end, or NaN when either is non-finite."""
        if not np.isfinite(start_value) or not np.isfinite(end_value):
            return np.nan
        return ((end_value - start_value) / (start_value + epsilon)) * 100

    def plot_safely(self, input_dir: str, input_name: str) -> None:
        """Run plot() (and receptive-field plots when enabled), logging instead of raising on failure."""
        try:
            self.plot(input_dir=input_dir, input_name=input_name)
            if self.receptive_fields_enabled():
                self.save_receptive_field_plots(input_name=input_name)
        except Exception as error:
            message = f'Plot skipped for {input_name}: {type(error).__name__}: {error}'
            try:
                self.print_and_log(message)
            except Exception:
                print(message)

    
    def plot(self, input_dir: str, input_name: str) -> None:
        """Plot training metrics (and TCRU activation curves, when saved) from a diagnostics pickle."""
        printlog = self.print_and_log
        
        diags_name = self.diagnostics_file_name(input_name)
        epoch_n = self.epoch_from_output_name(input_name)
        diags_path = join(input_dir, str(epoch_n), diags_name)
        
        # Load the model
        with open(diags_path, 'rb') as file:
            diags = pickle.load(file)
        
        printlog(f'\nplot:\nloaded diag file:\n{diags_path}')
        printlog(self.summarize_softmaxed_activation_diagnostics(diags.get('softmaxed_activations')))
        prediction_diversity_summary = self.summarize_prediction_diversity_diagnostics(
            diags.get('prediction_diversity')
        )
        printlog(prediction_diversity_summary)
        
        # clip diags at epoch_n
        diags['Jr'] = diags['Jr'][:epoch_n + 1]
        classification_metrics_available = diags.get('Jc') is not None and diags.get('accuracy') is not None
        if classification_metrics_available:
            diags['Jc'] = diags['Jc'][:epoch_n + 1]
            diags['accuracy'] = diags['accuracy'][:epoch_n + 1]
            
        epochs = range(len(diags['Jr']))
            
        if classification_metrics_available:
            # Turn each accuracy into an actual percent
            diags['accuracy'] = [acc * 100 for acc in diags['accuracy']]
        Jr_plot = self.finite_plot_values(diags['Jr'])
        if classification_metrics_available:
            Jc_plot = self.finite_plot_values(diags['Jc'])
            accuracy_plot = self.finite_plot_values(diags['accuracy'])

        # Calculate percent changes
        percent_change_Jr = self.finite_percent_change(diags['Jr'][0], diags['Jr'][-1])
        if classification_metrics_available:
            percent_change_Jc = self.finite_percent_change(diags['Jc'][0], diags['Jc'][-1])
            percent_change_accuracy = self.finite_percent_change(diags['accuracy'][0], diags['accuracy'][-1])

        fig, ax1 = plt.subplots()

        # Plot Jr
        ax1.plot(epochs, Jr_plot, 'y-', label='Jr')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Jr', color='y')
        ax1.tick_params(axis='y', labelcolor='y')
        self.set_finite_ylim(ax1, Jr_plot)
        self.add_learning_rate_traces_to_plot(ax1, list(epochs))

        if classification_metrics_available:
            # Create a second y-axis for Jc
            ax2 = ax1.twinx()
            ax2.plot(epochs, Jc_plot, 'b-', label='Jc')
            ax2.set_ylabel('Jc', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            self.set_finite_ylim(ax2, Jc_plot)

            # Create a third y-axis for accuracy
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(epochs, accuracy_plot, 'r-', label='Accuracy', linewidth=0.5)
            ax3.set_ylabel('Accuracy (%)', color='r')
            ax3.tick_params(axis='y', labelcolor='r')
            finite_accuracy = self.finite_values(accuracy_plot)
            if finite_accuracy.size == 0:
                min_accuracy = 0
                max_accuracy = 1
            else:
                min_accuracy = float(np.min(finite_accuracy))
                max_accuracy = float(np.max(finite_accuracy))
            if min_accuracy == max_accuracy:
                accuracy_margin = abs(max_accuracy) * 0.05
                if accuracy_margin == 0:
                    accuracy_margin = 0.05
                ax3.set_ylim(min_accuracy - accuracy_margin, max_accuracy + accuracy_margin)
            else:
                ax3.set_ylim(min_accuracy * 1.05, max_accuracy * 1.05)

        # Add legends
        lines, labels = ax1.get_legend_handles_labels()
        if classification_metrics_available:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines += lines2 + lines3
            labels += labels2 + labels3
        ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(-.3, .9))

        # Add percent change text
        plt.text(0.1, 0.95, f'Jr % Change: {percent_change_Jr:.2f}%', transform=ax1.transAxes, color='y')
        if classification_metrics_available:
            plt.text(0.1, 0.90, f'Jc % Change: {percent_change_Jc:.2f}%', transform=ax1.transAxes, color='b')
            plt.text(0.1, 0.85, f'Accuracy % Change: {percent_change_accuracy:.2f}%', transform=ax1.transAxes, color='r')
            plt.text(0.1, 0.80, prediction_diversity_summary, transform=ax1.transAxes, color='k', fontsize=8)
        else:
            plt.text(0.1, 0.90, 'Classification disabled: Jc and accuracy not calculated', transform=ax1.transAxes, color='k', fontsize=8)
        
        plt.title('Training Diagnostics\n' + input_name + '\n')
        
        results_folder = self.epoch_artifact_dir(getattr(self, 'plots_dir', 'results/plots'), input_name)
        results_path = join(results_folder, 'eval-accuracy-during-training.' + diags_name)
        fig.savefig(results_path + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        if self.softmaxed_activation_plots_enabled():
            activation_fig, activation_ax = plt.subplots()
            softmaxed_activations = diags.get('softmaxed_activations')
            activation_summary = self.summarize_softmaxed_activation_diagnostics(softmaxed_activations)
            no_correct_eval_message = 'no correct guesses during eval'
            no_correct_target_classifications = False

            if softmaxed_activations is None:
                activation_ax.text(0.5, 0.5, activation_summary, ha='center', va='center', transform=activation_ax.transAxes)
                activation_ax.set_axis_off()
            else:
                softmaxed_activations = np.asarray(softmaxed_activations)
                no_correct_target_classifications = softmaxed_activations.shape == (4, self.num_ts) and np.isnan(softmaxed_activations).all()
                if no_correct_target_classifications:
                    activation_ax.text(0.5, 0.5, no_correct_eval_message, ha='center', va='center', transform=activation_ax.transAxes)
                    activation_ax.set_axis_off()
                elif softmaxed_activations.shape == (4, self.num_ts) and not np.isnan(softmaxed_activations).all():
                    timesteps = range(self.num_ts)
                    activation_ax.plot(timesteps, softmaxed_activations[0], color='b', label='Target')
                    activation_ax.plot(timesteps, softmaxed_activations[1], color='r', label='Cohort')
                    activation_ax.plot(timesteps, softmaxed_activations[2], color='orange', label='Rhyme')
                    activation_ax.plot(timesteps, softmaxed_activations[3], color='g', label='Unrelated')
                    activation_ax.set_xlabel('Timestep')
                    activation_ax.set_ylabel('Softmax Activation')
                    activation_ax.legend(loc='best')
                    finite_activations = self.finite_values(softmaxed_activations)
                    ymax = float(np.max(finite_activations)) if finite_activations.size else 1.0
                    activation_ax.set_ylim(0, max(1.0, ymax * 1.05))
                else:
                    activation_ax.text(0.5, 0.5, activation_summary, ha='center', va='center', transform=activation_ax.transAxes)
                    activation_ax.set_axis_off()

            activation_title = 'Softmaxed Activation Diagnostics\n' + input_name
            if not self.classification_enabled():
                activation_title += '\nClassification disabled'
            elif no_correct_target_classifications:
                activation_title += '\n(No correct target classifications)'
            activation_ax.set_title(activation_title + '\n')
            activation_results_path = join(results_folder, 'avg-activations.' + diags_name)
            activation_fig.savefig(activation_results_path + '.png', dpi=300, bbox_inches='tight')
            plt.close(activation_fig)
        else:
            printlog('Softmaxed activation plotting disabled; skipping activation diagnostics panel.')
        
        printlog(f'Saved plot:\n',results_path,'\n')
        if self.softmaxed_activation_plots_enabled():
            printlog(f'Saved activation plot:\n',activation_results_path,'\n')
            
        
class StaticPCC(PredictiveCodingClassifier):
    """Static predictive-coding classifier (sPCC).

    Bind layer-count-specific r/U update and cost methods from
    StaticCostFunction plus the configured classification method.
    """

    REQUIRED_CONFIG_KEYS = PredictiveCodingClassifier.REQUIRED_CONFIG_KEYS_SPCC

    def __init__(self, params: dict[str, Any]) -> None:
        """Configure the static model and bind its update/cost/classification methods."""
        super().__init__(params)
        self._configure_model()
        
        # Cost functions
        static_cost_func_class = StaticCostFunction(self)
        self.Uo_update = static_cost_func_class.Uo_update
        
        # Component cost funcs: r, U, Uo updates, and cost calculators
        n = self.num_layers
        if n == 1:
            self.r_updates = static_cost_func_class.r_updates_n_1
            self.U_updates = static_cost_func_class.U_updates_n_1
            self.rep_cost = static_cost_func_class.rep_cost_n_1
        elif n == 2:
            self.r_updates = static_cost_func_class.r_updates_n_2
            self.U_updates = static_cost_func_class.U_updates_n_gt_eq_2
            self.rep_cost = static_cost_func_class.rep_cost_n_2
        elif n >= 3:
            self.r_updates = static_cost_func_class.r_updates_n_gt_eq_3
            self.U_updates = static_cost_func_class.U_updates_n_gt_eq_2
            self.rep_cost = static_cost_func_class.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
        # Components together
        self.component_updates = [self.r_updates, self.U_updates]
        if self.classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
        self.rW_instdeltas_updates = partial(static_cost_func_class.rW_instdeltas_updates,
                            weight_updates=self.component_updates[1:])
        self.rW_instdeltas_component_updates = [self.rW_instdeltas_updates]
        self.r_instdeltas_component_updates = [static_cost_func_class.r_instdeltas_update]

        # Dictionaries for methods from base class and static cost function class
        self.update_method_dict = {'rW_instdeltas_niters': partial(self.update_method_rW_instdeltas_niters, component_updates=self.rW_instdeltas_component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters_W, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq_W, component_updates=self.component_updates)}
        
        self.update_method_no_weight_dict = {'rW_instdeltas_niters': partial(self.update_method_r_instdeltas_niters, component_updates=self.r_instdeltas_component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq, component_updates=self.component_updates)}
        
        self.classif_cost_dict = {'c1': static_cost_func_class.classif_cost_c1,
                                'c2': static_cost_func_class.classif_cost_c2,
                                None: static_cost_func_class.classif_cost_None}
        
        self.classif_guess_dict = {'c1': static_cost_func_class.classif_guess_c1,
                                'c2': static_cost_func_class.classif_guess_c2,
                                None: static_cost_func_class.classif_guess_None}
        
        # Classification now
        self.classif_cost = self.classif_cost_dict[self.classif_method]
        self.classify = self.classif_guess_dict[self.classif_method]

    def _configure_model(self) -> None:
        """Validate parameters and initialize static model state."""
        self.configure_static_model()

    def validate_model_layer_config(self) -> None:
        """Require per-layer keys in kr/kU/alph/lam and in ssq (which includes layer 0)."""
        layer_keys = range(1, self.num_layers + 1)
        self.require_dict_keys('kr', layer_keys)
        self.require_dict_keys('kU', layer_keys)
        self.require_dict_keys('alph', layer_keys)
        self.require_dict_keys('lam', layer_keys)
        self.require_dict_keys('ssq', range(0, self.num_layers + 1))

class RecurrentPCC(PredictiveCodingClassifier):
    """Recurrent predictive-coding classifier (rPCC).

    Add recurrent weights V[i], per-timestep bar/hat state trajectories
    (bar = predicted pre-correction state, hat = corrected state), and the
    context state r['c'] driven by classification at the top layer.
    """

    REQUIRED_CONFIG_KEYS = PredictiveCodingClassifier.REQUIRED_CONFIG_KEYS_RPCC

    def __init__(self, params: dict[str, Any]) -> None:
        """Configure the recurrent model and bind its update/cost/classification methods."""
        super().__init__(params)
        self._configure_model()
        
        # Cost functions
        recurrent_cost_func_class = RecurrentCostFunction(self)
        
        # Component cost funcs: r, U, Uo updates, and cost calculators
        n = self.num_layers
        if n == 1:
            self.r_updates = recurrent_cost_func_class.r_updates_n_gteq_1
            self.U_updates = recurrent_cost_func_class.U_updates_n_gteq_1
            self.V_updates = recurrent_cost_func_class.V_updates_n_gteq_1
            self.rep_cost = recurrent_cost_func_class.rep_cost_n_gteq_1
            self.Uo_update = recurrent_cost_func_class.Uo_update
        elif n == 2:
            self.r_updates = recurrent_cost_func_class.r_updates_n_2
            self.U_updates = recurrent_cost_func_class.U_updates_n_2
            self.V_updates = recurrent_cost_func_class.V_updates_n_2
            self.rep_cost = recurrent_cost_func_class.rep_cost_n_2
            self.Uo_update = recurrent_cost_func_class.Uo_update
            
        elif n >= 3:
            self.r_updates = recurrent_cost_func_class.r_updates_n_gteq_1
            self.U_updates = recurrent_cost_func_class.U_updates_n_gteq_1
            self.V_updates = recurrent_cost_func_class.V_updates_n_gteq_1
            self.rep_cost = recurrent_cost_func_class.rep_cost_n_gteq_1
            self.Uo_update = recurrent_cost_func_class.Uo_update
        else:
            raise ValueError("Number of layers must be at least 1.")
        # Components together
        self.component_updates = [self.r_updates, self.U_updates]
        if self.classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
        self.component_updates.append(self.V_updates)
        
        
        # Dictionaries for methods from base class and static cost function class
        self.update_method_dict = {'rW_seq_niters': partial(self.update_method_rW_seq_niters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters_W, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq_W, component_updates=self.component_updates)}
        
        self.update_method_no_weight_dict = {'rW_seq_niters': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq, component_updates=self.component_updates)}
        
        self.classif_cost_dict = {'c1': partial(recurrent_cost_func_class.classif_cost_c1, num_ts=self.num_ts),
                                'c2': getattr(recurrent_cost_func_class, 'classif_cost_c2', None),
                                None: getattr(recurrent_cost_func_class, 'classif_cost_None', None)}
        
        self.classif_guess_dict = {'c1': partial(recurrent_cost_func_class.classif_guess_c1, num_ts=self.num_ts),
                                'c2': getattr(recurrent_cost_func_class, 'classif_guess_c2', None),
                                None: getattr(recurrent_cost_func_class, 'classif_guess_None', None)}
        # Classification now
        self.classif_cost = self.classif_cost_dict[self.classif_method]
        self.classify = self.classif_guess_dict[self.classif_method]

        
        
    def _configure_model(self) -> None:
        """Validate parameters and initialize recurrent r/U/V state, trajectories, and diagnostics."""
        self.configure_shared_model()
        self.validate_recurrent_stub_controls()

        n = self.num_layers
        num_ts = self.num_ts

        # Inits on representations and weights
        self.r = {}
        self.r_expansion = {}
        self.U = {}
        self.U_expansion = {}
        self.V = {}
        self.V_expansion = {}
        
        # r0
        self.r[0] = np.zeros(self.input_shape)
        print(f'total input shape (1 input. if speech, 1 word): {(self.r[0].shape, self.num_ts)}')
        print(f'r0 shape (slice at t): {self.r[0].shape}')
        print('Inits:')
        lyr_sizes = self.hidden_lyr_sizes + [self.top_lyr_size]

        # Draw all U first, then all V with shape-only metadata so weight RNG
        # alignment matches inext (which initializes weights before recurrent states).
        prev_size = self.input_shape[0]
        for i in range(1, n + 1):
            curr_size = lyr_sizes[i - 1]
            self.U[i] = self.sample_U_prior(size=(prev_size, curr_size))
            self.U_expansion[i] = np.repeat(self.U[i][:, :, None], num_ts, axis=2)
            print(f'U{i} shape: {self.U[i].shape}', f'U{i} expansion shape: {self.U_expansion[i].shape}')
            prev_size = curr_size
        for i in range(1, n + 1):
            curr_size = lyr_sizes[i - 1]
            self.V[i] = self.V_prior_dist(size=(curr_size, curr_size))
            self.V_expansion[i] = np.repeat(self.V[i][:, :, None], num_ts, axis=2)
            print(f'V{i} shape: {self.V[i].shape}', f'V{i} expansion shape: {self.V_expansion[i].shape}')

        # r1 - rn initialize from the configured r prior after U/V draws.
        for i in range(1, n + 1):
            initial_state = self.sample_r_prior(size=(lyr_sizes[i - 1],))
            self.r[i] = np.array(initial_state, copy=True)
            self.r_expansion[i] = np.repeat(initial_state[:, None], num_ts, axis=1)
            print(f'r{i} shape: {self.r[i].shape}', f'r{i} expansion shape: {self.r_expansion[i].shape}')

        # rc (r_context / r2_hat_x in inext) uses the same r prior policy.
        rc_initial_state = self.sample_r_prior(size=(self.top_lyr_size,))
        self.r['c'] = np.array(rc_initial_state, copy=True)
        self.r_expansion['c'] = np.repeat(rc_initial_state[:, None], num_ts, axis=1)
            
        # Classification now
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.top_lyr_size)
            self.U['o'] = self.sample_U_prior(size=Uo_size)
            self.U_expansion['o'] = np.repeat(self.U['o'][:, :, None], num_ts, axis=2)
            print(f'Uo shape: {self.U["o"].shape}', f'Uo expansion shape: {self.U_expansion["o"].shape}')
        
        # Create independent (deep) copies for rhat, rbar, Uhat, Ubar, Vhat, Vbar.
        # Each hat/bar must own its array; sharing would alias the buffers and
        # zero out difference-based weight updates (e.g. rhat - rbar).
        print('self.r,U,V are kept for final state save.')
        print('self.r_expansion,U_expansion,V_expansion have had copies made and will be used for rbars, hats, Ubars, hats, Vbars, hats.')
        self.rhat = {k: np.array(v, copy=True) for k, v in self.r_expansion.items()}
        self.rbar = {k: np.array(v, copy=True) for k, v in self.r_expansion.items()}
        self.Uhat = {k: np.array(v, copy=True) for k, v in self.U_expansion.items()}
        self.Ubar = {k: np.array(v, copy=True) for k, v in self.U_expansion.items()}
        self.Vhat = {k: np.array(v, copy=True) for k, v in self.V_expansion.items()}
        self.Vbar = {k: np.array(v, copy=True) for k, v in self.V_expansion.items()}
        
        for i in range(1, n + 1):
            print(f'rhat{i} shape: {self.rhat[i].shape}', f'rbar{i} shape: {self.rbar[i].shape}')
            print(f'Uhat{i} shape: {self.Uhat[i].shape}', f'Ubar{i} shape: {self.Ubar[i].shape}')
            print(f'Vhat{i} shape: {self.Vhat[i].shape}', f'Vbar{i} shape: {self.Vbar[i].shape}')
        
        # All layer sizes for priors
        self.all_lyr_sizes = self.hidden_lyr_sizes.copy()
        self.all_lyr_sizes.append(self.top_lyr_size)
        
        # Set softmax: softmax_k drives update equations; softmax_k_eval drives evaluation/guessing/diagnostics.
        self.softmax_func = partial(self.set_softmax_func, softmax_type=self.softmax_type, k=self.softmax_k)
        self.softmax_func_eval = partial(self.set_softmax_func, softmax_type=self.softmax_type, k=self.softmax_k_eval)
        self.rate_schedule = getattr(self, 'rate_schedule', None)
        self.rate_schedules = self.normalize_rate_schedules(self.rate_schedule)
        self.base_kr = copy(self.kr)
        self.base_kU = copy(self.kU)
        self.base_kV = copy(self.kV)
        
        # Initiate Jr, Jc, and accuracy (diagnostics) for storage, print, plot
        epoch_n = self.epoch_n
        self.Jr = [0] * (epoch_n + 1)
        self.Jc = [0] * (epoch_n + 1) if self.classification_enabled() else None
        self.accuracy = [0] * (epoch_n + 1) if self.classification_enabled() else None
        self.prediction_unique_count = [0] * (epoch_n + 1)
        self.prediction_dominant_class = [None] * (epoch_n + 1)
        self.prediction_dominant_count = [0] * (epoch_n + 1)

    def validate_model_layer_config(self) -> None:
        """Require per-layer keys in kr/kU/kV/ssqV and in ssqr (which includes layer 0)."""
        layer_keys = range(1, self.num_layers + 1)
        self.require_dict_keys('kr', layer_keys)
        self.require_dict_keys('kU', layer_keys)
        self.require_dict_keys('kV', layer_keys)
        self.require_dict_keys('ssqr', range(0, self.num_layers + 1))
        self.require_dict_keys('ssqV', layer_keys)
        
    def V_prior_dist(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Draw a V weight sample (same prior as U)."""
        return self.sample_V_prior(size=size)

    def hard_set_recurrent_r_prior_dist(self) -> None:
        """Cache one recurrent prior sample per layer size for per-input resets."""
        self.r_dists_hard_recurrent = {}
        n = self.num_layers
        for i in range(1, n + 1):
            lyr_size = self.all_lyr_sizes[i - 1]
            self.r_dists_hard_recurrent[lyr_size] = self.sample_r_prior(size=lyr_size)

    def load_hard_recurrent_r_prior_dist(self, size: int | tuple[int, ...]) -> np.ndarray:
        """Return a copy of the cached recurrent prior sample to avoid aliasing."""
        return np.array(self.r_dists_hard_recurrent[size], copy=True)

    def recurrent_reset_prior_sampler(self) -> Callable[..., np.ndarray]:
        """Return the recurrent r-reset sampler for the configured r_reset_mode."""
        mode = self.configured_r_reset_mode()
        if mode == 'reset_from_single_sample':
            self.hard_set_recurrent_r_prior_dist()
            return self.load_hard_recurrent_r_prior_dist
        return self.sample_r_prior
    
    def fill_UsVs_with_last_timestep(self) -> None:
        """Broadcast the last-timestep U/V slices across all timesteps.

        Effectively 'freezes' U and V so all r updates use the most recent U, V.
        """
        for i in range(1, self.num_layers + 1):
            self.Ubar[i] = np.repeat(self.Ubar[i][:, :, -1][:, :, None], self.num_ts, axis=2)
            self.Uhat[i] = np.repeat(self.Uhat[i][:, :, -1][:, :, None], self.num_ts, axis=2)
            self.Vbar[i] = np.repeat(self.Vbar[i][:, :, -1][:, :, None], self.num_ts, axis=2)
            self.Vhat[i] = np.repeat(self.Vhat[i][:, :, -1][:, :, None], self.num_ts, axis=2)
        if self.classif_method == 'c2':
            self.Ubar['o'] = np.repeat(self.Ubar['o'][:, :, -1][:, :, None], self.num_ts, axis=2)
            self.Uhat['o'] = np.repeat(self.Uhat['o'][:, :, -1][:, :, None], self.num_ts, axis=2)

    def valid_timesteps_until_nan(self, input: np.ndarray) -> int:
        """Return the number of leading timesteps before the first NaN column (NaN-padded inputs).

        Raises:
            ValueError: When timestep 0 already contains NaNs.
        """
        nan_timesteps = np.where(np.any(np.isnan(input), axis=0))[0]
        if len(nan_timesteps) == 0:
            return self.num_ts
        valid_num_ts = int(nan_timesteps[0])
        if valid_num_ts == 0:
            raise ValueError('Input contains NaNs at timestep 0; no valid recurrent timesteps to process.')
        return valid_num_ts

    def normalize_rate_schedules(self, rate_schedule: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> list[dict[str, Any]]:
        """Normalize the rate_schedule config value to a list of schedule dicts.

        Raises:
            ValueError: For non-dict entries or unsupported container types.
        """
        if rate_schedule is None:
            return []
        if isinstance(rate_schedule, dict):
            return [rate_schedule]
        if isinstance(rate_schedule, (list, tuple)):
            normalized_schedules = list(rate_schedule)
            if not all(isinstance(schedule, dict) for schedule in normalized_schedules):
                raise ValueError('rate_schedule iterable must contain only schedule dictionaries.')
            return normalized_schedules
        raise ValueError('rate_schedule must be None, a schedule dictionary, or a list/tuple of schedule dictionaries.')

    def scheduled_rate_value(self, schedule: dict[str, Any], epoch: int) -> float:
        """Return a schedule's learning-rate value at an epoch.

        Supports 'piecewise_linear' (interpolated (epoch, value) points) and
        'two_phase_linear_decay' (base value, then linear decay to final_value).

        Raises:
            ValueError: For unsupported types or malformed schedule entries.
        """
        schedule_type = schedule.get('type')

        if schedule_type == 'piecewise_linear':
            points = schedule['points']
            if not points:
                raise ValueError('piecewise_linear rate_schedule requires at least one point.')
            for point_index in range(len(points) - 1):
                if points[point_index + 1][0] <= points[point_index][0]:
                    raise ValueError('piecewise_linear rate_schedule points must have strictly increasing epochs.')

            first_epoch, first_value = points[0]
            if epoch <= first_epoch:
                return first_value

            for point_index in range(len(points) - 1):
                start_epoch, start_value = points[point_index]
                end_epoch, end_value = points[point_index + 1]
                if start_epoch <= epoch < end_epoch:
                    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                    return start_value + progress * (end_value - start_value)

            return points[-1][1]

        if schedule_type != 'two_phase_linear_decay':
            raise ValueError(f'Unsupported rate_schedule type: {schedule_type}')

        param_name = schedule['param']
        param_key = schedule['key']
        start_epoch = schedule['start_epoch']
        end_epoch = schedule['end_epoch']
        final_value = schedule['final_value']

        if end_epoch < start_epoch:
            raise ValueError('rate_schedule end_epoch must be greater than or equal to start_epoch.')

        base_rates = getattr(self, f'base_{param_name}')
        base_value = base_rates[param_key]
        if epoch < start_epoch:
            return base_value
        if epoch >= end_epoch:
            return final_value

        if end_epoch == start_epoch:
            return final_value
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return base_value + progress * (final_value - base_value)

    def apply_rate_schedule(self, epoch: int) -> list[tuple[dict[str, Any], str, int | str, float]] | None:
        """Apply every configured schedule's value for this epoch to the live rate dicts.

        Returns:
            List of (schedule, param_name, param_key, value) applied, or None
            when no schedules are configured.
        """
        if not self.rate_schedules:
            return None

        scheduled_rates = []
        for schedule in self.rate_schedules:
            param_name = schedule['param']
            param_key = schedule['key']
            if param_name not in ('kr', 'kU', 'kV'):
                raise ValueError(f'Unsupported rate_schedule param: {param_name}')

            rates = getattr(self, param_name)
            scheduled_value = self.scheduled_rate_value(schedule, epoch)
            rates[param_key] = scheduled_value
            scheduled_rates.append((schedule, param_name, param_key, scheduled_value))
        return scheduled_rates

    def rate_schedule_log_epochs(self, schedule: dict[str, Any]) -> set[int]:
        """Return the epochs at which a schedule's value changes should be logged."""
        schedule_type = schedule.get('type')
        if schedule_type == 'piecewise_linear':
            return {point[0] for point in schedule['points']}
        if schedule_type == 'two_phase_linear_decay':
            return {schedule['start_epoch'], schedule['end_epoch']}
        return set()

    def ensure_training_tracking_state(self) -> None:
        """Backfill schedule and prediction-tracking attributes on models loaded from older checkpoints."""
        self.rate_schedule = getattr(self, 'rate_schedule', None)
        self.rate_schedules = self.normalize_rate_schedules(self.rate_schedule)
        epoch_n = self.epoch_n
        if not hasattr(self, 'prediction_unique_count'):
            self.prediction_unique_count = [0] * (epoch_n + 1)
        if not hasattr(self, 'prediction_dominant_class'):
            self.prediction_dominant_class = [None] * (epoch_n + 1)
        if not hasattr(self, 'prediction_dominant_count'):
            self.prediction_dominant_count = [0] * (epoch_n + 1)
    
    def train(self, X: np.ndarray, Y: np.ndarray, save_checkpoint: dict[str, Any] | None = None, load_checkpoint: dict[str, Any] | None = None, plot: bool = False) -> None:
        """Train the recurrent model: per input, unfold timesteps and update r (and weights) at each ts.

        Args:
            X: Training inputs, one (features, timesteps) matrix per input; may be NaN-padded.
            Y: One-hot labels.
            save_checkpoint: {'save_every': N} or {'fraction': frac or (num, den)} checkpoint policy.
            load_checkpoint: Resume marker; when set, training resumes from self.load_epoch.
            plot: Whether to render diagnostics plots at each checkpoint save.
        """
        printlog = self.print_and_log
        self.ensure_training_tracking_state()
        if not self.softmaxed_activation_plots_enabled():
            self.softmaxed_activations_over_all_correct_inputs = None
            self.average_softmaxed_activations_over_correct_inputs = None

        # Log initial shapes and prior previews.
        printlog('\n\n')
        printlog('shapes')
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} shape: {self.r[i].shape}')
            if i > 0:
                # Check if the array is 3D or 5D
                printlog(f'U{i} shape: {self.U[i].shape}')
        if self.classif_method == 'c2':
            printlog(f'Uo shape: {self.U["o"].shape}')
        printlog('\n')
            
        printlog(
            f'r_prior_dist: {self.r_prior_dist_label}, r_prior_cost: {self.r_prior_cost_name}, '
            f'U_prior_dist: {self.U_prior_dist_label}, U_prior_cost: {self.U_prior_cost_name} init preview:'
        )
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} first 3: {self.r[i][:3]}')
            if i > 0:
                # Check if the array is 2D,3 or 5D
                if self.U[i].ndim == 3:
                    printlog(f'U{i} first 3x3x3: {self.U[i][:3, :3, :3]}')
                elif self.U[i].ndim == 5:
                    printlog(f'U{i} first 3x3x3x3x3: {self.U[i][:3, :3, :3, :3, :3]}')
                else:
                    printlog(f'U{i} first 3x3: {self.U[i][:3, :3]}')
                printlog(f'V{i} first 3x3: {self.V[i][:3, :3]}')
        if self.classif_method == 'c2':
            printlog(f'Uo first 3x3: {self.U["o"][:3, :3]}')
        printlog('\n')

        epoch_n = self.epoch_n
        num_inps = self.num_inps

        printlog('Train init:')
        printlog('X shape:', X.shape)
        printlog('Y shape:', Y.shape)
        
        
        prior_dist = self.recurrent_reset_prior_sampler()
        printlog(f'r_reset_mode: {self.configured_r_reset_mode()}')

        # Methods
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        
        fill_UsVs_with_last_timestep = self.fill_UsVs_with_last_timestep
        
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_all_components = partial(self.update_method_dict[update_method_name], update_method_number)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        if self.rate_schedule is not None:
            printlog(f'Rate schedule: {self.rate_schedule}')
        
        rep_cost = partial(self.rep_cost, num_ts=self.num_ts)
        
        classification_enabled = self.classification_enabled()
        classif_cost = partial(self.classif_cost, num_ts=self.num_ts) if classification_enabled else None
        
        evaluate = partial(self.evaluate, update_method_name=update_method_name, update_method_number=update_method_number, plot=None) if classification_enabled else None
        
        
        
        # Checkpointing
        if 'save_every' in save_checkpoint:
            # If N
            checkpoint = save_checkpoint['save_every'] 
        elif 'fraction' in save_checkpoint:
            fraction_value = save_checkpoint['fraction']
            # If frac tuple
            if isinstance(fraction_value, tuple):
                numerator, denominator = fraction_value
                checkpoint = (numerator/denominator) * epoch_n
            # If it's a decimal
            else:
                checkpoint = fraction_value * epoch_n
            # Round up to the nearest whole number
            checkpoint = np.ceil(checkpoint)
        else:
            checkpoint = None  # Default case if neither key is found
        if checkpoint is not None:
            printlog(f'\nSaving checkpoints.')
            printlog(f'Checkpoint method: {save_checkpoint}', '\n', 'saving every', checkpoint, '\n')
            
        # If loading
        if load_checkpoint is not None:
            printlog('Loaded a checkpoint for training.')
            printlog(f'Load checkpoint method: {load_checkpoint}')
            start_epoch = self.load_epoch
            printlog('Starting epoch is:', start_epoch,'\n')
        else:
            start_epoch = 0
            
        # if new max epoch (loaded, new goal)
        config_epoch_n = getattr(self, 'config_epoch_n', None)
        if config_epoch_n:
            # this will only exist in the loaded checkpoint case.
            # it will either be the same as the checkpoint epoch_n
            if config_epoch_n == epoch_n:
                pass
            # or it will be greater (pushing the number of epochs now past the original experiment)
            elif config_epoch_n > epoch_n:
                printlog(f"Requested config epoch_n (max epochs): {config_epoch_n} greater than \nyour checkpoint's epoch_n: {epoch_n}.\nRaising model.epoch_n (max epochs) to: {config_epoch_n}")
                # Calculate the number of zeros to add to evaluation lists
                num_zeros_to_add = config_epoch_n - epoch_n
                # Extend each list with the calculated number of zeros
                self.Jr.extend([0] * num_zeros_to_add)
                if classification_enabled:
                    self.Jc.extend([0] * num_zeros_to_add)
                    self.accuracy.extend([0] * num_zeros_to_add)
                self.prediction_unique_count.extend([0] * num_zeros_to_add)
                self.prediction_dominant_class.extend([None] * num_zeros_to_add)
                self.prediction_dominant_count.extend([0] * num_zeros_to_add)
                self.epoch_n = config_epoch_n
                epoch_n = self.epoch_n
                
            # or less, probably accidentally
            else:
                raise ValueError(f'You have loaded a model checkpoint originally set to train to a greater number of epochs: {epoch_n}\n'+ \
                    f'than has now been requested in your config.txt: {config_epoch_n}.\nBoost epoch_n in config greater than {epoch_n} before training.')
        
        num_ts = self.num_ts

        # Epoch '0' evaluation (pre-training, or if checkpoint has been loaded, pre-additional-training)
        Jr0 = 0
        Jc0 = 0 if classification_enabled else None
        accuracy = 0 if classification_enabled else None
        printlog('\n')
        printlog(f'Epoch: {start_epoch}')
        for inp in range(num_inps):
            reset_rs_gteq1(prior_dist=prior_dist)
            input = X[inp]
            label = Y[inp]
            # If non-weight components are being updated only,
            # U, V "timeslices" will all be accessing values from ts == num_ts
            fill_UsVs_with_last_timestep()
            # TS Differentiates it from static
            valid_num_ts = self.valid_timesteps_until_nan(input)
            for ts in range(valid_num_ts):
                self.r[0] = input[:, ts]
                update_non_weight_components(ts=ts, label=label)
            # Rep cost we're interested in every timestep
            Jr0 += rep_cost(input=input, num_ts=valid_num_ts)
            # Classification accuracy is only based on rbar_n at final timestep
            # But Classification Cost we'll do at every timestep, in an effort to keep magnitude of Jr ~ Jc
            if classification_enabled:
                Jc0 += classif_cost(label=label, num_ts=valid_num_ts)
            
        # Do this before running
        evaluation = evaluate(X, Y, return_details=True) if classification_enabled else None
        if classification_enabled:
            accuracy += evaluation['accuracy']
        
        printlog(self.format_epoch_metrics(Jr0, Jc0, accuracy))
        self.Jr[start_epoch] = Jr0
        if classification_enabled:
            self.Jc[start_epoch] = Jc0
            self.accuracy[start_epoch] = accuracy
            self.prediction_unique_count[start_epoch] = evaluation['unique_prediction_count']
            self.prediction_dominant_class[start_epoch] = evaluation['dominant_prediction_class']
            self.prediction_dominant_count[start_epoch] = evaluation['dominant_prediction_count']
        
        # Training
        t_start_train = datetime.now()
        printlog('Training...')
        self.log_training_event(
            f'Training started at {t_start_train.strftime("%Y-%m-%d %H:%M:%S")}'
        )
        # Resume-aware: train only the epochs remaining until epoch_n
        # (fresh runs have start_epoch=0, so behavior there is unchanged).
        for e in range(epoch_n - start_epoch):
            epoch = e + 1 + start_epoch
            scheduled_rate = self.apply_rate_schedule(epoch)
            printlog(f'Epoch {epoch}')
            if scheduled_rate is not None:
                for schedule, param_name, param_key, scheduled_value in scheduled_rate:
                    if epoch in self.rate_schedule_log_epochs(schedule) or epoch == start_epoch + 1:
                        printlog(f'Rate schedule value: {param_name}[{param_key}]={scheduled_value}')
            t_start_epoch = datetime.now()
            Jre = 0
            Jce = 0 if classification_enabled else None
            accuracy = 0 if classification_enabled else None
            # Shuffle X, Y
            shuffle_indices = np.random.permutation(num_inps)
            X_shuff = X[shuffle_indices]
            Y_shuff = Y[shuffle_indices]
            for inp in range(num_inps):
                reset_rs_gteq1(prior_dist=prior_dist)
                input = X_shuff[inp]
                label = Y_shuff[inp]
                valid_num_ts = self.valid_timesteps_until_nan(input)
                for ts in range(valid_num_ts):
                    self.r[0] = input[:, ts]
                    update_all_components(ts=ts, label=label)
                # Propagate final per-ts weights to ts=0 slot so the next word's
                # ts=0 starts from the correct accumulated weight (matching inext's
                # sequential self.U1 = U1_hat.copy() carryover between words).
                fill_UsVs_with_last_timestep()
                Jre += rep_cost(input=input, num_ts=valid_num_ts)
                if classification_enabled:
                    Jce += classif_cost(label=label, num_ts=valid_num_ts)
                
            printlog(f'eval {epoch}')
            evaluation = evaluate(X, Y, return_details=True) if classification_enabled else None
            if classification_enabled:
                accuracy += evaluation['accuracy']
            self.Jr[epoch] = Jre
            if classification_enabled:
                self.Jc[epoch] = Jce
                self.accuracy[epoch] = accuracy
                self.prediction_unique_count[epoch] = evaluation['unique_prediction_count']
                self.prediction_dominant_class[epoch] = evaluation['dominant_prediction_class']
                self.prediction_dominant_count[epoch] = evaluation['dominant_prediction_count']
            
            # Checkpointing
            if checkpoint is not None and epoch % checkpoint == 0:
                checkpoint_time = datetime.now()
                self.log_training_event(
                    f'Checkpoint {epoch} reached at {checkpoint_time.strftime("%Y-%m-%d %H:%M:%S")} '
                    f'(elapsed {self.format_elapsed(t_start_train, checkpoint_time)})'
                )
                # Save checkpoint model
                checkpoint_name = self.generate_output_name(self.mod_name, epoch)
                self.save_model(output_dir=self.checkpoints_dir, output_name=checkpoint_name)
                # Save checkpoint lexical activations (whichever the top layer one hot it: s(rn) for c1, s(rn*Uo) for c2))
                if classification_enabled and self.softmaxed_activation_plots_enabled():
                    self.save_top_layer_softmaxed_activations(X, Y, output_dir=self.diagnostics_dir, output_name=checkpoint_name)
                # Save diagnostics only when a checkpoint is saved.
                self.save_diagnostics(output_dir=self.diagnostics_dir, output_name=checkpoint_name)
                if plot:
                    self.plot_safely(input_dir=self.diagnostics_dir, input_name=checkpoint_name)
                checkpoint_path = join(self.checkpoints_dir, checkpoint_name)
                printlog(f'Checkpoint model saved at epoch {epoch}:\n{checkpoint_path}')
            
            printlog(self.format_epoch_metrics(Jre, Jce, accuracy))
            t_end_epoch = datetime.now()
            printlog(f'Epoch time: {t_end_epoch - t_start_epoch}.')
            printlog(f'Est. time remaining: {(t_end_epoch - t_start_epoch) * (epoch_n - epoch)}.')
            if epoch == 1:
                printlog(f'Est. tot time: {(t_end_epoch - t_start_epoch) * epoch_n}.')
            
        printlog('Training complete.')
        tot_time = t_end_epoch - t_start_train
        printlog(f'Tot time: {tot_time}.')
        if checkpoint is not None:
            final_save_time = datetime.now()
            self.log_training_event(
                f'Final save at {final_save_time.strftime("%Y-%m-%d %H:%M:%S")} '
                f'(elapsed {self.format_elapsed(t_start_train, final_save_time)})'
            )
            printlog('Saving final checkpoint model and diagnostics...')
            final_name = self.generate_output_name(self.mod_name, epoch)
            self.save_model(output_dir=self.models_dir, output_name=final_name)
            if classification_enabled and self.softmaxed_activation_plots_enabled():
                self.save_top_layer_softmaxed_activations(X, Y, output_dir=self.diagnostics_dir, output_name=final_name)
            self.save_diagnostics(output_dir=self.diagnostics_dir, output_name=final_name)
            if plot:
                self.plot_safely(input_dir=self.diagnostics_dir, input_name=final_name)
        else:
            printlog('Checkpoint saving disabled; skipping model checkpoint and diagnostics saves.')
        training_end_time = datetime.now()
        self.log_training_event(
            f'Training finished at {training_end_time.strftime("%Y-%m-%d %H:%M:%S")} '
            f'(elapsed {self.format_elapsed(t_start_train, training_end_time)})'
        )
        
        self.log_final_kpis(epoch)
        
    def evaluate(self, X: np.ndarray, Y: np.ndarray, update_method_name: str, update_method_number: int | float, plot: bool | None = None, return_details: bool = False) -> float | dict[str, Any] | None:
        """Score classification accuracy over X with weights frozen, unfolding timesteps per input.

        Returns:
            Accuracy (float), or a details dict with 'accuracy', 'predictions',
            and prediction-diversity stats when return_details is True; None
            (or a None-filled details dict) when classification is disabled.
        """
        if not self.classification_enabled():
            if return_details:
                return {
                    'accuracy': None,
                    'predictions': None,
                    'unique_prediction_count': None,
                    'dominant_prediction_class': None,
                    'dominant_prediction_count': None,
                }
            return None
        
        
        prior_dist = self.recurrent_reset_prior_sampler()
        
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        fill_UsVs_with_last_timestep = self.fill_UsVs_with_last_timestep
        
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        classify = self.classify
        num_inps = self.num_inps
        
        accuracy = 0
        predictions = []
        for inp in range(num_inps):
            reset_rs_gteq1(prior_dist=prior_dist)
            input = X[inp]
            label = Y[inp]
            # If non-weight components are being updated only,
            # U, V "timeslices" will all be accessing values from ts == num_ts
            fill_UsVs_with_last_timestep()
            valid_num_ts = self.valid_timesteps_until_nan(input)
            for ts in range(valid_num_ts):
                self.r[0] = input[:, ts]
                update_non_weight_components(ts=ts, label=label)
            guess = classify(label, num_ts=valid_num_ts)
            accuracy += guess
            # Track the argmax prediction only when it is unique.
            probs = self.top_layer_softmaxed_activation_vector(valid_num_ts - 1)
            if np.sum(probs == probs.max()) == 1:
                predictions.append(int(np.argmax(probs)))
            else:
                predictions.append(None)
        accuracy /= num_inps

        if return_details:
            valid_predictions = [prediction for prediction in predictions if prediction is not None]
            if valid_predictions:
                unique_predictions, prediction_counts = np.unique(valid_predictions, return_counts=True)
                dominant_index = int(np.argmax(prediction_counts))
                dominant_class = int(unique_predictions[dominant_index])
                dominant_count = int(prediction_counts[dominant_index])
                unique_prediction_count = int(len(unique_predictions))
            else:
                dominant_class = None
                dominant_count = 0
                unique_prediction_count = 0
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'unique_prediction_count': unique_prediction_count,
                'dominant_prediction_class': dominant_class,
                'dominant_prediction_count': dominant_count,
            }
    
        return accuracy

    def top_layer_softmaxed_activation_vector(self, ts: int) -> np.ndarray:
        """Return the eval-softmaxed top-layer activation at a timestep.

        Uses rhat (the corrected state), matching legacy softmaxd_r2_hat
        documentation/scoring; c1 softmaxes r_n, c2 softmaxes Uo . r_n.

        Raises:
            ValueError: When classif_method is not c1 or c2.
        """
        top_r = self.rhat[self.num_layers][:, ts]
        if self.classif_method == 'c1':
            vector = top_r
        elif self.classif_method == 'c2':
            vector = np.matmul(self.Uhat['o'][:, :, ts], top_r)
        else:
            raise ValueError(f'Top-layer activations require classif_method c1 or c2, got {self.classif_method}.')
        return self.softmax_func_eval(vector=vector)

    def save_top_layer_softmaxed_activations(self, X: np.ndarray, Y: np.ndarray, output_dir: str, output_name: str) -> None:
        """Save per-class softmaxed activation trajectories for correctly classified inputs.

        Runs each input twice with weights frozen: once to check correctness,
        once to record the (num_classes, num_ts) activation trajectory
        (NaN-padded past the input's valid timesteps). Also computes the TCRU
        averages.

        Raises:
            ValueError: When two correct inputs share the same label index.
        """
        if not self.softmaxed_activation_plots_enabled():
            return

        epoch_dir = self.epoch_artifact_dir(output_dir, output_name)
        output_path = join(epoch_dir, self.activations_file_name(output_name))

        prior_dist = self.recurrent_reset_prior_sampler()
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        fill_UsVs_with_last_timestep = self.fill_UsVs_with_last_timestep

        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)

        softmaxed_activations_over_all_correct_inputs = {}
        num_correct_guesses = 0

        for inp in range(self.num_inps):
            input = X[inp]
            label = Y[inp]

            # First pass: check correctness.
            reset_rs_gteq1(prior_dist=prior_dist)
            fill_UsVs_with_last_timestep()
            valid_num_ts = self.valid_timesteps_until_nan(input)
            for ts in range(valid_num_ts):
                self.r[0] = input[:, ts]
                update_non_weight_components(ts=ts, label=label)

            if self.classify(label, num_ts=valid_num_ts) != 1:
                continue

            num_correct_guesses += 1

            # Second pass: record the activation trajectory.
            reset_rs_gteq1(prior_dist=prior_dist)
            fill_UsVs_with_last_timestep()
            softmaxed_activation_vectors = []
            for ts in range(valid_num_ts):
                self.r[0] = input[:, ts]
                update_non_weight_components(ts=ts, label=label)
                softmaxed_activation_vectors.append(self.top_layer_softmaxed_activation_vector(ts))

            if valid_num_ts < self.num_ts:
                softmaxed_activation_vectors.extend(
                    [np.full(self.num_classes, np.nan) for _ in range(self.num_ts - valid_num_ts)]
                )

            label_key = int(np.argmax(label))
            if label_key in softmaxed_activations_over_all_correct_inputs:
                raise ValueError(
                    f'Multiple correct inputs produced label index {label_key}; '
                    'TCRU activation saving currently expects one correct input per class.'
                )
            softmaxed_activations_over_all_correct_inputs[label_key] = np.array(softmaxed_activation_vectors).T

        self.log_correct_guess_count(num_correct_guesses, self.num_inps)
        if num_correct_guesses == 0:
            self.print_and_log('no correct model classifications: saving empty activation dict')
        elif num_correct_guesses == 1:
            self.print_and_log('only one correct model classification: saving activation dict with only one TCRU activation matrix')
            
        self.softmaxed_activations_over_all_correct_inputs = softmaxed_activations_over_all_correct_inputs
        self.calculate_average_softmaxed_activations_over_correct_inputs()
        with open(output_path, 'wb') as f:
            pickle.dump(softmaxed_activations_over_all_correct_inputs, f)
    
    def reset_rs_gteq1(self, all_lyr_sizes: list[int | tuple[int, ...]], prior_dist: Callable[..., np.ndarray]) -> None:
        """Re-sample r[1..n] and r['c'] from the prior and re-initialize the rhat/rbar trajectories."""
        # Reset once per input: explicit plain recurrent state then explicit
        # recurrent trajectory initialization from that state.
        n = self.num_layers
        for i in range(1, n + 1):
            self.r[i] = prior_dist(size=all_lyr_sizes[i - 1])
        self.r['c'] = prior_dist(size=all_lyr_sizes[n - 1])

        # Explicitly initialize rhat/rbar trajectories for this input.
        # ts=0 rhat starts from the freshly reset recurrent state; all later
        # timesteps are overwritten by update equations as the input unfolds.
        for i in range(1, n + 1):
            self.rhat[i][:, :] = 0
            self.rbar[i][:, :] = 0
            self.rhat[i][:, 0] = np.array(self.r[i], copy=True)
        self.rhat['c'][:, :] = 0
        self.rbar['c'][:, :] = 0
        self.rhat['c'][:, 0] = np.array(self.r['c'], copy=True)

    def update_method_rW_seq_niters(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]], ts: int) -> None:
        """Apply sequential r-then-weight updates at one timestep (Li standard); niters is always 1 in rPCC."""
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        for _ in range(niters):
            r_updates(ts=ts, label=label)
            for w in range_num_weight_updates:
                # For as many weight sets are there are to update, update them.
                weight_updates[w](ts=ts, label=label)

    def update_method_r_niters(self, niters: int, label: np.ndarray, component_updates: list[Callable[..., None]], ts: int) -> None:
        """Apply the r-only update at one timestep; niters is always 1 in rPCC (Li)."""
        r_updates = component_updates[0]
        
        for _ in range(niters):
            r_updates(ts=ts, label=label)


def create_model(params: dict[str, Any]) -> PredictiveCodingClassifier:
    """Instantiate StaticPCC or RecurrentPCC from params['model_type'].

    Raises:
        ValueError: For unsupported model_type values.
    """
    model_classes = {
        'static': StaticPCC,
        'recurrent': RecurrentPCC,
    }
    model_type = params.get('model_type')
    try:
        model_class = model_classes[model_type]
    except KeyError as error:
        supported = ', '.join(sorted(model_classes))
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Expected one of: {supported}."
        ) from error
    return model_class(params)