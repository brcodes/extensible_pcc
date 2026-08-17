"""Focused tests for PCC factory and concrete-class construction.

Script-style suite validating construction soundness for StaticPCC and
RecurrentPCC: factory-versus-direct construction equivalence, a feature
matrix of "connector" call-chain tests (update dispatch -> costs ->
classify/evaluate, with determinism fingerprints), invalid-combination
rejection, config-key validation via load_params, and miniature end-to-end
training-pipeline plumbing (checkpoints, artifacts, reload, resume,
determinism). Invoke from the pypredcoding root:
``python tests/test_model_construction.py``; prints "status: PASS" on
success. See TEST_CONSTRUCTION_README.md for full documentation.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import pickle
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PYPREDCODING_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PYPREDCODING_DIR.parent
for import_path in (ROOT_DIR, PYPREDCODING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from model import (  # type: ignore  # noqa: E402
    PredictiveCodingClassifier,
    RecurrentPCC,
    StaticPCC,
    create_model,
)
from pypredcoding.train_model import load_params  # type: ignore  # noqa: E402


CONFIG_DIR = PYPREDCODING_DIR / 'config'


def normalize_top_layer_param(params: dict) -> dict:
    """Return a copy of params with the legacy output_lyr_size key renamed to top_lyr_size."""
    params = dict(params)
    if 'output_lyr_size' in params:
        params['top_lyr_size'] = params.pop('output_lyr_size')
    return params


def load_test_params(config_name: str) -> dict:
    """Load a config from config/ into a params dict bounded to one epoch."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'config.txt'
        temp_path.write_text(load_config_text(config_name))
        params = load_params(str(temp_path))
    params['epoch_n'] = 1
    return params


def assert_same_initial_weights(
    factory_model: PredictiveCodingClassifier,
    direct_model: PredictiveCodingClassifier,
) -> None:
    """Assert factory- and directly-constructed models share identical initial weights."""
    for layer in range(1, factory_model.num_layers + 1):
        if not np.array_equal(factory_model.U[layer], direct_model.U[layer]):
            raise AssertionError(f'U[{layer}] differs between factory and direct construction')
        if isinstance(factory_model, RecurrentPCC):
            if not np.array_equal(factory_model.V[layer], direct_model.V[layer]):
                raise AssertionError(f'V[{layer}] differs between factory and direct construction')


def assert_construction(config_name: str, expected_class: type) -> None:
    """Assert create_model builds the expected class with weights matching direct construction."""
    params = load_test_params(config_name)

    np.random.seed(11)
    factory_model = create_model(params)
    if type(factory_model) is not expected_class:
        raise AssertionError(f'Expected {expected_class.__name__}, got {type(factory_model).__name__}')

    np.random.seed(11)
    direct_model = expected_class(params)
    assert_same_initial_weights(factory_model, direct_model)


def load_config_text(config_name: str) -> str:
    """Read a config file's text, renaming a legacy output_lyr_size key to top_lyr_size."""
    return re.sub(
        r'^(\s*)output_lyr_size\s*=',
        r'\1top_lyr_size=',
        (CONFIG_DIR / config_name).read_text(),
        count=1,
        flags=re.MULTILINE,
    )


def set_config_literal(config_text: str, key: str, literal_text: str) -> str:
    """Replace the single key=value assignment for key in config text.

    Raises:
        AssertionError: If the key does not appear exactly once.
    """
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*.*$', re.MULTILINE)
    replacement = f'{key}={literal_text}'
    updated_text, replacement_count = pattern.subn(replacement, config_text, count=1)
    if replacement_count != 1:
        raise AssertionError(f'Expected exactly one assignment for {key}, found {replacement_count}.')
    return updated_text


def delete_config_key(config_text: str, key: str) -> str:
    """Remove the single key=value assignment for key from config text.

    Raises:
        AssertionError: If the key does not appear exactly once.
    """
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*.*\n?', re.MULTILINE)
    updated_text, replacement_count = pattern.subn('', config_text, count=1)
    if replacement_count != 1:
        raise AssertionError(f'Expected exactly one assignment for {key}, found {replacement_count}.')
    return updated_text


def assert_load_params_fails(config_text: str, expected_parts: Sequence[str]) -> None:
    """Assert load_params rejects the config text with a ValueError naming each expected part."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'config.txt'
        temp_path.write_text(config_text)

        try:
            load_params(str(temp_path))
        except ValueError as error:
            message = str(error)
            for part in expected_parts:
                if part not in message:
                    raise AssertionError(f'Missing {part!r} in error: {message}')
        else:
            raise AssertionError('Expected load_params to raise ValueError')


def assert_load_params_and_model_succeed(config_text: str) -> None:
    """Assert the config text both parses and constructs a model without error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'config.txt'
        temp_path.write_text(config_text)
        params = normalize_top_layer_param(load_params(str(temp_path)))
        create_model(params)


def assert_load_params_succeeds_model_fails(config_text: str, expected_parts: Sequence[str]) -> None:
    """Assert the config text parses but create_model raises a ValueError naming each expected part."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'config.txt'
        temp_path.write_text(config_text)
        params = normalize_top_layer_param(load_params(str(temp_path)))
        try:
            create_model(params)
        except ValueError as error:
            message = str(error)
            for part in expected_parts:
                if part not in message:
                    raise AssertionError(f'Missing {part!r} in error: {message}')
        else:
            raise AssertionError('Expected create_model to raise ValueError')


def assert_static_one_layer_constructs_from_frozen_baseline(config_name: str) -> None:
    """Assert a one-layer non-classifying StaticPCC constructs from a frozen baseline config."""
    params = load_test_params(config_name)
    params['num_layers'] = 1
    params['hidden_lyr_sizes'] = []
    params['classif_method'] = None
    params['kr'] = {1: params['kr'][1]}
    params['kU'] = {1: params['kU'][1]}
    params['alph'] = {1: params['alph'][1]}
    params['lam'] = {1: params['lam'][1]}
    params['ssq'] = {0: params['ssq'][0], 1: params['ssq'][1]}

    model = create_model(params)
    if model.num_layers != 1:
        raise AssertionError('Expected one-layer static model')
    if model.all_lyr_sizes != [model.r[1].shape]:
        raise AssertionError(f'Unexpected one-layer static all_lyr_sizes: {model.all_lyr_sizes}')
    if 1 not in model.r or 1 not in model.U:
        raise AssertionError('One-layer static model missing r[1] or U[1]')


def assert_static_c2_allows_nonclass_top_size() -> None:
    """Assert c2 classification permits top_lyr_size != num_classes and shapes the top-down signal."""
    params = load_test_params('config_sPCC_Rogers_2026_trace212_frozen_baseline.txt')
    params['classif_method'] = 'c2'
    params['top_lyr_size'] = 16
    params['kU'] = {1: params['kU'][1], 2: params['kU'][2], 3: params['kU'][3], 'o': 0.001}
    params['lam'] = {1: params['lam'][1], 2: params['lam'][2], 3: params['lam'][3], 'o': 0.001}

    model = create_model(params)
    label = np.zeros(model.num_classes)
    label[0] = 1
    signal = model.r_updates.__self__.rn_topdown_upd_c2(label)
    if signal.shape != model.r[model.num_layers].shape:
        raise AssertionError(
            f'C2 top-layer signal shape {signal.shape} does not match top layer {model.r[model.num_layers].shape}'
        )


def assert_two_layer_reshaped_static_u2_flattens_expanded_r1() -> None:
    """Assert the reshaped 1->2 connection flattens an expanded r[1] into a 2-D U[2]."""
    params = load_test_params('config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt')
    params['num_layers'] = 2
    params['hidden_lyr_sizes'] = [32]
    params['top_lyr_size'] = params['num_classes']
    params['kr'] = {1: params['kr'][1], 2: params['kr'][2]}
    params['kU'] = {1: params['kU'][1], 2: params['kU'][2]}
    params['alph'] = {1: params['alph'][1], 2: params['alph'][2]}
    params['lam'] = {1: params['lam'][1], 2: params['lam'][2]}
    params['ssq'] = {0: params['ssq'][0], 1: params['ssq'][1], 2: params['ssq'][2]}

    model = create_model(params)
    expected_shape = (np.prod(model.r[1].shape), model.r[2].shape[0])
    if model.U[2].shape != expected_shape:
        raise AssertionError(f'Expected reshaped U[2] shape {expected_shape}, got {model.U[2].shape}')

    U2r2 = model.r_updates.__self__.U2r2_e1l_Li(model.U[2], model.r[2])
    if U2r2.shape != model.r[1].shape:
        raise AssertionError(f'Expected U2r2 shape {model.r[1].shape}, got {U2r2.shape}')


def assert_recurrent_one_layer_constructs_from_frozen_baseline() -> None:
    """Assert a one-layer RecurrentPCC constructs with all state/weight collections and 'c' context."""
    params = load_test_params('config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt')
    params['num_layers'] = 1
    params['hidden_lyr_sizes'] = []
    params['kr'] = {1: params['kr'][1]}
    params['kU'] = {1: params['kU'][1]}
    params['kV'] = {1: params['kV'][1]}
    params['ssqr'] = {0: params['ssqr'][0], 1: params['ssqr'][1]}
    params['ssqV'] = {1: params['ssqV'][1]}

    model = create_model(params)
    if model.num_layers != 1:
        raise AssertionError('Expected one-layer recurrent model')
    for collection_name in ('r', 'U', 'V', 'rhat', 'rbar', 'Uhat', 'Ubar', 'Vhat', 'Vbar'):
        collection = getattr(model, collection_name)
        if 1 not in collection:
            raise AssertionError(f'One-layer recurrent model missing {collection_name}[1]')
    if 'c' not in model.r or 'c' not in model.rhat or 'c' not in model.rbar:
        raise AssertionError('One-layer recurrent model missing r_context state')


def assert_no_classification_metrics_absent() -> None:
    """Assert non-classifying (classif_method=None) models skip Jc/accuracy yet still plot and save diagnostics."""
    static_params = load_test_params('config_sPCC_Rogers_2026_trace212_frozen_baseline.txt')
    static_params['classif_method'] = None
    static_model = create_model(static_params)
    if static_model.Jc is not None or static_model.accuracy is not None:
        raise AssertionError('Static NC model should not allocate Jc/accuracy histories')
    if static_model.evaluate(np.zeros((static_model.num_imgs, *static_model.input_shape)), np.zeros((static_model.num_imgs, static_model.num_classes)), 'rW_instdeltas_niters', 1) is not None:
        raise AssertionError('Static NC evaluate should return absent accuracy')
    static_model.Jr[1] = 1.0
    with tempfile.TemporaryDirectory() as temp_dir:
        diagnostics_dir = Path(temp_dir) / 'diagnostics'
        plots_dir = Path(temp_dir) / 'plots'
        static_model.plots_dir = str(plots_dir)
        output_name = static_model.generate_output_name('tmp_static_nc.pydb', 1)
        static_model.save_diagnostics(str(diagnostics_dir), output_name)
        static_model.plot(str(diagnostics_dir), output_name)
        if not (plots_dir / '1' / ('eval-accuracy-during-training.diagnostics.' + output_name + '.png')).exists():
            raise AssertionError('Static NC plot was not saved')

    static_params_rf = load_test_params('config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt')
    static_params_rf['classif_method'] = None
    static_model_rf = create_model(static_params_rf)
    static_model_rf.Jr[1] = 1.0
    with tempfile.TemporaryDirectory() as temp_dir:
        diagnostics_dir = Path(temp_dir) / 'diagnostics'
        plots_dir = Path(temp_dir) / 'plots'
        static_model_rf.plots_dir = str(plots_dir)
        output_name = static_model_rf.generate_output_name('tmp_static_rf.pydb', 1)
        static_model_rf.save_diagnostics(str(diagnostics_dir), output_name)
        static_model_rf.plot_safely(str(diagnostics_dir), output_name)
        expected_rf_path = plots_dir / '1' / 'receptive_fields' / f'receptive-fields-image0.{output_name}.png'
        if not expected_rf_path.exists():
            raise AssertionError(f'Static RF plot was not saved at {expected_rf_path}')

    static_params_rf_xonly = load_test_params('config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt')
    static_params_rf_xonly['classif_method'] = None
    static_model_rf_xonly = create_model(static_params_rf_xonly)
    static_model_rf_xonly.Jr[1] = 1.0
    x_only, _ = static_model_rf_xonly.load_configured_dataset()
    with tempfile.TemporaryDirectory() as temp_dir:
        diagnostics_dir = Path(temp_dir) / 'diagnostics'
        plots_dir = Path(temp_dir) / 'plots'
        dataset_path = Path(temp_dir) / 'x_only_dataset.pydb'
        with dataset_path.open('wb') as dataset_file:
            pickle.dump(np.asarray(x_only), dataset_file)
        static_model_rf_xonly.dataset_train = str(dataset_path)
        static_model_rf_xonly.plots_dir = str(plots_dir)
        output_name = static_model_rf_xonly.generate_output_name('tmp_static_rf_xonly.pydb', 1)
        static_model_rf_xonly.save_diagnostics(str(diagnostics_dir), output_name)
        static_model_rf_xonly.plot_safely(str(diagnostics_dir), output_name)
        expected_rf_path = plots_dir / '1' / 'receptive_fields' / f'receptive-fields-image0.{output_name}.png'
        if not expected_rf_path.exists():
            raise AssertionError(f'Static X-only RF plot was not saved at {expected_rf_path}')

    try:
        static_model_rf.render_static_array_to_2d(np.zeros((15, 15, 256), dtype=np.float32))
    except ValueError as exc:
        if 'Static tiled-flat rendering expects shape' not in str(exc):
            raise AssertionError(f'Unexpected tiled-flat mode validation error: {exc}')
    else:
        raise AssertionError('Expected tiled-flat RF rendering to reject tiled-expanded-shaped data')

    recurrent_params = load_test_params('config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt')
    recurrent_params['classif_method'] = None
    recurrent_model = create_model(recurrent_params)
    if recurrent_model.Jc is not None or recurrent_model.accuracy is not None:
        raise AssertionError('Recurrent NC model should not allocate Jc/accuracy histories')
    details = recurrent_model.evaluate(
        np.zeros((recurrent_model.num_inps, recurrent_model.input_shape[0], recurrent_model.num_ts)),
        np.zeros((recurrent_model.num_inps, recurrent_model.num_classes)),
        'rW_seq_niters',
        1,
        return_details=True,
    )
    if details['accuracy'] is not None or details['predictions'] is not None:
        raise AssertionError('Recurrent NC evaluate should return absent accuracy and predictions')

# ---------------------------------------------------------------------------
# Feature-matrix connector tests + training/eval plumbing tests
#
# Strategy: instead of full training runs over every feature combination,
# each valid combination gets a small "connector" test that exercises the
# exact call chain train()/evaluate() consume: construction -> r/weight
# update dispatch -> rep/classif costs -> classification guess -> eval-path
# (no-weight) updates. Each connector asserts architectural soundness
# (shapes), numerical soundness (finiteness), eval weight-freezing, and
# seeded determinism. Full train/checkpoint/load/artifact plumbing is then
# verified once per model type on a representative config (Part 2), since
# training is just these connectors plus data/artifact handling.
# ---------------------------------------------------------------------------

STATIC_BASE_CONFIG = 'config_sPCC_Rogers_2026_trace212_frozen_baseline.txt'
RECURRENT_BASE_CONFIG = 'config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt'


def assert_finite(name: str, value: Any) -> None:
    """Assert an array-like contains only finite values."""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise AssertionError(f'{name} contains non-finite values')


def layer_rate_dict(num_layers: int, value: float, include_output: bool = False) -> dict:
    """Build a per-layer rate dict {1..num_layers: value}, optionally adding an 'o' key."""
    rates = {i: value for i in range(1, num_layers + 1)}
    if include_output:
        rates['o'] = value
    return rates


def one_hot_labels(num_inps: int, num_classes: int) -> np.ndarray:
    """Build cycling one-hot labels of shape (num_inps, num_classes)."""
    labels = np.zeros((num_inps, num_classes))
    for i in range(num_inps):
        labels[i, i % num_classes] = 1.0
    return labels


def make_static_params(*, num_layers: int, hidden: Sequence[int], top: int,
                       input_shape: Sequence[int], ntiles: int | None, flat: bool,
                       arch: str, conn: str, classif: str | None, activ: str = 'linear',
                       prior: str = 'gaussian', num_classes: int | None = None,
                       update_method: dict | None = None,
                       r_prior_dist: tuple | None = None,
                       num_inps: int = 2, epoch_n: int = 1) -> dict:
    """Build a small StaticPCC params dict from the frozen trace212 baseline plus overrides."""
    params = load_test_params(STATIC_BASE_CONFIG)
    include_output = classif == 'c2'
    if num_classes is None:
        num_classes = top if classif == 'c1' else 4
    params.update({
        'num_layers': num_layers,
        'hidden_lyr_sizes': list(hidden),
        'top_lyr_size': top,
        'input_shape': tuple(input_shape),
        'ntiles_per_input': ntiles,
        'flat_input': flat,
        'architecture': arch,
        'first_to_second_lyr_connection_architecture': conn,
        'classif_method': classif,
        'activ_func': activ,
        'num_inps': num_inps,
        'num_classes': num_classes,
        'epoch_n': epoch_n,
        'update_method': dict(update_method) if update_method else {'rW_instdeltas_niters': 2},
        'kr': layer_rate_dict(num_layers, 0.05),
        'kU': layer_rate_dict(num_layers, 0.05, include_output=include_output),
        'alph': layer_rate_dict(num_layers, 0.02),
        'lam': layer_rate_dict(num_layers, 0.02, include_output=include_output),
        'ssq': {i: 1.0 for i in range(0, num_layers + 1)},
        'r_prior_dist': r_prior_dist if r_prior_dist is not None else ('pseudo_sparse_kurtotic',),
        'r_prior_cost': prior,
        'r_prior_cost_denominator': 2,
        'U_prior_dist': ('gaussian', 0.0, 0.1),
        'U_prior_cost': prior,
        'U_prior_cost_denominator': 2,
        'plot_train': False,
        'plot_receptive_fields': False,
        'plot_softmaxed_activations': False,
        'save_checkpoint': {},
        'load_checkpoint': None,
    })
    return params


def make_recurrent_params(*, num_layers: int, hidden: Sequence[int], top: int,
                          classif: str | None, activ: str = 'linear',
                          softmax_type: str = 'stable', num_classes: int | None = None,
                          input_size: int = 8, num_ts: int = 5, num_inps: int = 2,
                          epoch_n: int = 1) -> dict:
    """Build a small RecurrentPCC params dict from the frozen cvcv12 baseline plus overrides."""
    params = load_test_params(RECURRENT_BASE_CONFIG)
    include_output = classif == 'c2'
    if num_classes is None:
        num_classes = top if classif == 'c1' else 4
    params.update({
        'num_layers': num_layers,
        'hidden_lyr_sizes': list(hidden),
        'top_lyr_size': top,
        'input_shape': (input_size,),
        'num_ts': num_ts,
        'classif_method': classif,
        'activ_func': activ,
        'softmax_type': softmax_type,
        'softmax_k_eval': 20,
        'num_inps': num_inps,
        'num_classes': num_classes,
        'epoch_n': epoch_n,
        'update_method': {'rW_seq_niters': 1},
        'kr': layer_rate_dict(num_layers, 0.05),
        'kU': layer_rate_dict(num_layers, 0.05, include_output=include_output),
        'kV': layer_rate_dict(num_layers, 0.05),
        'ssqr': {i: 1.0 for i in range(0, num_layers + 1)},
        'ssqV': {i: 1.0 for i in range(1, num_layers + 1)},
        'plot_train': False,
        'plot_softmaxed_activations': False,
        'save_checkpoint': {},
        'load_checkpoint': None,
    })
    return params


def static_state_fingerprint(model: StaticPCC, extra_scalars: Sequence[float] = ()) -> np.ndarray:
    """Concatenate all r/U states (plus U['o'] under c2) and extra scalars into a determinism fingerprint."""
    pieces = []
    for i in range(1, model.num_layers + 1):
        pieces.append(np.ravel(model.r[i]))
        pieces.append(np.ravel(model.U[i]))
    if model.classif_method == 'c2':
        pieces.append(np.ravel(model.U['o']))
    pieces.append(np.asarray(extra_scalars, dtype=np.float64))
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in pieces])


def recurrent_state_fingerprint(model: RecurrentPCC, extra_scalars: Sequence[float] = ()) -> np.ndarray:
    """Concatenate all bar/hat states and weights (plus Uhat['o'] under c2) and extra scalars into a determinism fingerprint."""
    pieces = []
    for i in range(1, model.num_layers + 1):
        for collection in (model.rbar, model.rhat, model.Uhat, model.Vhat):
            pieces.append(np.ravel(collection[i]))
    pieces.append(np.ravel(model.rbar['c']))
    pieces.append(np.ravel(model.rhat['c']))
    if model.classif_method == 'c2':
        pieces.append(np.ravel(model.Uhat['o']))
    pieces.append(np.asarray(extra_scalars, dtype=np.float64))
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in pieces])


def assert_static_architecture_sound(model: StaticPCC, spec: dict) -> None:
    """Assert layer bookkeeping, U/r shape agreement, and any expected shapes from the spec."""
    if len(model.all_lyr_sizes) != model.num_layers:
        raise AssertionError('all_lyr_sizes length does not match num_layers')
    input_shape = tuple(model.input_shape)
    if tuple(model.U[1].shape[:len(input_shape)]) != input_shape:
        raise AssertionError(f'U[1] leading dims {model.U[1].shape} do not match input shape {input_shape}')
    if model.U[1].shape[-1] != model.r[1].shape[-1]:
        raise AssertionError('U[1] trailing dim does not match r[1] trailing dim')
    expected_r1_shape = spec.get('expected_r1_shape')
    if expected_r1_shape is not None and tuple(model.r[1].shape) != tuple(expected_r1_shape):
        raise AssertionError(f'r[1] shape {model.r[1].shape} != expected {expected_r1_shape}')
    expected_u2_shape = spec.get('expected_U2_shape')
    if expected_u2_shape is not None and tuple(model.U[2].shape) != tuple(expected_u2_shape):
        raise AssertionError(f'U[2] shape {model.U[2].shape} != expected {expected_u2_shape}')
    if model.num_layers >= 2 and model.classif_method in ('c1', 'c2'):
        if model.r[model.num_layers].shape != (model.top_lyr_size,):
            raise AssertionError('Classifying static model must have a vector top layer')
    if model.classif_method == 'c2' and model.U['o'].shape != (model.num_classes, model.top_lyr_size):
        raise AssertionError(f"U['o'] shape {model.U['o'].shape} != (num_classes, top_lyr_size)")


def run_static_connector(spec: dict, seed: int = 7) -> np.ndarray:
    """Construct and drive the exact train()/evaluate() call chain for one static feature combo.

    Returns:
        Determinism fingerprint of final states, weights, and cost/accuracy scalars.
    """
    params = make_static_params(**{k: v for k, v in spec.items()
                                   if k not in ('name', 'expected_r1_shape', 'expected_U2_shape')})
    np.random.seed(seed)
    model = create_model(params)
    if type(model) is not StaticPCC:
        raise AssertionError('Static connector did not build a StaticPCC')
    assert_static_architecture_sound(model, spec)

    rng = np.random.RandomState(seed + 1)
    X = rng.uniform(0.05, 0.95, size=(model.num_inps,) + tuple(model.input_shape))
    Y = one_hot_labels(model.num_inps, model.num_classes)

    update_name = next(iter(model.update_method))
    update_number = model.update_method[update_name]
    update_all = partial(model.update_method_dict[update_name], update_number)
    prior_dist = model.static_reset_prior_sampler()

    # Training-path connector: r + weight updates, then costs.
    for inp in range(model.num_inps):
        model.reset_rs_gteq1(model.all_lyr_sizes, prior_dist=prior_dist)
        model.r[0] = X[inp]
        update_all(label=Y[inp])
    scalars = [model.rep_cost()]
    if model.classification_enabled():
        scalars.append(model.classif_cost(Y[0]))

    # Eval-path connector: no-weight updates must leave weights untouched.
    weights_before = {key: np.array(value, copy=True) for key, value in model.U.items()}
    update_eval = partial(model.update_method_no_weight_dict[update_name], update_number)
    for inp in range(model.num_inps):
        model.reset_rs_gteq1(model.all_lyr_sizes, prior_dist=prior_dist)
        model.r[0] = X[inp]
        update_eval(label=np.zeros_like(Y[inp]))
        if model.classification_enabled():
            guess = model.classify(Y[inp])
            if guess not in (0, 1):
                raise AssertionError(f'classify returned {guess}, expected 0 or 1')
    for key, before in weights_before.items():
        if not np.array_equal(before, model.U[key]):
            raise AssertionError(f'Eval path modified U[{key}]')

    if model.classification_enabled():
        accuracy = model.evaluate(X, Y, update_name, update_number)
        if not (0.0 <= accuracy <= 1.0):
            raise AssertionError(f'evaluate accuracy out of range: {accuracy}')
        scalars.append(accuracy)
    else:
        if model.evaluate(X, Y, update_name, update_number) is not None:
            raise AssertionError('evaluate must return None when classification is disabled')

    for name in ('rep/classif costs',):
        assert_finite(name, scalars)
    fingerprint = static_state_fingerprint(model, extra_scalars=scalars)
    assert_finite(f"static connector '{spec['name']}' state", fingerprint)
    return fingerprint


def assert_recurrent_architecture_sound(model: RecurrentPCC, spec: dict) -> None:
    """Assert per-layer bar/hat state shapes, weight presence, 'c' context, and c2 output weights."""
    if len(model.all_lyr_sizes) != model.num_layers:
        raise AssertionError('all_lyr_sizes length does not match num_layers')
    n = model.num_layers
    for i in range(1, n + 1):
        expected_size = model.all_lyr_sizes[i - 1]
        for name, collection in (('rbar', model.rbar), ('rhat', model.rhat)):
            if collection[i].shape != (expected_size, model.num_ts):
                raise AssertionError(f'{name}[{i}] shape {collection[i].shape} != ({expected_size}, num_ts)')
        for name, collection in (('Uhat', model.Uhat), ('Vhat', model.Vhat)):
            if i not in collection:
                raise AssertionError(f'Missing {name}[{i}]')
    for name, collection in (('rbar', model.rbar), ('rhat', model.rhat)):
        if 'c' not in collection:
            raise AssertionError(f"Missing {name}['c'] context state")
    if model.classif_method == 'c2' and 'o' not in model.Uhat:
        raise AssertionError("c2 recurrent model missing Uhat['o']")


def run_recurrent_connector(spec: dict, seed: int = 7) -> np.ndarray:
    """Construct and drive the exact train()/evaluate() call chain for one recurrent feature combo.

    Returns:
        Determinism fingerprint of final states, weights, and cost/accuracy scalars.
    """
    params = make_recurrent_params(**{k: v for k, v in spec.items() if k != 'name'})
    np.random.seed(seed)
    model = create_model(params)
    if type(model) is not RecurrentPCC:
        raise AssertionError('Recurrent connector did not build a RecurrentPCC')
    assert_recurrent_architecture_sound(model, spec)

    rng = np.random.RandomState(seed + 1)
    X = rng.uniform(0.05, 0.95, size=(model.num_inps, model.input_shape[0], model.num_ts))
    Y = one_hot_labels(model.num_inps, model.num_classes)

    update_name = next(iter(model.update_method))
    update_number = model.update_method[update_name]
    update_all = partial(model.update_method_dict[update_name], update_number)
    prior_dist = model.recurrent_reset_prior_sampler()

    # Training-path connector: full sequential r/U/V (and Uo for c2) updates.
    scalars = []
    for inp in range(model.num_inps):
        model.reset_rs_gteq1(model.all_lyr_sizes, prior_dist=prior_dist)
        for ts in range(model.num_ts):
            model.r[0] = X[inp][:, ts]
            update_all(ts=ts, label=Y[inp])
        model.fill_UsVs_with_last_timestep()
        scalars.append(model.rep_cost(input=X[inp], num_ts=model.num_ts))
        if model.classification_enabled():
            scalars.append(model.classif_cost(label=Y[inp], num_ts=model.num_ts))

    # Eval-path connector: evaluate() must not modify weights.
    weights_before = {key: np.array(value, copy=True) for key, value in model.Uhat.items()}
    v_before = {key: np.array(value, copy=True) for key, value in model.Vhat.items()}
    details = model.evaluate(X, Y, update_name, update_number, return_details=True)
    for key, before in weights_before.items():
        if not np.array_equal(before, model.Uhat[key]):
            raise AssertionError(f'Recurrent evaluate modified Uhat[{key}]')
    for key, before in v_before.items():
        if not np.array_equal(before, model.Vhat[key]):
            raise AssertionError(f'Recurrent evaluate modified Vhat[{key}]')
    if model.classification_enabled():
        if not (0.0 <= details['accuracy'] <= 1.0):
            raise AssertionError(f"evaluate accuracy out of range: {details['accuracy']}")
        if len(details['predictions']) != model.num_inps:
            raise AssertionError('evaluate must return one prediction per input')
        scalars.append(details['accuracy'])
    else:
        if details['accuracy'] is not None or details['predictions'] is not None:
            raise AssertionError('Non-classifying recurrent evaluate must return absent metrics')

    assert_finite('rep/classif costs', scalars)
    fingerprint = recurrent_state_fingerprint(model, extra_scalars=scalars)
    assert_finite(f"recurrent connector '{spec['name']}' state", fingerprint)
    return fingerprint


# Feature matrix legend (valid combinations only):
#   A model type: static vs recurrent      B layer count: n = 1, 2, 3, 4
#   C input type: tiled/untiled x flat/nonflat (static; recurrent is vector)
#   D first-to-second connection: structured vs reshaped
#   E architecture: expand_first_lyr vs flat_hidden_lyrs
#   F classification: c1, c2, None         G activation: linear, tanh
#   H prior costs: gaussian, sparse_kurtotic (static only; rPCC requires None)
STATIC_CONNECTOR_MATRIX = [
    dict(name='n1_untiled_flat_fhl_none_linear_gaussian',
         num_layers=1, hidden=[], top=8, input_shape=(24,), ntiles=None, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif=None),
    dict(name='n1_tiled_flat_fhl_c1_tanh_gaussian',
         num_layers=1, hidden=[], top=6, input_shape=(4, 12), ntiles=4, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif='c1', activ='tanh'),
    dict(name='n1_tiled_flat_efl_none_tanh_gaussian',
         num_layers=1, hidden=[], top=6, input_shape=(4, 12), ntiles=4, flat=True,
         arch='expand_first_lyr', conn='structured', classif=None, activ='tanh',
         expected_r1_shape=(4, 6)),
    dict(name='n2_untiled_nonflat_fhl_c2_linear_gaussian',
         num_layers=2, hidden=[10], top=9, input_shape=(6, 4), ntiles=None, flat=False,
         arch='flat_hidden_lyrs', conn='structured', classif='c2'),
    dict(name='n2_untiled_flat_efl_c1_tanh_gaussian',
         num_layers=2, hidden=[10], top=6, input_shape=(24,), ntiles=None, flat=True,
         arch='expand_first_lyr', conn='structured', classif='c1', activ='tanh',
         expected_r1_shape=(10,)),
    dict(name='n2_tiled_flat_efl_structured_c1_linear_kurtotic',
         num_layers=2, hidden=[6], top=5, input_shape=(4, 12), ntiles=4, flat=True,
         arch='expand_first_lyr', conn='structured', classif='c1', prior='sparse_kurtotic',
         expected_r1_shape=(4, 6)),
    dict(name='n2_tiled_flat_efl_reshaped_none_tanh_gaussian',
         num_layers=2, hidden=[6], top=5, input_shape=(4, 12), ntiles=4, flat=True,
         arch='expand_first_lyr', conn='reshaped', classif=None, activ='tanh',
         expected_r1_shape=(4, 6), expected_U2_shape=(24, 5)),
    dict(name='n3_tiled_flat_efl_reshaped_c2_linear_gaussian',
         num_layers=3, hidden=[6, 10], top=8, input_shape=(4, 12), ntiles=4, flat=True,
         arch='expand_first_lyr', conn='reshaped', classif='c2',
         expected_r1_shape=(4, 6), expected_U2_shape=(24, 10)),
    dict(name='n2_tiled_nonflat_efl_c1_linear_gaussian',
         num_layers=2, hidden=[6], top=5, input_shape=(2, 2, 3, 4), ntiles=4, flat=False,
         arch='expand_first_lyr', conn='structured', classif='c1',
         expected_r1_shape=(2, 2, 6)),
    dict(name='n2_tiled_nonflat_fhl_c1_linear_gaussian',
         num_layers=2, hidden=[6], top=5, input_shape=(2, 2, 3, 4), ntiles=4, flat=False,
         arch='flat_hidden_lyrs', conn='structured', classif='c1',
         expected_r1_shape=(6,)),
    dict(name='n3_untiled_flat_fhl_c1_tanh_kurtotic',
         num_layers=3, hidden=[12, 10], top=5, input_shape=(24,), ntiles=None, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif='c1', activ='tanh',
         prior='sparse_kurtotic'),
    dict(name='n4_untiled_flat_fhl_c1_linear_gaussian',
         num_layers=4, hidden=[12, 10, 8], top=5, input_shape=(24,), ntiles=None, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif='c1'),
    dict(name='n2_untiled_flat_fhl_c1_linear_gaussian_rnitersW',
         num_layers=2, hidden=[10], top=6, input_shape=(24,), ntiles=None, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif='c1',
         update_method={'r_niters_W': 2}),
    dict(name='n2_untiled_flat_fhl_c1_linear_gaussian_reqW',
         num_layers=2, hidden=[10], top=6, input_shape=(24,), ntiles=None, flat=True,
         arch='flat_hidden_lyrs', conn='structured', classif='c1',
         update_method={'r_eq_W': 60.0}, r_prior_dist=('gaussian', 0.0, 0.5)),
]

RECURRENT_CONNECTOR_MATRIX = [
    dict(name='rn1_c1_linear_stable', num_layers=1, hidden=[], top=6, classif='c1'),
    dict(name='rn2_c1_linear_stable', num_layers=2, hidden=[10], top=6, classif='c1'),
    dict(name='rn2_c2_tanh_normal', num_layers=2, hidden=[10], top=9, classif='c2',
         activ='tanh', softmax_type='normal', num_classes=5),
    dict(name='rn3_c1_tanh_stable', num_layers=3, hidden=[10, 8], top=6, classif='c1',
         activ='tanh'),
    dict(name='rn4_none_linear_stable', num_layers=4, hidden=[10, 8, 7], top=6, classif=None),
]


def assert_connector_matrix() -> None:
    """Run every valid connector combo twice and assert seeded determinism of each fingerprint."""
    for spec in STATIC_CONNECTOR_MATRIX:
        try:
            first = run_static_connector(spec)
            second = run_static_connector(spec)
        except Exception as error:
            raise AssertionError(f"static connector '{spec['name']}' failed: {error}") from error
        if not np.array_equal(first, second):
            raise AssertionError(f"static connector '{spec['name']}' is not deterministic under a fixed seed")

    for spec in RECURRENT_CONNECTOR_MATRIX:
        try:
            first = run_recurrent_connector(spec)
            second = run_recurrent_connector(spec)
        except Exception as error:
            raise AssertionError(f"recurrent connector '{spec['name']}' failed: {error}") from error
        if not np.array_equal(first, second):
            raise AssertionError(f"recurrent connector '{spec['name']}' is not deterministic under a fixed seed")


def assert_invalid_combinations_rejected() -> None:
    """Assert known-invalid feature combinations raise ValueError with the expected messages."""
    # Static n=1 + tiled + expand_first_lyr forbids classification (non-vectorial top).
    invalid = make_static_params(
        num_layers=1, hidden=[], top=6, input_shape=(4, 12), ntiles=4, flat=True,
        arch='expand_first_lyr', conn='structured', classif='c1')
    try:
        create_model(invalid)
    except ValueError as error:
        if 'non-vectorial' not in str(error):
            raise AssertionError('non-vectorial top-layer combination raised the wrong error') from error
    else:
        raise AssertionError('n=1 + tiled + expand_first_lyr + c1 did not raise ValueError')

    # reshaped requires tiled flat input.
    invalid = make_static_params(
        num_layers=2, hidden=[6], top=5, input_shape=(24,), ntiles=None, flat=True,
        arch='expand_first_lyr', conn='reshaped', classif=None)
    try:
        create_model(invalid)
    except ValueError as error:
        if 'reshaped' not in str(error):
            raise AssertionError('untiled reshaped combination raised the wrong error') from error
    else:
        raise AssertionError('untiled reshaped connection did not raise ValueError')

    # rPCC requires prior costs disabled.
    invalid = make_recurrent_params(num_layers=2, hidden=[10], top=6, classif='c1')
    invalid['r_prior_cost'] = 'gaussian'
    try:
        create_model(invalid)
    except ValueError as error:
        if 'no prior cost drivers' not in str(error):
            raise AssertionError('recurrent prior-cost combination raised the wrong error') from error
    else:
        raise AssertionError('recurrent r_prior_cost did not raise ValueError')


# ---------------------------------------------------------------------------
# Part 2: end-to-end training/eval plumbing (representative configs only)
# ---------------------------------------------------------------------------

def prepare_pipeline_model(params: dict, temp_dir: str, mod_name: str, seed: int) -> Any:
    """Create a seeded model whose output directories are redirected into temp_dir.

    Returns:
        The constructed StaticPCC or RecurrentPCC model.
    """
    np.random.seed(seed)
    model = create_model(params)
    model.mod_name = mod_name
    model.models_dir = str(Path(temp_dir) / 'models')
    model.checkpoints_dir = str(Path(temp_dir) / 'models' / 'checkpoints')
    model.diagnostics_dir = str(Path(temp_dir) / 'diagnostics')
    model.plots_dir = str(Path(temp_dir) / 'plots')
    return model


def assert_pipeline_artifacts(model: Any, epoch_n: int) -> tuple[Path, Path]:
    """Assert the epoch-1 checkpoint, final model, and diagnostics artifacts exist and are complete.

    Returns:
        Paths to the epoch-1 checkpoint and the final saved model.
    """
    checkpoint_name = model.generate_output_name(model.mod_name, 1)
    checkpoint_path = Path(model.checkpoints_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise AssertionError(f'Missing epoch-1 checkpoint: {checkpoint_path}')
    final_name = model.generate_output_name(model.mod_name, epoch_n)
    final_path = Path(model.models_dir) / final_name
    if not final_path.exists():
        raise AssertionError(f'Missing final model: {final_path}')
    diagnostics_path = (
        Path(model.diagnostics_dir) / str(epoch_n) / model.diagnostics_file_name(final_name)
    )
    if not diagnostics_path.exists():
        raise AssertionError(f'Missing final diagnostics: {diagnostics_path}')
    with open(diagnostics_path, 'rb') as diagnostics_file:
        diagnostics = pickle.load(diagnostics_file)
    for key in ('Jr', 'Jc', 'accuracy'):
        if key not in diagnostics:
            raise AssertionError(f'Diagnostics artifact missing {key}')
    return checkpoint_path, final_path


def assert_training_histories(model: Any, epoch_n: int) -> None:
    """Assert Jr (and Jc/accuracy when classifying) histories have the expected length and are finite."""
    if len(model.Jr) != epoch_n + 1:
        raise AssertionError('Jr history length does not match epoch_n + 1')
    assert_finite('Jr history', model.Jr)
    if model.classification_enabled():
        assert_finite('Jc history', model.Jc)
        assert_finite('accuracy history', model.accuracy)


def run_static_pipeline(temp_dir: str, run_tag: str, seed: int = 13) -> tuple:
    """Train a representative StaticPCC end to end in temp_dir and verify its artifacts.

    Returns:
        (model, X, Y, checkpoint_path, final_path).
    """
    epoch_n = 2
    params = make_static_params(
        num_layers=2, hidden=[6], top=5, input_shape=(4, 12), ntiles=4, flat=True,
        arch='expand_first_lyr', conn='reshaped', classif='c1', epoch_n=epoch_n)
    model = prepare_pipeline_model(params, temp_dir, f'mod.static.exp_000000_0000{run_tag}.{epoch_n}.pydb', seed)

    rng = np.random.RandomState(seed + 1)
    X = rng.uniform(0.05, 0.95, size=(model.num_inps,) + tuple(model.input_shape))
    Y = one_hot_labels(model.num_inps, model.num_classes)

    np.random.seed(seed + 2)
    with redirect_stdout(io.StringIO()):
        model.train(X, Y, save_checkpoint={'save_every': 1}, load_checkpoint=None, plot=False)
    assert_training_histories(model, epoch_n)
    checkpoint_path, final_path = assert_pipeline_artifacts(model, epoch_n)
    return model, X, Y, checkpoint_path, final_path


def run_recurrent_pipeline(temp_dir: str, run_tag: str, seed: int = 17) -> tuple:
    """Train a representative RecurrentPCC end to end in temp_dir and verify its artifacts.

    Returns:
        (model, X, Y, checkpoint_path, final_path).
    """
    epoch_n = 2
    params = make_recurrent_params(
        num_layers=2, hidden=[6], top=4, classif='c1', epoch_n=epoch_n)
    model = prepare_pipeline_model(params, temp_dir, f'mod.recurrent.exp_000000_0000{run_tag}.{epoch_n}.pydb', seed)

    rng = np.random.RandomState(seed + 1)
    X = rng.uniform(0.05, 0.95, size=(model.num_inps, model.input_shape[0], model.num_ts))
    Y = one_hot_labels(model.num_inps, model.num_classes)

    np.random.seed(seed + 2)
    with redirect_stdout(io.StringIO()):
        model.train(X, Y, save_checkpoint={'save_every': 1}, load_checkpoint=None, plot=False)
    assert_training_histories(model, epoch_n)
    checkpoint_path, final_path = assert_pipeline_artifacts(model, epoch_n)
    return model, X, Y, checkpoint_path, final_path


def assert_same_final_weights(model: Any, reloaded: Any) -> None:
    """Assert two models (e.g. in-memory versus reloaded) share identical final weights."""
    for layer in range(1, model.num_layers + 1):
        if not np.array_equal(model.U[layer], reloaded.U[layer]):
            raise AssertionError(f'Reloaded U[{layer}] differs from in-memory weights')
        if isinstance(model, RecurrentPCC):
            if not np.array_equal(model.Uhat[layer], reloaded.Uhat[layer]):
                raise AssertionError(f'Reloaded Uhat[{layer}] differs from in-memory weights')
            if not np.array_equal(model.Vhat[layer], reloaded.Vhat[layer]):
                raise AssertionError(f'Reloaded Vhat[{layer}] differs from in-memory weights')


def assert_static_training_pipeline() -> None:
    """Verify static train/checkpoint/reload/resume plumbing and cross-run determinism."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model, X, Y, checkpoint_path, final_path = run_static_pipeline(temp_dir, '01')

        # Load fidelity: pickled final model matches the in-memory model and evaluates identically.
        with open(final_path, 'rb') as model_file:
            reloaded = pickle.load(model_file)
        assert_same_final_weights(model, reloaded)
        update_name = next(iter(model.update_method))
        update_number = model.update_method[update_name]
        accuracy_live = model.evaluate(X, Y, update_name, update_number)
        accuracy_reloaded = reloaded.evaluate(X, Y, update_name, update_number)
        if accuracy_live != accuracy_reloaded:
            raise AssertionError('Reloaded static model does not evaluate identically')

        # Resume plumbing: continue training from the epoch-1 checkpoint.
        with open(checkpoint_path, 'rb') as checkpoint_file:
            resumed = pickle.load(checkpoint_file)
        resumed.load_epoch = 1
        with redirect_stdout(io.StringIO()):
            resumed.train(X, Y, save_checkpoint={'save_every': 1}, load_checkpoint=1, plot=False)
        assert_training_histories(resumed, resumed.epoch_n)
        resumed_final = Path(resumed.models_dir) / resumed.generate_output_name(resumed.mod_name, resumed.epoch_n)
        if not resumed_final.exists():
            raise AssertionError('Resumed static training did not save a final model')

    # Determinism: an identical pipeline run reproduces histories and weights exactly.
    with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
        model_a, _, _, _, _ = run_static_pipeline(temp_dir_a, '02')
        model_b, _, _, _, _ = run_static_pipeline(temp_dir_b, '03')
        if model_a.Jr != model_b.Jr or model_a.accuracy != model_b.accuracy:
            raise AssertionError('Static training histories are not deterministic under fixed seeds')
        assert_same_final_weights(model_a, model_b)


def assert_recurrent_training_pipeline() -> None:
    """Verify recurrent train/checkpoint/reload/resume plumbing and cross-run determinism."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model, X, Y, checkpoint_path, final_path = run_recurrent_pipeline(temp_dir, '01')

        with open(final_path, 'rb') as model_file:
            reloaded = pickle.load(model_file)
        assert_same_final_weights(model, reloaded)
        update_name = next(iter(model.update_method))
        update_number = model.update_method[update_name]
        details_live = model.evaluate(X, Y, update_name, update_number, return_details=True)
        details_reloaded = reloaded.evaluate(X, Y, update_name, update_number, return_details=True)
        if details_live['accuracy'] != details_reloaded['accuracy']:
            raise AssertionError('Reloaded recurrent model does not evaluate identically')
        if details_live['predictions'] != details_reloaded['predictions']:
            raise AssertionError('Reloaded recurrent model predictions differ')

        with open(checkpoint_path, 'rb') as checkpoint_file:
            resumed = pickle.load(checkpoint_file)
        resumed.load_epoch = 1
        with redirect_stdout(io.StringIO()):
            resumed.train(X, Y, save_checkpoint={'save_every': 1}, load_checkpoint=1, plot=False)
        assert_training_histories(resumed, resumed.epoch_n)
        resumed_final = Path(resumed.models_dir) / resumed.generate_output_name(resumed.mod_name, resumed.epoch_n)
        if not resumed_final.exists():
            raise AssertionError('Resumed recurrent training did not save a final model')

    with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
        model_a, _, _, _, _ = run_recurrent_pipeline(temp_dir_a, '02')
        model_b, _, _, _, _ = run_recurrent_pipeline(temp_dir_b, '03')
        if model_a.Jr != model_b.Jr or model_a.accuracy != model_b.accuracy:
            raise AssertionError('Recurrent training histories are not deterministic under fixed seeds')
        assert_same_final_weights(model_a, model_b)


def main() -> int:
    """Run all construction, connector, rejection, config-validation, and pipeline checks.

    Returns:
        0 after printing "status: PASS"; assertions raise on any failure.
    """
    assert_construction(
        'config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt',
        RecurrentPCC,
    )
    assert_construction(
        'config_sPCC_Rogers_2026_trace212_frozen_baseline.txt',
        StaticPCC,
    )
    assert_static_one_layer_constructs_from_frozen_baseline(
        'config_sPCC_Rogers_2026_trace212_frozen_baseline.txt'
    )
    assert_static_one_layer_constructs_from_frozen_baseline(
        'config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt'
    )
    assert_static_c2_allows_nonclass_top_size()
    assert_two_layer_reshaped_static_u2_flattens_expanded_r1()
    assert_recurrent_one_layer_constructs_from_frozen_baseline()
    assert_no_classification_metrics_absent()

    assert_connector_matrix()
    assert_invalid_combinations_rejected()
    assert_static_training_pipeline()
    assert_recurrent_training_pipeline()

    # Cross-type key hygiene: each model type constructs without the other's keys.
    recurrent_params = load_test_params(
        'config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt'
    )
    for static_key in ('ssq', 'alph', 'lam', 'ntiles_per_input', 'flat_input'):
        recurrent_params.pop(static_key)
    create_model(recurrent_params)

    recurrent_expand_first = dict(recurrent_params)
    recurrent_expand_first['architecture'] = 'expand_first_lyr'
    try:
        create_model(recurrent_expand_first)
    except ValueError as error:
        if "rPCC requires architecture='flat_hidden_lyrs'" not in str(error):
            raise AssertionError('Invalid recurrent architecture was not identified') from error
    else:
        raise AssertionError('Invalid recurrent architecture did not raise ValueError')

    static_params = load_test_params(
        'config_sPCC_Rogers_2026_trace212_frozen_baseline.txt'
    )
    for recurrent_key in ('num_ts', 'kV', 'ssqr', 'ssqV'):
        static_params.pop(recurrent_key)
    create_model(static_params)

    # Config-key validation: load_params must reject malformed rate/variance/plotting keys.
    recurrent_text = load_config_text('config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt')
    assert_load_params_fails(
        set_config_literal(recurrent_text, 'kr', "{1:0.1, 2:5.0, 'o':0.0}"),
        ["kr must not contain an 'o' key"],
    )

    static_text = load_config_text('config_sPCC_Rogers_2026_trace212_frozen_baseline.txt')
    assert_load_params_fails(
        set_config_literal(static_text, 'classif_method', "'c2'"),
        [
            "kU must include an 'o' key when classif_method='c2'",
            "lam must include an 'o' key when classif_method='c2'",
        ],
    )

    rpcc_c2_text = set_config_literal(recurrent_text, 'classif_method', "'c2'")
    assert_load_params_fails(
        rpcc_c2_text,
        ["kU must include an 'o' key when classif_method='c2'"],
    )

    static_c2_text = set_config_literal(static_text, 'classif_method', "'c2'")
    static_c2_text = set_config_literal(static_c2_text, 'kU', "{1:0.001, 2:0.001, 3:0.001, 'o':0.001}")
    assert_load_params_fails(
        static_c2_text,
        ["lam must include an 'o' key when classif_method='c2'"],
    )

    ssq_text = set_config_literal(static_text, 'ssq', "{0:1, 1:10, 2:10, 3:2, 'o':0.5}")
    assert_load_params_fails(
        ssq_text,
        ["ssq must not contain an 'o' key"],
    )

    ssqr_text = set_config_literal(recurrent_text, 'ssqr', "{0:10, 1:10, 2:5, 'o':1}")
    assert_load_params_fails(
        ssqr_text,
        ["ssqr must not contain an 'o' key"],
    )

    ssqv_text = set_config_literal(recurrent_text, 'ssqV', "{1:1, 2:1, 'o':1}")
    assert_load_params_fails(
        ssqv_text,
        ["ssqV must not contain an 'o' key"],
    )

    recurrent_rf_text = set_config_literal(recurrent_text, 'plot_receptive_fields', 'True')
    assert_load_params_fails(
        recurrent_rf_text,
        ["plot_receptive_fields is only supported for static models"],
    )

    missing_rf_flag_text = delete_config_key(static_text, 'plot_receptive_fields')
    assert_load_params_fails(
        missing_rf_flag_text,
        ["plot_receptive_fields must be set to a bool or a dict"],
    )

    rf_without_plot_train_text = set_config_literal(static_text, 'plot_receptive_fields', 'True')
    rf_without_plot_train_text = set_config_literal(rf_without_plot_train_text, 'plot_train', 'False')
    assert_load_params_fails(
        rf_without_plot_train_text,
        ["plot_receptive_fields requires plot_train=True"],
    )

    rf_mode_none_text = set_config_literal(static_text, 'plot_receptive_fields', "{'mode': None}")
    assert_load_params_and_model_succeed(rf_mode_none_text)

    rf_mode_all_text = set_config_literal(static_text, 'plot_receptive_fields', "{'mode': 'all'}")
    assert_load_params_and_model_succeed(rf_mode_all_text)

    rf_mode_random_sample_text = set_config_literal(
        static_text,
        'plot_receptive_fields',
        "{'mode': {'random_sample': 5}}",
    )
    assert_load_params_and_model_succeed(rf_mode_random_sample_text)

    rf_mode_random_sample_bool_text = set_config_literal(
        static_text,
        'plot_receptive_fields',
        "{'mode': {'random_sample': True}}",
    )
    assert_load_params_fails(
        rf_mode_random_sample_bool_text,
        ["plot_receptive_fields random_sample must be a positive integer"],
    )

    missing_activation_flag_text = delete_config_key(static_text, 'plot_softmaxed_activations')
    assert_load_params_fails(
        missing_activation_flag_text,
        ["plot_softmaxed_activations must be explicitly set to True or False"],
    )

    no_plot_train_activation_text = set_config_literal(recurrent_text, 'plot_softmaxed_activations', 'True')
    no_plot_train_activation_text = set_config_literal(no_plot_train_activation_text, 'plot_train', 'False')
    assert_load_params_fails(
        no_plot_train_activation_text,
        ["plot_softmaxed_activations requires plot_train=True"],
    )

    static_activation_text = set_config_literal(static_text, 'plot_softmaxed_activations', 'True')
    assert_load_params_fails(
        static_activation_text,
        ["plot_softmaxed_activations is not supported for static models"],
    )

    rate_schedule_text = set_config_literal(
        recurrent_text,
        'rate_schedule',
        "{'type': 'piecewise_linear', 'param': 'kU', 'key': 'o', 'points': [(0, 0.1), (1, 0.2)]}",
    )
    assert_load_params_fails(
        rate_schedule_text,
        ["rate_schedule key 'o' is only allowed when classif_method='c2'"],
    )

    c2_rate_schedule_text = set_config_literal(recurrent_text, 'classif_method', "'c2'")
    c2_rate_schedule_text = set_config_literal(c2_rate_schedule_text, 'kU', "{1:0.1, 2:0.1, 'o':0.0}")
    c2_rate_schedule_text = set_config_literal(
        c2_rate_schedule_text,
        'rate_schedule',
        "{'type': 'two_phase_linear_decay', 'param': 'kU', 'key': 'o', 'start_epoch': 0, 'end_epoch': 1, 'final_value': 0.05}",
    )
    static_c2_success_text = set_config_literal(static_text, 'classif_method', "'c2'")
    static_c2_success_text = set_config_literal(static_c2_success_text, 'kU', "{1:0.001, 2:0.001, 3:0.001, 'o':0.001}")
    static_c2_success_text = set_config_literal(static_c2_success_text, 'lam', "{1:0.001, 2:0.001, 3:0.001, 'o':0.001}")
    static_c2_success_text = set_config_literal(static_c2_success_text, 'top_lyr_size', '211')
    static_c2_success_text = set_config_literal(static_c2_success_text, 'num_classes', '211')

    assert_load_params_and_model_succeed(c2_rate_schedule_text)
    assert_load_params_and_model_succeed(static_c2_success_text)

    c1_size_mismatch_text = set_config_literal(static_text, 'top_lyr_size', '211')
    assert_load_params_fails(
        c1_size_mismatch_text,
        ["num_classes must equal top_lyr_size when classif_method='c1'"],
    )

    nc_alias_text = set_config_literal(recurrent_text, 'classif_method', "'nc'")
    assert_load_params_fails(
        nc_alias_text,
        ["classif_method must be 'c1', 'c2', or None"],
    )

    static_untiled_text = set_config_literal(static_text, 'ntiles_per_input', 'None')
    static_untiled_text = set_config_literal(static_untiled_text, 'first_to_second_lyr_connection_architecture', "'structured'")
    assert_load_params_and_model_succeed(static_untiled_text)

    static_untiled_nonflat_text = set_config_literal(static_untiled_text, 'flat_input', 'False')
    assert_load_params_and_model_succeed(static_untiled_nonflat_text)

    static_missing_flat_text = delete_config_key(static_text, 'flat_input')
    assert_load_params_succeeds_model_fails(
        static_missing_flat_text,
        ['Config missing required keys: flat_input'],
    )

    static_nonbool_flat_text = set_config_literal(static_text, 'flat_input', 'None')
    assert_load_params_succeeds_model_fails(
        static_nonbool_flat_text,
        ['flat_input must be explicitly set to True or False for static models.'],
    )

    # Factory-level rejection: missing controls, unknown model types, abstract base class.
    missing_recurrent_control = dict(recurrent_params)
    missing_recurrent_control.pop('ssqr')
    try:
        create_model(missing_recurrent_control)
    except ValueError as error:
        if 'ssqr' not in str(error):
            raise AssertionError('Missing recurrent control was not identified') from error
    else:
        raise AssertionError('Missing recurrent control did not raise ValueError')

    missing_static_control = dict(static_params)
    missing_static_control.pop('ssq')
    try:
        create_model(missing_static_control)
    except ValueError as error:
        if 'ssq' not in str(error):
            raise AssertionError('Missing static control was not identified') from error
    else:
        raise AssertionError('Missing static control did not raise ValueError')

    try:
        create_model({'model_type': 'unsupported'})
    except ValueError as error:
        if 'static' not in str(error) or 'recurrent' not in str(error):
            raise AssertionError('Invalid-model error must list supported model types') from error
    else:
        raise AssertionError('Unsupported model_type did not raise ValueError')

    try:
        PredictiveCodingClassifier({})
    except TypeError:
        pass
    else:
        raise AssertionError('PredictiveCodingClassifier must not be directly instantiable')

    print('status: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())