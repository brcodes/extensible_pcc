"""Shared implementation for the bounded extensible-versus-legacy parity runners.

Provide frozen-task resolution, seeded legacy and extensible training runs,
parity-metric computation against tolerances, and provenance/result logging
shared by test_smoke_1input.py and test_equivalence_3epoch.py. Not invoked
directly; run a wrapper from the pypredcoding root, e.g.
``python tests/test_smoke_1input.py --model rPCC --task cvcv12``. Results are
appended under test_results/ and run logs are written under log/.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


PYPREDCODING_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PYPREDCODING_DIR.parent
CONFIG_DIR = PYPREDCODING_DIR / "config"
TEST_RESULTS_DIR = PYPREDCODING_DIR / "test_results"

for import_path in (ROOT_DIR, PYPREDCODING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from pypredcoding.train_model import inspect_bounded_inference, load_configured_data, load_params, run_bounded_experiment
from validated_legacy_models.recurrentPCC import rPCC_Trainer
from validated_legacy_models.staticPCC import sPCC_Trainer, sPCC_Trainer5Nat


PARITY_FAILURE_MESSAGE = (
    "Either current pypredcoding model code/files were changed or former "
    "validated_legacy_models model code/files were changed such that pypredcoding "
    "(extensible) models do not produce legacy behavior, or "
    "validated_legacy_models (inextensible) models do not produce legacy behavior. "
    "Return to previous working commit to preserve parity between validated legacy "
    "behavior (cvcv12, trace212 or raonaturalimages5 classification behavior "
    "reflecting published information in Rogers 2026, Neural Computation) and "
    "replicated/identical (or when legacy seed was unspecified, analogous) extensible model behavior."
)


TASKS = {
    ("rPCC", "cvcv12"): {
        "config": CONFIG_DIR / "config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt",
        "kind": "recurrent",
        "parameterization": "Rogers_2026 CVCV12 recurrent classifier",
        "inext_files": (
            ROOT_DIR / "validated_legacy_models/recurrentPCC/rPCC_Trainer.py",
            ROOT_DIR / "validated_legacy_models/recurrentPCC/rPCC_Model.py",
            ROOT_DIR / "validated_legacy_models/recurrentPCC/data/CVCV_12_prepro.pkl",
        ),
    },
    ("sPCC", "trace212"): {
        "config": CONFIG_DIR / "config_sPCC_Rogers_2026_trace212_frozen_baseline.txt",
        "kind": "static_trace212",
        "parameterization": "Rogers_2026 trace212 static classifier",
        "inext_files": (
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_Trainer.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_Model.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/Dataset.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_parameters.pkl",
        ),
    },
    ("sPCC", "raonaturalimages5"): {
        "config": CONFIG_DIR / "config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt",
        "kind": "static_5nat",
        "parameterization": "Rogers_2026 5 natural images static classifier",
        "inext_files": (
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_Trainer5Nat.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_Model.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/Dataset.py",
            ROOT_DIR / "validated_legacy_models/staticPCC/sPCC_parameters_5nat.pkl",
        ),
    },
}

EXT_FILES = (
    PYPREDCODING_DIR / "train_model.py",
    PYPREDCODING_DIR / "model.py",
    PYPREDCODING_DIR / "cost_functions.py",
)


def resolve_task(model_name: str, task_name: str) -> dict:
    """Return the frozen TASKS entry for a model/task pair.

    Raises:
        ValueError: If the combination is not a supported parity task.
        FileNotFoundError: If the frozen baseline config file is missing.
    """
    task = TASKS.get((model_name, task_name))
    if task is None:
        valid = ", ".join(f"{model}/{name}" for model, name in TASKS)
        raise ValueError(f"Unsupported --model/--task combination: {model_name}/{task_name}. Valid: {valid}")
    if not task["config"].is_file():
        raise FileNotFoundError(f"Frozen parity config is missing: {task['config']}")
    return task


def _softmax_c1(state: np.ndarray) -> np.ndarray:
    """Softmax a state vector in float64 with max-subtraction, matching the legacy rPCC c1 path."""
    state = np.asarray(state, dtype=np.float64)
    exponentials = np.exp(state - np.max(state))
    return exponentials / np.sum(exponentials)


def _softmax_static(state: np.ndarray) -> np.ndarray:
    """Softmax a state vector in float128 without max-subtraction, matching the legacy sPCC path."""
    exponentials = np.exp(np.asarray(state, dtype=np.float128))
    return (exponentials / np.sum(exponentials)).astype(np.float64)


def _rel_fro(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute the Frobenius-norm difference relative to the reference norm."""
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        denominator = 1.0
    return float(np.linalg.norm(reference - current) / denominator)


def _weight_metrics(
    names: Sequence[str],
    legacy_weights: Sequence[np.ndarray],
    ext_weights: Sequence[np.ndarray],
) -> dict:
    """Compute shape and elementwise-difference metrics for paired legacy/extensible weights."""
    metrics = {}
    for name, legacy_weight, ext_weight in zip(names, legacy_weights, ext_weights):
        legacy_weight = np.asarray(legacy_weight, dtype=np.float64)
        ext_weight = np.asarray(ext_weight, dtype=np.float64)
        metrics[f"{name}_baseline_shape"] = legacy_weight.shape
        metrics[f"{name}_ext_shape"] = ext_weight.shape
        metrics[f"{name}_shape_match"] = legacy_weight.shape == ext_weight.shape
        metrics[f"{name}_same_size"] = legacy_weight.size == ext_weight.size
        if legacy_weight.size != ext_weight.size:
            metrics[f"{name}_rel_fro_diff"] = None
            metrics[f"{name}_max_abs_diff"] = None
            continue
        aligned_ext = ext_weight.reshape(legacy_weight.shape)
        metrics[f"{name}_rel_fro_diff"] = _rel_fro(legacy_weight, aligned_ext)
        metrics[f"{name}_max_abs_diff"] = float(np.max(np.abs(legacy_weight - aligned_ext)))
    return metrics


def _print_provenance(
    model_name: str,
    task_name: str,
    task: dict,
    test_name: str,
    epochs: int,
    seed_init: int,
    seed_shuffle: int,
) -> None:
    """Print run-provenance lines to stdout."""
    for line in _provenance_lines(model_name, task_name, task, test_name, epochs, seed_init, seed_shuffle):
        print(line)


def _dataset_inext_path(task: dict) -> Path:
    """Resolve the legacy (inextensible) dataset path for a task."""
    if task["kind"] == "recurrent":
        return task["inext_files"][-1]
    if task["kind"] == "static_trace212":
        params = sPCC_Trainer.load_parity_params(task["inext_files"][-1])
        return Path(__file__).resolve().parents[2] / "validated_legacy_models/staticPCC" / params.IN_DIR
    params = sPCC_Trainer5Nat.load_5nat_params(str(task["inext_files"][-1]))
    return Path(sPCC_Trainer5Nat.resolve_data_path(str(params["IN_DIR"])))


def _provenance_lines(
    model_name: str,
    task_name: str,
    task: dict,
    test_name: str,
    epochs: int,
    seed_init: int,
    seed_shuffle: int,
) -> list[str]:
    """Build provenance lines naming the config, datasets, and code files a run touches."""
    config_path = task["config"].resolve()
    ext_data_path = PYPREDCODING_DIR / "data" / load_params(str(config_path))["dataset_train"]
    lines = [
        f"test: {test_name}",
        f"model: {model_name}",
        f"task: {task_name}",
        f"selected_parameterization: {task['parameterization']}",
        f"epochs: {epochs}",
        f"seed_init: {seed_init}",
        f"seed_shuffle: {seed_shuffle}",
        f"resolved_config: {config_path}",
        f"dataset_inext: {_dataset_inext_path(task).resolve()}",
        f"dataset_ext: {ext_data_path.resolve()}",
        "accessed_ext_files:",
    ]
    lines.extend(f"  {path.resolve()}" for path in EXT_FILES)
    lines.append("accessed_inext_files:")
    lines.extend(f"  {path.resolve()}" for path in task["inext_files"])
    return lines


def _append_result(lines: list[str], result_prefix: str) -> None:
    """Append a timestamped result block to test_results/<prefix>_test_result.txt."""
    TEST_RESULTS_DIR.mkdir(exist_ok=True)
    result_path = TEST_RESULTS_DIR / f"{result_prefix}_test_result.txt"
    with result_path.open("a", encoding="utf-8") as result_file:
        result_file.write("\n" + "=" * 72 + "\n")
        result_file.write(f"timestamp: {datetime.now().isoformat()}\n")
        result_file.write("\n".join(lines) + "\n")


def _test_run_name(one_input: bool, task_name: str) -> str:
    """Build a timestamped run name distinguishing smoke from full-equivalence runs."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    suffix = "smoke_1inp" if one_input else "equivalence_3epoch"
    return f"test_{timestamp}_{suffix}_{task_name}"


def _run_recurrent(
    task: dict,
    task_name: str,
    epochs: int,
    seed_init: int,
    seed_shuffle: int,
    one_input: bool,
    run_name: str,
) -> tuple[dict, tuple[str, ...]]:
    """Run the rPCC cvcv12 legacy/extensible pair and compute parity metrics.

    Returns:
        Metrics dict and the names of the weight matrices compared.
    """
    input_dict = rPCC_Trainer.load_parity_inputs()
    if one_input:
        X, Y = load_configured_data(task["config"])
        X, Y = X[:1, :, :1], Y[:1]
        legacy = rPCC_Trainer.build_parity_model(input_dict, seed_init)
        rPCC_Trainer.train_parity_model(legacy, input_dict, 1, seed_shuffle, num_ts=1, word_indices=[0])
        legacy_states, legacy_probs, legacy_accuracy = rPCC_Trainer.inspect_parity_model(
            legacy, input_dict, word_indices=[0], num_ts=1
        )
        ext, _, _, _, _ = run_bounded_experiment(
            task["config"],
            seed=seed_init,
            epochs=1,
            X=X,
            Y=Y,
            num_ts=1,
            run_name=run_name,
            train_seed=seed_shuffle,
        )
    else:
        X, Y = load_configured_data(task["config"])
        ext, _, _, _, _ = run_bounded_experiment(
            task["config"],
            seed=seed_init,
            epochs=epochs,
            X=X,
            Y=Y,
            run_name=run_name,
            train_seed=seed_shuffle,
        )
        legacy = rPCC_Trainer.build_parity_model(input_dict, seed_init)
        rPCC_Trainer.train_parity_model(legacy, input_dict, epochs, seed_shuffle)
        legacy_states, legacy_probs, legacy_accuracy = rPCC_Trainer.inspect_parity_model(legacy, input_dict)

    ext_states, ext_accuracy = inspect_bounded_inference(ext, X, Y)
    ext_probs = np.stack([_softmax_c1(state) for state in ext_states])
    metrics = {
        "final_acc_baseline": legacy_accuracy,
        "final_acc_ext": ext_accuracy,
        "final_acc_abs_diff": abs(legacy_accuracy - ext_accuracy),
        "probs_max_abs_diff": float(np.max(np.abs(legacy_probs - ext_probs))),
        "r2_max_abs_diff": float(np.max(np.abs(legacy_states - ext_states))),
    }
    metrics.update(_weight_metrics(
        ("U1", "U2", "V1", "V2"),
        (legacy.U1, legacy.U2, legacy.V1, legacy.V2),
        (ext.Uhat[1][:, :, -1], ext.Uhat[2][:, :, -1], ext.Vhat[1][:, :, -1], ext.Vhat[2][:, :, -1]),
    ))
    return metrics, ("U1", "U2", "V1", "V2")


def _run_static(
    task: dict,
    task_name: str,
    epochs: int,
    seed_init: int,
    seed_shuffle: int,
    one_input: bool,
    run_name: str,
) -> tuple[dict, tuple[str, ...]]:
    """Run an sPCC (trace212 or raonaturalimages5) legacy/extensible pair and compute parity metrics.

    Returns:
        Metrics dict and the names of the weight matrices compared.

    Raises:
        ValueError: If the configured extensible dataset diverges from legacy preprocessing.
    """
    if task["kind"] == "static_trace212":
        trainer = sPCC_Trainer
        params = trainer.load_parity_params()
        dataset = trainer.build_parity_dataset(params, shuffle=False)
    else:
        trainer = sPCC_Trainer5Nat
        params = trainer.load_5nat_params(str(task["inext_files"][-1]))
        dataset = trainer.build_parity_dataset(params, shuffle=False, gauss_mask_sigma=float(params["TEST_GAUSS_MASK_SIGMA"]))

    legacy_X = np.asarray([dataset.get_rf1_patches(index) for index in range(len(dataset.rf2_patches))], dtype=np.float32)
    legacy_Y = np.asarray(dataset.labels, dtype=np.float32)
    X, Y = load_configured_data(task["config"])
    if not np.array_equal(X, legacy_X) or not np.array_equal(Y, legacy_Y):
        raise ValueError(
            "Configured extensible dataset does not match static legacy preprocessing. "
            "Regenerate it with the matching pypredcoding/data.py CLI option."
        )
    input_indices = None
    if one_input:
        X, Y = X[:1], Y[:1]
        input_indices = [0]

    legacy = trainer.build_parity_model(dataset, params, seed_init)
    trainer.train_parity_model(
        legacy, params, epochs, seed_shuffle, input_indices=input_indices,
        shuffle=not one_input, dataset=dataset if one_input else None,
    )
    legacy_states, legacy_probs, legacy_accuracy = trainer.inspect_parity_model(
        legacy, dataset, input_indices=input_indices
    )
    ext, _, _, _, _ = run_bounded_experiment(
        task["config"],
        seed=seed_init,
        epochs=epochs,
        X=X,
        Y=Y,
        run_name=run_name,
        train_seed=seed_shuffle,
    )
    ext_states, ext_accuracy = inspect_bounded_inference(ext, X, Y)
    ext_probs = np.stack([_softmax_static(state) for state in ext_states])
    metrics = {
        "final_acc_baseline": legacy_accuracy,
        "final_acc_ext": ext_accuracy,
        "final_acc_abs_diff": abs(legacy_accuracy - ext_accuracy),
        "probs_max_abs_diff": float(np.max(np.abs(legacy_probs - ext_probs))),
        "r3_max_abs_diff": float(np.max(np.abs(legacy_states - ext_states))),
    }
    metrics.update(_weight_metrics(("U1", "U2", "U3"), (legacy.U1, legacy.U2, legacy.U3), (ext.U[1], ext.U[2], ext.U[3])))
    return metrics, ("U1", "U2", "U3")


def run_parity(
    model_name: str,
    task_name: str,
    epochs: int,
    seed_init: int,
    seed_shuffle: int,
    one_input: bool,
) -> int:
    """Run one parity task, check metrics against frozen tolerances, and log the outcome.

    Returns:
        Process exit code: 0 on PASS, 1 on FAIL.
    """
    task = resolve_task(model_name, task_name)
    test_name = _test_run_name(one_input, task_name)
    result_prefix = test_name
    _print_provenance(model_name, task_name, task, test_name, epochs, seed_init, seed_shuffle)

    if task["kind"] == "recurrent":
        metrics, weight_names = _run_recurrent(task, task_name, epochs, seed_init, seed_shuffle, one_input, test_name)
        state_name = "r2"
        probability_tolerance = 1e-9
        state_tolerance = 1e-8
    else:
        metrics, weight_names = _run_static(task, task_name, epochs, seed_init, seed_shuffle, one_input, test_name)
        state_name = "r3"
        probability_tolerance = 2e-9 if task_name == "raonaturalimages5" and not one_input else 1e-9
        state_tolerance = 1.5e-8 if task_name == "raonaturalimages5" and not one_input else 1e-8

    tolerances = {
        "acc_abs_diff_max": 1e-12,
        "weight_rel_fro_diff_max": 1e-5,
        "probs_max_abs_diff_max": probability_tolerance,
        f"{state_name}_max_abs_diff_max": state_tolerance,
    }
    passed = (
        metrics["final_acc_abs_diff"] <= tolerances["acc_abs_diff_max"]
        and metrics["probs_max_abs_diff"] <= tolerances["probs_max_abs_diff_max"]
        and metrics[f"{state_name}_max_abs_diff"] <= tolerances[f"{state_name}_max_abs_diff_max"]
        and all(metrics[f"{name}_rel_fro_diff"] is not None and metrics[f"{name}_rel_fro_diff"] <= tolerances["weight_rel_fro_diff_max"] for name in weight_names)
    )

    lines = _provenance_lines(model_name, task_name, task, test_name, epochs, seed_init, seed_shuffle)
    lines.extend((f"config: {task['config'].resolve()}", "tolerances:"))
    lines.extend(f"  {name}: {value}" for name, value in tolerances.items())
    lines.extend(f"{name}: {value}" for name, value in metrics.items())
    lines.append(f"status: {'PASS' if passed else 'FAIL'}")
    if not passed:
        lines.append(PARITY_FAILURE_MESSAGE)
    _append_result(lines, result_prefix)
    for line in lines:
        print(line)
    return 0 if passed else 1