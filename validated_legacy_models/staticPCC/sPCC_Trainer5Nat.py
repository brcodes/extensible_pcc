"""Legacy static PCC trainer for 5-image natural-image classification.

Train the static PCC model on the five rb99 natural images loaded from raw
PNGs, checkpoint weights per epoch, evaluate classification accuracy, and
save accuracy and receptive-field plots. This is part of the validated
legacy reference implementation of the Li/Rogers static PCC model; it also
exposes parity helpers used by the equivalence tests.
"""

from __future__ import annotations

from datetime import datetime
import glob
import os
import pickle
import re
import shutil
import sys
from typing import Any, Iterable, List, Optional, Tuple
from pathlib import Path

# Use a non-interactive backend for unattended/batch runs.
os.environ.setdefault("MPLBACKEND", "Agg")

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

script_dir = Path(__file__).resolve().parent

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from validated_legacy_models.staticPCC.Dataset import Dataset
from validated_legacy_models.staticPCC.sPCC_Model import Model

# When True, skip training and only generate plots from existing OUT_DIR artifacts.
plot_only = False


def resolve_data_path(path_value: str) -> str:
    """Resolve IN_DIR as absolute or relative to this script directory."""
    if os.path.isabs(path_value):
        return path_value
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, path_value))


def build_parity_dataset(params: dict, shuffle: bool, gauss_mask_sigma: float) -> Dataset:
    """Build the legacy 5Nat Dataset with the active trainer settings."""
    return Dataset(
        scale=float(params["INPUT_SCALE"]),
        shuffle=shuffle,
        data_dir=resolve_data_path(str(params["IN_DIR"])),
        rf1_x=int(params["RF1_SIZE"]["x"]),
        rf1_y=int(params["RF1_SIZE"]["y"]),
        rf1_offset_x=int(params["RF1_OFFSET"]["x"]),
        rf1_offset_y=int(params["RF1_OFFSET"]["y"]),
        rf1_layout_x=int(params["RF1_LAYOUT"]["x"]),
        rf1_layout_y=int(params["RF1_LAYOUT"]["y"]),
        gauss_mask_sigma=float(gauss_mask_sigma),
    )


def load_5nat_params(path: str) -> dict:
    """Load 5nat params from plain pickle dict, with pandas-pickle fallback."""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        return payload

    if hasattr(payload, "to_dict"):
        return payload.to_dict()

    raise TypeError(f"Unsupported parameter payload type: {type(payload)}")


def parse_optional_int(value: Any, field_name: str) -> int | None:
    """Return value as int, or None if unset. field_name is accepted but unused."""
    if value is None:
        return None
    return int(value)


def parse_optional_float(value: Any, field_name: str) -> float | None:
    """Return value as float, or None if unset. field_name is accepted but unused."""
    if value is None:
        return None
    return float(value)


def parse_optional_seed(value: Any) -> int | None:
    """Return value as an int seed, or None if unset."""
    if value is None:
        return None
    return int(value)


def env_flag(name: str) -> bool:
    """Interpret the named environment variable as a boolean flag."""
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def simulate_k_u_schedule(
    initial_k_u: float,
    decay_cycle: Optional[int],
    decay_rate: Optional[float],
    input_count: int,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Simulate the per-input k_U decay schedule for one epoch without training.

    Returns:
        The final k_U and a list of (input_index, previous_k_u, new_k_u) decay events.
    """
    current_k_u = float(initial_k_u)
    decay_events: List[Tuple[int, float, float]] = []

    if decay_cycle is None or decay_rate is None:
        return current_k_u, decay_events

    if decay_cycle <= 0:
        raise ValueError("K2_DECAY_CYCLE must be positive when decay is enabled.")
    if decay_rate == 0:
        raise ValueError("K2_DECAY_RATE must be non-zero when decay is enabled.")

    for input_index in range(int(input_count)):
        if input_index % decay_cycle == 0:
            previous_k_u = current_k_u
            current_k_u = current_k_u / decay_rate
            decay_events.append((input_index, previous_k_u, current_k_u))

    return current_k_u, decay_events


def print_learning_rate_dry_run(model: Model, train_set: Dataset) -> None:
    """Print the active learning rates and one-epoch k_U decay schedule."""
    input_count = len(train_set.rf2_patches)
    final_k_u, decay_events = simulate_k_u_schedule(
        initial_k_u=model.k_U_init,
        decay_cycle=model.k_U_decay_cycle,
        decay_rate=model.k_U_decay_rate,
        input_count=input_count,
    )

    print("Learning-rate dry run")
    print(f"  activation: {model.activation}")
    print(f"  apply_input: {model.apply_input.__name__}")
    print(f"  k_r (K1): {model.k_r:.12g}")
    print(f"  k_U_init (K2): {model.k_U_init:.12g}")
    print(f"  inputs per epoch: {input_count}")

    if model.k_U_decay_cycle is None or model.k_U_decay_rate is None:
        print("  k_U decay: disabled")
        print(f"  k_U at end of epoch: {final_k_u:.12g}")
        return

    print(
        "  k_U decay: enabled "
        f"(cycle={model.k_U_decay_cycle}, rate={model.k_U_decay_rate:.12g})"
    )
    if decay_events:
        print(
            "  first decay event is applied after training input "
            f"{decay_events[0][0]}"
        )
        print("  first decay events:")
        for input_index, previous_k_u, current_k_u in decay_events[:8]:
            print(
                "    input "
                f"{input_index}: {previous_k_u:.12g} -> {current_k_u:.12g}"
            )
        remaining_events = len(decay_events) - 8
        if remaining_events > 0:
            print(f"    ... {remaining_events} additional decay events omitted")
    else:
        print("  k_U decay: configured but no events occur within one epoch")

    print(f"  k_U at end of epoch: {final_k_u:.12g}")


def configure_parity_model(model: Model, params: dict) -> Model:
    """Apply the active 5Nat trainer configuration to a legacy model."""
    model.prior = str(params["PRIOR"]).strip().lower()
    model.k_r = float(params["K1"])
    model.k_U_init = float(params["K2"])
    model.k_U = model.k_U_init
    model.k_U_decay_cycle = parse_optional_int(params["K2_DECAY_CYCLE"], "K2_DECAY_CYCLE")
    model.k_U_decay_rate = parse_optional_float(params["K2_DECAY_RATE"], "K2_DECAY_RATE")
    model.sigma_sq0 = float(params["SS0"])
    model.sigma_sq1 = float(params["SS1"])
    model.sigma_sq2 = float(params["SS2"])
    model.sigma_sq3 = float(params["SS3"])
    model.alpha1 = float(params["ALPHA1"])
    model.alpha2 = float(params["ALPHA2"])
    model.alpha3 = float(params["ALPHA3"])
    model.lambda1 = float(params["LAMBDA1"])
    model.lambda2 = float(params["LAMBDA2"])
    model.lambda3 = float(params["LAMBDA3"])
    model.activation = str(params["ACTIVATION"]).strip().lower()
    if model.activation == "tanh":
        model.apply_input = model.apply_input_tanh
    return model


def build_parity_model(dataset: Dataset, params: dict, seed: int) -> Model:
    """Build the configured legacy 5Nat model with a deterministic seed."""
    np.random.seed(seed)
    return configure_parity_model(Model(iteration=int(params["ITER_N"]), dataset=dataset), params)


def train_parity_model(model: Model, params: dict, epochs: int, seed_shuffle: int,
                       input_indices: Iterable[int] | None = None, shuffle: bool = True,
                       dataset: Dataset | None = None) -> Model:
    """Run bounded 5Nat epochs through the legacy Model.train path."""
    np.random.seed(seed_shuffle)
    for _ in range(int(epochs)):
        train_set = dataset
        if train_set is None:
            train_set = build_parity_dataset(
                params, shuffle=shuffle, gauss_mask_sigma=float(params["TRAIN_GAUSS_MASK_SIGMA"])
            )
        model.train(train_set, input_indices=input_indices)
    return model


def inspect_parity_model(model: Model, dataset: Dataset,
                         input_indices: Iterable[int] | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return 5Nat top states, probabilities, and accuracy from legacy inference."""
    if input_indices is None:
        input_indices = range(len(dataset.rf2_patches))
    else:
        input_indices = tuple(int(index) for index in input_indices)

    states = []
    probabilities = []
    correct = 0
    # Softmax over the top-layer state in extended precision for numerical stability.
    for index in input_indices:
        _, _, state, _, _, _ = model.apply_input(
            dataset.get_rf1_patches(index), dataset.labels[index], training=False
        )
        probability = np.exp(state.astype(np.float128))
        probability = (probability / np.sum(probability)).astype(np.float64)
        states.append(np.asarray(state, dtype=np.float64))
        probabilities.append(probability)
        response = int(np.argmax(probability)) if np.sum(probability == probability.max()) == 1 else None
        correct += int(int(np.argmax(dataset.labels[index])) == response)
    return np.stack(states), np.stack(probabilities), correct / len(input_indices)


def configure_plot_style() -> None:
    """Apply a seaborn-like paper style across matplotlib versions."""
    for style_name in ("seaborn-paper", "seaborn-v0_8-paper", "seaborn"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an image array to uint8 range [0, 255] for plotting parity with notebook."""
    image = np.asarray(image, dtype=np.float32)
    vmin = float(np.min(image))
    vmax = float(np.max(image))
    if np.isclose(vmax, vmin):
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - vmin) / (vmax - vmin)
    return np.round(scaled * 255.0).astype(np.uint8)


def nearest_upsample(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """Upsample with nearest-neighbor behavior (equivalent to cv2.INTER_NEAREST)."""
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def save_receptive_field_panels(
    model: Model,
    test_set: Dataset,
    out_dir: str,
    epoch: int,
    zero_pad_len: int,
) -> None:
    """Save input/level1/level2 reconstruction panels for every test image at an epoch."""
    model.load(os.path.join(out_dir, f"epoch_{epoch:0>{zero_pad_len}d}"))

    receptive_fields_dir = os.path.join(out_dir, "receptive_fields")
    os.makedirs(receptive_fields_dir, exist_ok=True)

    for i in range(len(test_set.filtered_images)):
        filtered_img = nearest_upsample(test_set.filtered_images[i], scale=4)
        filtered_img = normalize_to_uint8(filtered_img)

        inputs = test_set.get_rf1_patches(i)
        label = test_set.labels[i]
        r1, r2, _, _, _, _ = model.apply_input(inputs, label, training=False)
        level1_img = nearest_upsample(model.reconstruct(r1, level=1), scale=4)
        level2_img = nearest_upsample(model.reconstruct(r2, level=2), scale=4)
        level1_img = normalize_to_uint8(level1_img)
        level2_img = normalize_to_uint8(level2_img)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=80)
        axes[0].set_title(f"input: image_{i}")
        axes[0].imshow(filtered_img, cmap="gray")
        axes[1].set_title("level 1")
        axes[1].imshow(level1_img, cmap="gray")
        axes[2].set_title("level 2")
        axes[2].imshow(level2_img, cmap="gray")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

        file_name = f"epoch_{epoch:0>{zero_pad_len}d}_image_{i}_input_level1_level2.png"
        fig.savefig(os.path.join(receptive_fields_dir, file_name), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    """Run the full 5Nat training, evaluation, and plotting pipeline."""
    print("Start:", datetime.now().strftime("%c"))
    param = load_5nat_params(script_dir / "sPCC_parameters_5nat.pkl")
    print(f"params are:\n{param}")
    prior = str(param.get("PRIOR", "kurtotic")).strip().lower()
    if prior not in {"kurtotic", "gaussian"}:
        raise ValueError(f"Unsupported PRIOR '{prior}'. Expected 'kurtotic' or 'gaussian'.")

    activation = str(param.get("ACTIVATION", "linear")).strip().lower()
    if activation not in {"linear", "tanh"}:
        raise ValueError(f"Unsupported ACTIVATION '{activation}'. Expected 'linear' or 'tanh'.")

    in_dir = resolve_data_path(param["IN_DIR"])
    out_dir = os.path.join(script_dir, param["OUT_DIR"])
    os.makedirs(out_dir, exist_ok=True)

    random_seed = parse_optional_seed(param.get("RANDOM_SEED"))
    if random_seed is not None:
        np.random.seed(random_seed)

    test_set = Dataset(
        scale=float(param["INPUT_SCALE"]),
        shuffle=False,
        data_dir=in_dir,
        rf1_x=param["RF1_SIZE"]["x"],
        rf1_y=param["RF1_SIZE"]["y"],
        rf1_offset_x=param["RF1_OFFSET"]["x"],
        rf1_offset_y=param["RF1_OFFSET"]["y"],
        rf1_layout_x=param["RF1_LAYOUT"]["x"],
        rf1_layout_y=param["RF1_LAYOUT"]["y"],
        gauss_mask_sigma=float(param["TEST_GAUSS_MASK_SIGMA"]),
    )

    epoch_n = int(param["EPOCH_N"])
    zero_pad_len = len(str(epoch_n))

    model = Model(iteration=int(param["ITER_N"]), dataset=test_set)
    model.prior = prior

    model.k_r = float(param["K1"])
    model.k_U_init = float(param["K2"])
    model.k_U = model.k_U_init
    model.k_U_decay_cycle = parse_optional_int(param["K2_DECAY_CYCLE"], "K2_DECAY_CYCLE")
    model.k_U_decay_rate = parse_optional_float(param["K2_DECAY_RATE"], "K2_DECAY_RATE")
    model.sigma_sq0 = float(param["SS0"])
    model.sigma_sq1 = float(param["SS1"])
    model.sigma_sq2 = float(param["SS2"])
    model.sigma_sq3 = float(param["SS3"])
    model.alpha1 = float(param["ALPHA1"])
    model.alpha2 = float(param["ALPHA2"])
    model.alpha3 = float(param["ALPHA3"])
    model.lambda1 = float(param["LAMBDA1"])
    model.lambda2 = float(param["LAMBDA2"])
    model.lambda3 = float(param["LAMBDA3"])
    model.activation = activation
    if activation == "tanh":
        model.apply_input = model.apply_input_tanh

    mpl.rcdefaults()
    configure_plot_style()
    plt.rcParams["image.aspect"] = "auto"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    sns.set_palette("colorblind")
    sns.set_context(context="paper", font_scale=1.2, rc=None)

    if not plot_only and env_flag("SPCC_DRY_RUN_LR_TRACE"):
        train_set = Dataset(
            scale=float(param["INPUT_SCALE"]),
            shuffle=True,
            data_dir=in_dir,
            rf1_x=param["RF1_SIZE"]["x"],
            rf1_y=param["RF1_SIZE"]["y"],
            rf1_offset_x=param["RF1_OFFSET"]["x"],
            rf1_offset_y=param["RF1_OFFSET"]["y"],
            rf1_layout_x=param["RF1_LAYOUT"]["x"],
            rf1_layout_y=param["RF1_LAYOUT"]["y"],
            gauss_mask_sigma=float(param["TRAIN_GAUSS_MASK_SIGMA"]),
        )
        print_learning_rate_dry_run(model, train_set)
        return

    random_seed_shuffle = parse_optional_seed(param.get("RANDOM_SEED_SHUFFLE"))
    if random_seed_shuffle is not None:
        np.random.seed(random_seed_shuffle)

    if param["CLEAR_SAVED_WEIGHTS"] and not plot_only and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # Resume from the latest epoch checkpoint, or create/load the pretraining snapshot.
    out_dir_all = glob.glob(os.path.join(out_dir, "*"))
    out_dir_epoch = glob.glob(os.path.join(out_dir, "epoch_*"))
    out_dir_pretrain = os.path.join(out_dir, "pretraining")
    results_path = os.path.join(out_dir, "results.pkl")

    if plot_only and len(out_dir_epoch) == 0 and not os.path.exists(results_path):
        raise RuntimeError(
            "plot_only=True requires existing artifacts in OUT_DIR. "
            "Expected epoch_* checkpoints and/or results.pkl."
        )

    if len(out_dir_epoch) > 0:
        regex = re.compile(os.path.join(out_dir, "epoch_(?P<epoch>.*)"))
        epoch_all = [int(regex.match(x).group("epoch")) for x in out_dir_epoch]
        epoch_max_idx = np.argmax(epoch_all)
        epoch_max = epoch_all[epoch_max_idx]
        model.load(out_dir_epoch[epoch_max_idx])
    elif out_dir_pretrain not in out_dir_all and not plot_only:
        epoch_max = -1
        model.save(os.path.join(out_dir, "pretraining"))
    else:
        epoch_max = -1
        if not plot_only:
            model.load(os.path.join(out_dir, "pretraining"))

    if not plot_only:
        # Epoch loop: rebuild a shuffled dataset each epoch; checkpoint at epoch 0 and every 10th.
        for i in [x for x in range(epoch_n) if x > epoch_max]:
            train_set = Dataset(
                scale=float(param["INPUT_SCALE"]),
                shuffle=True,
                data_dir=in_dir,
                rf1_x=param["RF1_SIZE"]["x"],
                rf1_y=param["RF1_SIZE"]["y"],
                rf1_offset_x=param["RF1_OFFSET"]["x"],
                rf1_offset_y=param["RF1_OFFSET"]["y"],
                rf1_layout_x=param["RF1_LAYOUT"]["x"],
                rf1_layout_y=param["RF1_LAYOUT"]["y"],
                gauss_mask_sigma=float(param["TRAIN_GAUSS_MASK_SIGMA"]),
            )
            model.train(train_set)

            if i == 0 or i % 10 == 9:
                model.save(os.path.join(out_dir, f"epoch_{i:0>{zero_pad_len}d}"))

    if plot_only and os.path.exists(results_path):
        results_df = pd.read_pickle(results_path)
    else:
        # Evaluate every saved 10th-epoch checkpoint on the unshuffled test set.
        results = {
            "epoch": [],
            "target_v": [],
            "response_v": [],
            "target_i": [],
            "response_i": [],
            "accuracy": [],
        }

        eval_epoch_candidates = [i for i in range(epoch_n) if i % 10 == 9]
        eval_epochs = [
            i
            for i in eval_epoch_candidates
            if os.path.isdir(os.path.join(out_dir, f"epoch_{i:0>{zero_pad_len}d}"))
        ]

        for i in eval_epochs:
            model.load(os.path.join(out_dir, f"epoch_{i:0>{zero_pad_len}d}"))

            for j in range(len(test_set.rf2_patches)):
                inputs = test_set.get_rf1_patches(j)
                label = test_set.labels[j]
                _, _, r3, _, _, _ = model.apply_input(inputs, label, training=False)

                target_i = int(np.argmax(label))
                if np.sum(r3 == r3.max()) != 1:
                    response_i = None
                else:
                    response_i = int(np.argmax(r3))

                if target_i == response_i:
                    accuracy = 1
                else:
                    accuracy = 0

                results["epoch"].append(i)
                results["target_v"].append(label)
                results["response_v"].append(r3)
                results["target_i"].append(target_i)
                results["response_i"].append(response_i)
                results["accuracy"].append(accuracy)

        results_df = pd.DataFrame.from_dict(results)
        if results_df.empty:
            raise RuntimeError(
                "No evaluation rows were collected. "
                "In plot_only mode, ensure OUT_DIR has saved epoch_* checkpoints/results."
            )
        results_df.to_pickle(results_path)

    plt.figure(figsize=(4, 3))
    acc_by_epoch = results_df.groupby("epoch")["accuracy"].mean()
    # With short runs (e.g., 10 epochs), only epoch 9 may be evaluated.
    # Use markers so single-point curves remain visible.
    ax = acc_by_epoch.plot(marker="o", linewidth=1.25, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, symbol=None))
    plt.savefig(os.path.join(out_dir, "sPCC_train_acc.png"), dpi=600, bbox_inches="tight")
    plt.show()

    eval_epochs = sorted(results_df["epoch"].unique().tolist())
    if eval_epochs:
        first_epoch = int(eval_epochs[0])
        last_epoch = int(eval_epochs[-1])
        save_receptive_field_panels(model, test_set, out_dir, first_epoch, zero_pad_len)
        if last_epoch != first_epoch:
            save_receptive_field_panels(model, test_set, out_dir, last_epoch, zero_pad_len)
        print(
            "Saved input/level1/level2 receptive-field panels for "
            f"epochs {first_epoch} and {last_epoch} in {os.path.join(out_dir, 'receptive_fields')}"
        )

    print("End:", datetime.now().strftime("%c"))


if __name__ == "__main__":
    main()
