"""Hyperparameter definition for the legacy static PCC 5Nat trainer.

Define the single hyperparameter dict for the five-natural-image (rb99)
classification task and pickle it to sPCC_parameters_5nat.pkl for
sPCC_Trainer5Nat.py to consume. This is part of the validated legacy
reference implementation of the Li/Rogers static PCC model.
"""

from __future__ import annotations

from datetime import datetime
import pickle
from pathlib import Path

script_dir = Path(__file__).resolve().parent
CONFIG_FILENAME = script_dir / "sPCC_parameters_5nat.pkl"

PARAMS = {
    "PRIOR": "kurtotic",
    "ACTIVATION": "linear",
    "ITER_N": 30,
    "EPOCH_N": 500,
    "K1": 0.005,
    "K2": 0.005,
    "K2_DECAY_CYCLE": None,
    "K2_DECAY_RATE": None,
    "SS0": 1.0,
    "SS1": 10.0,
    "SS2": 10.0,
    "SS3": 10.0,
    "ALPHA1": 1.0,
    "ALPHA2": 0.05,
    "ALPHA3": 0.05,
    "LAMBDA1": 0.02,
    "LAMBDA2": 0.00001,
    "LAMBDA3": 0.02,
    "RF1_SIZE": {"x": 16, "y": 16},
    "RF1_OFFSET": {"x": 8, "y": 8},
    "RF1_LAYOUT": {"x": 15, "y": 15},
    "RANDOM_SEED": 1.0,
    "RANDOM_SEED_SHUFFLE": 1.0,
    "INPUT_SCALE": 1.0,
    "TEST_GAUSS_MASK_SIGMA": 1.0,
    "TRAIN_GAUSS_MASK_SIGMA": 1.0,
    "CLEAR_SAVED_WEIGHTS": True,
    "IN_DIR": "./data/raw/images_rao_128x128",
    "OUT_DIR": './models_plus_results/sPCC_raonaturalimages5_train',
}

def main() -> None:
    """Write the 5nat training parameters to CONFIG_FILENAME."""
    with open(CONFIG_FILENAME, "wb") as f:
        pickle.dump(PARAMS, f, protocol=4)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    print(f"Saved config at {timestamp}: {CONFIG_FILENAME}")


if __name__ == "__main__":
    main()
