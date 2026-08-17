"""Hyperparameter grid definition for the legacy static PCC trace212 trainer.

Build a scikit-learn ParameterGrid over static predictive-coding classifier
hyperparameters and pickle each grid entry to sPCC_parameters.pkl for
sPCC_Trainer.py to consume. This is part of the validated legacy reference
implementation of the Li/Rogers static PCC model.
"""

from __future__ import annotations

from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
from IPython.display import display
from pathlib import Path

# GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
# if not os.path.exists(GIT_COMMIT_HASH):
#     os.system('mkdir {}'.format(GIT_COMMIT_HASH))

script_dir = Path(__file__).resolve().parent
CONFIG_FILENAME = script_dir / 'sPCC_parameters.pkl'

# Each value is a single-item list because ParameterGrid expects per-key option lists.
PARAMS = {'PRIOR': ['kurtotic'],
          'USE_MASK': [False],
          'GAUSS_MASK_SIGMA': [1.0],
          'IMAGE_FILTER': ['DoG'],
          'DOG_KSIZE': [(5,5)],
          'DOG_SIGMA1': [1.3],
          'DOG_SIGMA2': [2.6],
          'INPUT_SCALE': [1.0],
          'ITER_N': [30],
          'EPOCH_N': [1000],
          'K1': [0.001],
          'K2': [0.005],
          'SS0': [1.0],
          'SS1': [10.0],
          'SS2': [10.0],
          'SS3': [2.0],
          'ALPHA1': [1.0],
          'ALPHA2': [0.05],
          'ALPHA3': [0.05],
          'LAMBDA1': [0.001],
          'LAMBDA2': [0.001],
          'LAMBDA3': [0.001],
          'RANDOM_SEED': [1],
          'RANDOM_SEED_SHUFFLE': [1],
          'CLEAR_SAVED_WEIGHTS': [True],
          'IN_DIR': ["./data/raw/slex"],
          'OUT_DIR': ["./models_plus_results/sPCC_trace212_train"],
          'RF1_SIZE': [{'x': 36, 'y': 24}],
          'RF1_OFFSET': [{'x': 32, 'y': 20}],
          'RF1_LAYOUT': [{'x': 4, 'y': 4}]}

GRID = list(ParameterGrid(PARAMS))
GRID_df = pd.DataFrame(GRID)


def main() -> None:
    """Write parameter grid entries to CONFIG_FILENAME one by one.

    Each entry overwrites the previous pickle at the same path, so only the
    last grid entry persists after the loop.
    """
    for i in GRID_df.index:
        param_df = GRID_df.iloc[i]
        param_df.to_pickle(CONFIG_FILENAME)
        timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S-%f')
        display(f"Saved config {i} at {timestamp}")


if __name__ == "__main__":
    main()
