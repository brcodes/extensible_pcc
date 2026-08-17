"""Legacy static PCC trainer for the trace212 word-identification task.

Load the pickled trace212 hyperparameters, train the static PCC model over
epochs with per-epoch shuffled datasets, checkpoint weights, evaluate
classification accuracy per saved epoch, and plot results. This is part of
the validated legacy reference implementation of the Li/Rogers static PCC
model; it also exposes parity helpers used by the equivalence tests.
"""

from __future__ import annotations

from typing import Iterable

from datetime import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import re
import glob
import os
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

script_dir = Path(__file__).resolve().parent

from validated_legacy_models.staticPCC.Dataset import Dataset
from validated_legacy_models.staticPCC.sPCC_Model import Model


def load_parity_params(params_path: str | Path | None = None) -> pd.Series:
    """Load the trace212 parameters used by the relocated legacy trainer."""
    if params_path is None:
        params_path = Path(__file__).resolve().parent / 'sPCC_parameters.pkl'
    return pd.read_pickle(params_path)


def build_parity_dataset(param: pd.Series, shuffle: bool = False) -> Dataset:
    """Build the legacy trace212 Dataset with the trainer's active settings."""
    legacy_dir = Path(__file__).resolve().parent
    return Dataset(
        scale=param.INPUT_SCALE,
        shuffle=shuffle,
        data_dir=str(legacy_dir / param.IN_DIR),
        rf1_x=param.RF1_SIZE['x'],
        rf1_y=param.RF1_SIZE['y'],
        rf1_offset_x=param.RF1_OFFSET['x'],
        rf1_offset_y=param.RF1_OFFSET['y'],
        rf1_layout_x=param.RF1_LAYOUT['x'],
        rf1_layout_y=param.RF1_LAYOUT['y'],
        use_mask=param.USE_MASK,
        gauss_mask_sigma=param.GAUSS_MASK_SIGMA,
        image_filter=param.IMAGE_FILTER,
        DoG_ksize=param.DOG_KSIZE,
        DoG_sigma1=param.DOG_SIGMA1,
        DoG_sigma2=param.DOG_SIGMA2,
    )


def configure_parity_model(model: Model, param: pd.Series) -> Model:
    """Apply the legacy trainer's active trace212 model settings."""
    model.iteration = param.ITER_N
    model.prior = param.PRIOR
    model.k_r = param.K1
    model.k_U_init = param.K2
    model.k_U = model.k_U_init
    model.sigma_sq0 = param.SS0
    model.sigma_sq1 = param.SS1
    model.sigma_sq2 = param.SS2
    model.sigma_sq3 = param.SS3
    model.alpha1 = param.ALPHA1
    model.alpha2 = param.ALPHA2
    model.alpha3 = param.ALPHA3
    model.lambda1 = param.LAMBDA1
    model.lambda2 = param.LAMBDA2
    model.lambda3 = param.LAMBDA3
    return model


def build_parity_model(dataset: Dataset, param: pd.Series, seed: int) -> Model:
    """Build the configured legacy trace212 model with a deterministic seed."""
    np.random.seed(seed)
    return configure_parity_model(Model(dataset=dataset), param)


def train_parity_model(model: Model, param: pd.Series, epochs: int, seed_shuffle: int,
                       input_indices: Iterable[int] | None = None, shuffle: bool = True,
                       dataset: Dataset | None = None) -> Model:
    """Run bounded legacy trace212 epochs through Model.train()."""
    np.random.seed(seed_shuffle)
    for _ in range(int(epochs)):
        train_set = dataset if dataset is not None else build_parity_dataset(param, shuffle=shuffle)
        model.train(train_set, input_indices=input_indices)
    return model


def inspect_parity_model(model: Model, dataset: Dataset,
                         input_indices: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return legacy trace212 top states, probabilities, and accuracy."""
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


def category_given_target(target: str, word: str) -> str:
    """Assign a lexical-relation category (target/cohort/rhyme/embedded/other) for a word."""
    if target == word:
        return 'target'
    if target[0:2] == word[0:2]:
        return 'cohort'
    if target[1:] == word[1:]:
        return 'rhyme'
    if word in target:
        return 'embedded'
    return 'other'


def main() -> None:
    """Run the full trace212 training, evaluation, and plotting pipeline."""
    import pytz

    print('Start:', datetime.now(pytz.timezone('US/Eastern')).strftime('%c'))

    # Parameters
    param = pd.read_pickle(script_dir / 'sPCC_parameters.pkl')
    print(f'params are:\n{param}')

    in_dir = os.path.join(script_dir, param.IN_DIR)
    out_dir = os.path.join(script_dir, param.OUT_DIR)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    random_seed = param.get('RANDOM_SEED')
    if random_seed is not None:
        np.random.seed(int(random_seed))

    test_set = Dataset(
        scale=param.INPUT_SCALE,
        shuffle=False,
        data_dir=in_dir,
        rf1_x=param.RF1_SIZE['x'],
        rf1_y=param.RF1_SIZE['y'],
        rf1_offset_x=param.RF1_OFFSET['x'],
        rf1_offset_y=param.RF1_OFFSET['y'],
        rf1_layout_x=param.RF1_LAYOUT['x'],
        rf1_layout_y=param.RF1_LAYOUT['y'],
        use_mask=param.USE_MASK,
        gauss_mask_sigma=param.GAUSS_MASK_SIGMA,
        image_filter=param.IMAGE_FILTER,
        DoG_ksize=param.DOG_KSIZE,
        DoG_sigma1=param.DOG_SIGMA1,
        DoG_sigma2=param.DOG_SIGMA2,
    )

    epoch_n = param.EPOCH_N
    zero_pad_len = len(str(epoch_n))

    model = Model(dataset=test_set)
    model.iteration = param.ITER_N
    model.prior = param.PRIOR
    model.k_r = param.K1
    model.k_U_init = param.K2
    model.sigma_sq0 = param.SS0
    model.sigma_sq1 = param.SS1
    model.sigma_sq2 = param.SS2
    model.sigma_sq3 = param.SS3
    model.alpha1 = param.ALPHA1
    model.alpha2 = param.ALPHA2
    model.alpha3 = param.ALPHA3
    model.lambda1 = param.LAMBDA1
    model.lambda2 = param.LAMBDA2
    model.lambda3 = param.LAMBDA3

    # Plotting defaults
    mpl.rcdefaults()
    plt.style.use('seaborn-paper')
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    sns.set_palette('colorblind')
    sns.set_context(context='paper', font_scale=1.2, rc=None)

    # Quick example heatmaps
    cmap = 'cividis'
    nrows, ncols = 1, 2
    subplot_xy = (4.5, 3)
    figsize = tuple([(ncols, nrows)[i] * subplot_xy[i] for i in range(2)])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    ax1 = axes[0]
    ax2 = axes[1]
    sns.heatmap(
        test_set.images[0],
        cmap=cmap,
        cbar=True,
        cbar_kws={'shrink': 0.6},
        square=True,
        xticklabels=False,
        yticklabels=False,
        vmin=np.floor(np.min([x.min() for x in test_set.images])),
        vmax=np.ceil(np.max([x.max() for x in test_set.images])),
        ax=ax1,
    )
    ax1.set_title('Original Image')
    sns.heatmap(
        test_set.filtered_images[0],
        cmap=cmap,
        cbar=True,
        cbar_kws={'shrink': 0.6},
        square=True,
        xticklabels=False,
        yticklabels=False,
        vmin=np.floor(np.min([x.min() for x in test_set.filtered_images])),
        vmax=np.ceil(np.max([x.max() for x in test_set.filtered_images])),
        ax=ax2,
    )
    ax2.set_title('Edge Detected Image')
    fig.suptitle('/b^s/')
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    if param.CLEAR_SAVED_WEIGHTS and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Resume from the latest epoch checkpoint, or create/load the pretraining snapshot.
    out_dir_all = glob.glob(os.path.join(out_dir, '*'))
    out_dir_epoch = glob.glob(os.path.join(out_dir, 'epoch_*'))
    out_dir_pretrain = os.path.join(out_dir, 'pretraining')

    if len(out_dir_epoch) > 0:
        regex = re.compile(os.path.join(out_dir, 'epoch_(?P<epoch>.*)'))
        epoch_all = [int(regex.match(x).group('epoch')) for x in out_dir_epoch]
        epoch_max_idx = np.argmax(epoch_all)
        epoch_max = epoch_all[epoch_max_idx]
        model.load(out_dir_epoch[epoch_max_idx])
    elif out_dir_pretrain not in out_dir_all:
        epoch_max = -1
        model.save(os.path.join(out_dir, 'pretraining'))
    else:
        epoch_max = -1
        model.load(os.path.join(out_dir, 'pretraining'))

    random_seed_shuffle = param.get('RANDOM_SEED_SHUFFLE')
    if random_seed_shuffle is not None:
        np.random.seed(int(random_seed_shuffle))

    # Epoch loop: rebuild a shuffled dataset each epoch; checkpoint at epoch 0 and every 10th.
    for i in [x for x in range(epoch_n) if x > epoch_max]:
        train_set = Dataset(
            scale=param.INPUT_SCALE,
            shuffle=True,
            data_dir=in_dir,
            rf1_x=param.RF1_SIZE['x'],
            rf1_y=param.RF1_SIZE['y'],
            rf1_offset_x=param.RF1_OFFSET['x'],
            rf1_offset_y=param.RF1_OFFSET['y'],
            rf1_layout_x=param.RF1_LAYOUT['x'],
            rf1_layout_y=param.RF1_LAYOUT['y'],
            use_mask=param.USE_MASK,
            gauss_mask_sigma=param.GAUSS_MASK_SIGMA,
            image_filter=param.IMAGE_FILTER,
            DoG_ksize=param.DOG_KSIZE,
            DoG_sigma1=param.DOG_SIGMA1,
            DoG_sigma2=param.DOG_SIGMA2,
        )
        model.train(train_set)

        if i == 0 or i % 10 == 9:
            model.save(os.path.join(out_dir, f'epoch_{i:0>{zero_pad_len}d}'))

    filenames = sorted(glob.glob(os.path.join(in_dir, '*.png')))
    regex = re.compile(os.path.join(in_dir, '(?P<index>.*)_(?P<word>.*).png'))
    f_word = [regex.match(x).group('word') for x in filenames]

    def load_data(i: int) -> pd.DataFrame:
        """Evaluate the epoch-i checkpoint on the test set and return per-node results."""
        results = {
            'epoch': [], 'target': [], 'node': [], 'node_word': [], 'target_label': [],
            'activation_raw': [], 'activation': [], 'target_n': [], 'response_n': [], 'accuracy': [],
        }

        model.load(os.path.join(out_dir, f'epoch_{i:0>{zero_pad_len}d}'))
        for j in range(len(test_set.rf2_patches)):
            inputs = test_set.get_rf1_patches(j)
            label = test_set.labels[j]
            r1, r2, r3, e1, e2, e3 = model.apply_input(inputs, label, training=False)

            target_n = np.argmax(label)
            r3_raw = r3.astype(np.float128)
            r3 = np.exp(r3_raw) / np.sum(np.exp(r3_raw))
            response_n = None if sum(r3 == r3.max()) != 1 else int(np.argmax(r3))
            accuracy = 1 if target_n == response_n else 0

            results['epoch'] += [i] * len(r3)
            results['target'] += [f_word[target_n]] * len(r3)
            results['node'] += list(range(len(r3)))
            results['node_word'] += [f_word[x] for x in range(len(r3))]
            results['target_label'] += list(label)
            results['activation_raw'] += list(r3_raw)
            results['activation'] += list(r3)
            results['target_n'] += [target_n] * len(r3)
            results['response_n'] += [response_n] * len(r3)
            results['accuracy'] += [accuracy] * len(r3)

        return pd.DataFrame.from_dict(results, dtype='float128')

    df_list = [load_data(i) for i in range(epoch_n) if i == 0 or i % 10 == 9]
    results_df = pd.concat(df_list, ignore_index=True)
    results_df.to_pickle(os.path.join(out_dir, 'results.pkl'))

    results_df['category'] = results_df.apply(
        lambda x: category_given_target(x['target'], x['node_word']), axis=1)
    results_df.category = results_df.category.astype('category').cat.set_categories(
        ['target', 'cohort', 'rhyme', 'other', 'embedded'])

    plt.figure(figsize=(4, 3))
    ax = results_df.groupby('epoch')['accuracy'].mean().plot()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, symbol=None))
    plt.savefig(os.path.join(out_dir, 'sPCC_train_acc.png'), dpi=600, bbox_inches='tight')
    plt.show()

    print('End:', datetime.now(pytz.timezone('US/Eastern')).strftime('%c'))


if __name__ == '__main__':
    main()
