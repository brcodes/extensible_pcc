#!/usr/bin/env python
# coding: utf-8
"""Legacy trainer for the recurrent predictive-coding classifier (rPCC).

Train the two-layer recurrent model on the CVCV_12 lexical dataset with
seeded weight initialization and per-epoch shuffling, compute recognition
accuracy, and pickle parameters, per-timestep outputs, and weight snapshots.
This is the validated legacy reference implementation; the parity_* helpers
expose the exact settings and loops used by numerical parity tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

script_dir = Path(__file__).resolve().parent

from validated_legacy_models.recurrentPCC.rPCC_Model import Model, category_given_target, recog, softmax


def parity_params() -> dict[str, Any]:
    """Return the published recurrent settings used by the legacy trainer."""
    return {
        'act_func': 'linear',
        'alpha_1': 0.1,
        'alpha_2': 5.0,
        'beta_1': 0.1,
        'beta_2': 0.1,
        'gamma_1': 0.01,
        'gamma_2': 0.01,
        'r1_size': 10,
        's10': 10,
        's11': 1,
        's21': 10,
        's22': 1,
        's32': 5,
    }


def load_parity_inputs(data_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load the CVCV12 inputs used by the relocated legacy recurrent model."""
    if data_path is None:
        data_path = script_dir / 'data' / 'CVCV_12_prepro.pkl'
    with Path(data_path).open('rb') as input_file:
        return pickle.load(input_file)


def build_parity_model(input_dict: dict[str, np.ndarray], seed: int,
                       params: dict[str, Any] | None = None) -> Model:
    """Build and configure the legacy recurrent model through trainer settings."""
    params = parity_params() if params is None else dict(params)
    first_input = next(iter(input_dict.values()))
    class_count = len(input_dict)
    model = Model(first_input.shape[0], params['r1_size'], class_count, class_count, seed=seed)
    for attribute, key in (
        ('s10', 's10'), ('s11', 's11'), ('s21', 's21'), ('s22', 's22'), ('s32', 's32'),
        ('alpha_1', 'alpha_1'), ('alpha_2', 'alpha_2'), ('beta_1', 'beta_1'),
        ('beta_2', 'beta_2'), ('gamma_1', 'gamma_1'), ('gamma_2', 'gamma_2'),
    ):
        setattr(model, attribute, params[key])
    return model


def train_parity_model(model: Model, input_dict: dict[str, np.ndarray], epochs: int,
                       seed_shuffle: int, act_func: str = 'linear', num_ts: int | None = None,
                       word_indices: Sequence[int] | None = None) -> Model:
    """Train the legacy model using its own seeded per-epoch shuffling (mirrors main())."""
    words = list(input_dict)
    labels = np.eye(len(words), dtype=np.float64)
    if word_indices is not None:
        epoch_orders = [list(word_indices) for _ in range(epochs)]
    else:
        # Global-stream seeding matches the active extensible path (train_model.run_experiment:
        # np.random.seed(seed_shuffle) then np.random.permutation per epoch), so both paths
        # draw identical epoch orders from the same seed.
        np.random.seed(seed_shuffle)
        epoch_orders = [np.random.permutation(len(words)) for _ in range(epochs)]
    for order in epoch_orders:
        for index in order:
            word_index = int(index)
            word = words[word_index]
            input_value = input_dict[word]
            if num_ts is not None:
                input_value = input_value[:, :int(num_ts)]
            model.apply_input(word, input_value, labels[word_index], training=True, af=act_func)
    return model


def inspect_parity_model(model: Model, input_dict: dict[str, np.ndarray], act_func: str = 'linear',
                         word_indices: Sequence[int] | None = None,
                         num_ts: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return legacy final states, c=1 probabilities, and recognition accuracy."""
    words = list(input_dict)
    class_count = len(words)
    if word_indices is None:
        word_indices = range(class_count)
    else:
        word_indices = tuple(int(index) for index in word_indices)

    states = []
    probabilities = []
    correct = 0
    for index in word_indices:
        word = words[index]
        input_value = input_dict[word]
        if num_ts is not None:
            input_value = input_value[:, :int(num_ts)]
        output = model.apply_input(
            word, input_value, np.zeros(class_count), training=False, af=act_func
        )
        state = np.asarray(output['r2_hat'].iloc[-1], dtype=np.float64)
        probability = softmax(state, c=1)
        states.append(state)
        probabilities.append(probability)
        response = int(np.argmax(probability)) if np.sum(probability == probability.max()) == 1 else None
        correct += int(index == response)
    return np.stack(states), np.stack(probabilities), correct / len(word_indices)


def main(num_training_runs: int = 1) -> None:
    """Train the recurrent PCC model for the specified number of runs.

    Each run trains from scratch with run-indexed seeds, evaluates train/test
    recognition, and pickles the full results dict to out_dir.
    """

    for run in range(num_training_runs):
        # parameters for successful rPCC, as understood by BR 2023.01.06
        param_dict = {
            'act_func': 'linear',
            'alpha_1': 0.1,
            'alpha_2': 5,
            'beta_1': 0.1,
            'beta_2': 0.1,
            'cohort_len': 2,
            'epoch_n': 5000,
            'gamma_1': 0.01,
            'gamma_2': 0.01,
            'in_dir': str(script_dir / 'data'),
            'ipynb': 'toy_model_kalman_variant_4.ipynb',
            'nlines_max': 16,
            'out_dir': str(script_dir / 'models_plus_results' / 'rPCC_cvcv12_train'),
            'plot_weights': False,
            'r1_size': 10,
            'recog_mode': 1,
            'recog_value': 0,
            'rhyme_len': 3,
            's10': 10,
            's11': 1,
            's21': 10,
            's22': 1,
            's32': 5,
            'save_interval': 100,
            'softmax_c': 20,
            'timecourse_sharey': True,
            'timestamp': '2021-05-12_16-46-25',
            'weight_init_seed': run + 1,
            'shuffle_seed': run + 1,
            'param_id': 'a2887b6db7',
        }

        print(f'params are {param_dict}')
        print(f'run is {run}')
        print(f'seed is {param_dict["weight_init_seed"]}')

        param = pd.Series(param_dict)
        act_func = param['act_func']
        alpha_1 = param['alpha_1']
        alpha_2 = param['alpha_2']
        beta_1 = param['beta_1']
        beta_2 = param['beta_2']
        cohort_len = param['cohort_len']
        epoch_n = param['epoch_n']
        gamma_1 = param['gamma_1']
        gamma_2 = param['gamma_2']
        in_dir = param['in_dir']
        out_dir = param['out_dir']
        r1_size = param['r1_size']
        recog_mode = param['recog_mode']
        recog_value = param['recog_value']
        rhyme_len = param['rhyme_len']
        s10 = param['s10']
        s11 = param['s11']
        s21 = param['s21']
        s22 = param['s22']
        s32 = param['s32']
        save_interval = param['save_interval']
        softmax_c = param['softmax_c']
        weight_init_seed = param['weight_init_seed']
        shuffle_seed = param['shuffle_seed']

        os.makedirs(out_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------
        with open(os.path.join(in_dir, "CVCV_12_prepro.pkl"), "rb") as f:
            I_dict = pickle.load(f)

        I = I_dict["kisa"]

        epoch_max = max(range(epoch_n))
        save_epochs = [x for x in range(epoch_n) if x == 0 or x % save_interval == save_interval - 1]

        I_size = I.shape[0]
        L_size = len(I_dict)
        r2_size = L_size

        model = Model(I_size, r1_size, r2_size, L_size, seed=weight_init_seed)

        model.s10 = s10
        model.s11 = s11
        model.s21 = s21
        model.s22 = s22
        model.s32 = s32

        model.alpha_1 = alpha_1
        model.alpha_2 = alpha_2
        model.beta_1 = beta_1
        model.beta_2 = beta_2
        model.gamma_1 = gamma_1
        model.gamma_2 = gamma_2

        all_weights = {"epoch": [], "U1": [], "U2": [], "V1": [], "V2": []}
        output_train_df = pd.DataFrame()
        output_test_df = pd.DataFrame()

        # Seed the shuffle stream after model init, mirroring the extensible path
        # (seed_init -> build model -> seed_shuffle -> train), so both paths draw
        # identical per-epoch permutations from the same global stream.
        np.random.seed(shuffle_seed)

        print('training start')

        for epoch in range(epoch_n):
            print(f'epoch {epoch}')
            if epoch in save_epochs:
                for k in I_dict:
                    df = model.apply_input(k, I_dict[k], np.zeros(L_size), training=False, af=act_func)
                    df["epoch"] = epoch
                    output_test_df = output_test_df.append(df, ignore_index=True)

            # deterministic shuffling per epoch: global stream seeded once with shuffle_seed,
            # matching the extensible trainer's np.random.permutation draws exactly
            words = list(I_dict.keys())
            I_order = [words[i] for i in np.random.permutation(len(words))]

            for k in I_order:
                k_idx = list(I_dict.keys()).index(k)
                L = np.eye(len(I_dict))[k_idx]
                df = model.apply_input(k, I_dict[k], L, training=True, af=act_func)

                if epoch in save_epochs:
                    df["epoch"] = epoch
                    output_train_df = output_train_df.append(df, ignore_index=True)


            if epoch in save_epochs:
                all_weights["epoch"].append(epoch)
                all_weights["U1"].append(model.U1)
                all_weights["U2"].append(model.U2)
                all_weights["V1"].append(model.V1)
                all_weights["V2"].append(model.V2)

        all_weights_df = pd.DataFrame.from_dict(all_weights)

        # legacy notebook analysis blocks: r2_df_long is recomputed per block and unused here
        # ## First Epoch (Training)
        r2_ss = output_train_df.groupby("epoch").get_group(0).set_index(["label", "timestep"]).r2_hat
        r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
        r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
        r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)

        # ### Testing (Epoch 0)
        r2_ss = output_test_df.groupby("epoch").get_group(0).set_index(["label", "timestep"]).r2_hat
        r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
        r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
        r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)

        # ## Last Epoch (Training)
        r2_ss = output_train_df.groupby("epoch").get_group(epoch_max).set_index(["label", "timestep"]).r2_hat
        r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
        r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
        r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)

        # ### Testing (Last Epoch)
        r2_ss = output_test_df.groupby("epoch").get_group(epoch_max).set_index(["label", "timestep"]).r2_hat
        r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
        r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
        r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)

        output_test_df["r1_hat_recon"] = output_test_df.apply(lambda x: model.U1 @ x.r1_hat, axis=1)
        output_test_df["r1_bar_recon"] = output_test_df.apply(lambda x: model.U1 @ x.r1_bar, axis=1)
        output_test_df["r1_recon_diff"] = output_test_df.apply(lambda x: x.r1_bar_recon - x.r1_hat_recon, axis=1)

        # # Accuracy
        output_train_df["softmaxd_r2_hat"] = output_train_df.apply(lambda x: softmax(x.r2_hat, c=softmax_c), axis=1)

        train_recog_df = output_train_df.groupby(["epoch", "label"]).softmaxd_r2_hat.apply(lambda x: recog(x, mode=recog_mode, value=recog_value)).rename("recog_node").to_frame()
        output_train_df = pd.merge(output_train_df, train_recog_df,
                            left_on=["epoch", "label"], right_on=["epoch", "label"],
                            how="inner", suffixes=("_old", ""))
        output_train_df["label_node"] = output_train_df.apply(lambda x: list(I_dict.keys()).index(x.label), axis=1)
        output_train_df["accuracy"] = output_train_df.apply(lambda x: int(x.label_node == x.recog_node), axis=1)

        output_test_df["softmaxd_r2_hat"] = output_test_df.apply(lambda x: softmax(x.r2_hat, c=softmax_c), axis=1)

        test_recog_df = output_test_df.groupby(["epoch", "label"]).softmaxd_r2_hat.apply(lambda x: recog(x, mode=recog_mode, value=recog_value)).rename("recog_node").to_frame()
        output_test_df = pd.merge(output_test_df, test_recog_df,
                             left_on=["epoch", "label"], right_on=["epoch", "label"],
                             how="inner", suffixes=("_old", ""))
        output_test_df["label_node"] = output_test_df.apply(lambda x: list(I_dict.keys()).index(x.label), axis=1)
        output_test_df["accuracy"] = output_test_df.apply(lambda x: int(x.label_node == x.recog_node), axis=1)

        output_dict = {
            "param": param,
            "I_dict": I_dict,
            "output_test_df": output_test_df,
            "output_train_df": output_train_df,
            "all_weights_df": all_weights_df,
        }

        output_pkl = f"trained_rpcc_{run + 1}.pkl"
        output_pkl_path = os.path.join(out_dir, output_pkl)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(output_pkl_path, "wb") as output_file:
            pickle.dump(output_dict, output_file)
        print(f"output pickle is saved at {output_pkl_path}")

    print(f"\n{num_training_runs} training runs are completed.")


if __name__ == "__main__":
    main()