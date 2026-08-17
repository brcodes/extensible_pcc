
"""Lexical Prediction Paradigm (LPP) simulations for the legacy rPCC.

Load a trained recurrent PCC model, run it on trained vs. novel CVCV words,
pickle the hidden-state trajectories for phoneme clustering, and optionally
produce the LPP state/error comparison plots. This is part of the validated
legacy reference implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Iterable, Optional, Union
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
for path in (script_dir, repo_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validated_legacy_models.recurrentPCC.rPCC_Model import Model, f, f_prime, softmax
from Utils import add_vertical_timestep_lines, load_trained_namespace
from kbutil.plotting import pylab_pretty_plot as pretty_plot

PLOT_TIMESTEP_TICKS = [1, 20, 40, 60, 80, 100]
PLOT_TIMESTEP_ONSETS = [2, 21, 40, 59]
PLOT_TIMESTEP_XLIM = (-1, 100)
RESULTS_DIR = script_dir / "models_plus_results" / "rPCC_cvcv12_train"
DATA_DIR = script_dir / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _apply_mpl_defaults() -> None:
	"""Reset matplotlib rcParams to the defaults used by these plots."""
	mpl.rcdefaults()
	plt.rcParams['image.aspect'] = 'auto'
	plt.rcParams['figure.dpi'] = 600


def _aggregate_for_plot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Average ``value_col`` within word, then across words per condition/timestep.

    Returns:
        DataFrame with condition_word, timestep, mean, and standard error.
    """
    by_label = (
        df[["condition_word", "label", "timestep", value_col]]
        .dropna(subset=[value_col])
        .groupby(["condition_word", "label", "timestep"], as_index=False)
        .agg(value_mean=(value_col, "mean"), value_n=(value_col, "size"))
    )

    agg = (
        by_label
        .groupby(["condition_word", "timestep"], as_index=False)
        .agg(
            mean=("value_mean", "mean"),
            n=("value_mean", "size"),
            std=("value_mean", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        )
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"])
    return agg[["condition_word", "timestep", "mean", "se"]]


def comparison_plots(d: pd.DataFrame, ylabel: str | list[str], subplots: list[str], *,
                     onsets: list[int], subplots_title: list[str] | None = None,
                     sharex: bool = True, sharey: bool = False, ncols: int = 3,
                     subplot_xy: tuple[int, int] = (4, 3), err_style: str | None = None) -> None:
    """Plot trained-vs-novel timecourse comparisons and save the LPP figure.

    Args:
        subplots: Columns of ``d`` to plot, one subplot each.
        onsets: Phoneme-onset timesteps marked with vertical lines.
        err_style: "bars" for errorbars, otherwise SE bands.
    """
    _apply_mpl_defaults()
    pretty_plot(width=2, lines=1.5, labelsize=11, fontsize=11, lfontsize=8)

    if subplots_title is None:
        subplots_title = subplots

    nrows = len(subplots) // ncols + int(bool(len(subplots) % ncols))
    subplot_x, subplot_y = subplot_xy

    fig, axes = mpl.pyplot.subplots(nrows, ncols,
                             figsize=(subplot_x * ncols, subplot_y * nrows),
                             sharex=sharex,
                             sharey=sharey,
                             constrained_layout=True)

    for ax in axes.flatten()[len(subplots):]:
        ax.remove()

    for i, (s, t, ax) in enumerate(zip(subplots, subplots_title, axes.flatten())):
        df = _aggregate_for_plot(d[["condition_word", "label", "timestep", s]], s)
        df = df.assign(timestep_plot=lambda x: x["timestep"] + 1)

        cond_order = [c for c in ["trained", "novel"] if c in df["condition_word"].unique()]
        for c in df["condition_word"].unique():
            if c not in cond_order:
                cond_order.append(c)

        palette = sns.color_palette("colorblind", n_colors=len(cond_order))
        color_map = dict(zip(cond_order, palette))

        g = sns.lineplot(data=df, ax=ax,
                 x="timestep_plot", y="mean",
                 errorbar=None,
                 err_style=err_style,
                 style="condition_word",
                 hue="condition_word",
                 hue_order=cond_order,
                 style_order=cond_order,
                 palette=color_map,
                 )

        for cond in cond_order:
            cond_df = df[df["condition_word"] == cond].sort_values("timestep_plot")
            if cond_df.empty:
                continue
            lower = cond_df["mean"] - cond_df["se"]
            upper = cond_df["mean"] + cond_df["se"]
            if err_style == "bars":
                ax.errorbar(
                    cond_df["timestep_plot"],
                    cond_df["mean"],
                    yerr=cond_df["se"],
                    fmt="none",
                    ecolor=color_map[cond],
                    elinewidth=1,
                    capsize=2,
                    alpha=0.9,
                )
            else:
                ax.fill_between(
                    cond_df["timestep_plot"],
                    lower,
                    upper,
                    color=color_map[cond],
                    alpha=0.2,
                    linewidth=0,
                )

        add_vertical_timestep_lines(ax, onsets, highlight_special=[40])

        if i == 0:
            g.legend(labels=["Trained", "Novel"], loc="upper left")
        else:
            ax.get_legend().remove()
            
        labelpad = 5

        ax.set_title(t, fontsize=13, pad=8)
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_xlim(*PLOT_TIMESTEP_XLIM)
        ax.set_xticks(PLOT_TIMESTEP_TICKS)
        ax.set_xticklabels(PLOT_TIMESTEP_TICKS)
        if type(ylabel) == str:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=labelpad)
        if type(ylabel) == list:
            ax.set_ylabel(ylabel[i], fontsize=11, labelpad=labelpad)
        if i == 0:
            ax.set_ylabel(r"$\|$Activation$\|$", labelpad=labelpad)
        if i % 2 == 1:
            ax.set_ylabel("")

    fig.savefig(RESULTS_DIR / "rPCC_LPP_after_train.pdf", dpi=600, bbox_inches='tight')
    mpl.pyplot.show()


# ---------------------------------------------------------------------------
# Text helper for Matplotlib usetex output
# ---------------------------------------------------------------------------


def latex_safe_text(text: str) -> str:
    """Escape a small set of characters that break with Matplotlib usetex."""
    if not mpl.rcParams.get('text.usetex', False):
        return text
    return (
        text
        .replace('\\', r'\textbackslash{}')
        .replace('{', r'\{')
        .replace('}', r'\}')
        .replace('_', r'\_')
        .replace('%', r'\%')
        .replace('&', r'\&')
        .replace('#', r'\#')
        .replace('^', r'{\char94}')
    )

def main(
    pkl_path: str,
    allCVCV_pkl: str,
    states_words_onsets_to_cluster: str,
    *,
    save_states_words_onsets: bool = True,
    lpp_plot: bool = False,
) -> None:
    """Run Lexical Prediction Paradigm (LPP) simulations and plots.

    Args:
        pkl_path: Trained-model pickle (trained_rpcc_*.pkl).
        allCVCV_pkl: Preprocessed inputs covering trained and novel words.
        states_words_onsets_to_cluster: Output pickle path for hidden states,
            word lists, and onset timesteps.
    """
    

    # Load trained model + params
    ns = load_trained_namespace(pkl_path)
    I_dict = ns["I_dict"]
    output_train_df = ns["output_train_df"]
    all_weights_df = ns["all_weights_df"]
    act_func = ns["act_func"]
    softmax_c = ns["softmax_c"]

    I_size = I_dict[list(I_dict.keys())[0]].shape[0]
    L_size = len(I_dict)
    r2_size = L_size

    # rebuild the model with saved params and restore final-epoch trained weights
    model = Model(I_size, ns["r1_size"], r2_size, L_size, seed=ns["weight_init_seed"])
    model.s10 = ns["s10"]
    model.s11 = ns["s11"]
    model.s21 = ns["s21"]
    model.s22 = ns["s22"]
    model.s32 = ns["s32"]
    model.alpha_1 = ns["alpha_1"]
    model.alpha_2 = ns["alpha_2"]
    model.beta_1 = ns["beta_1"]
    model.beta_2 = ns["beta_2"]
    model.gamma_1 = ns["gamma_1"]
    model.gamma_2 = ns["gamma_2"]
    model.U1 = all_weights_df.U1.iloc[-1]
    model.U2 = all_weights_df.U2.iloc[-1]
    model.V1 = all_weights_df.V1.iloc[-1]
    model.V2 = all_weights_df.V2.iloc[-1]

    # Test Model with Trained vs. Novel Words
    allCVCV_dict = pickle.load(open(allCVCV_pkl, "rb"))
    trained_words = list(output_train_df.label.unique())
    novel_words = ['b^k^', 'k^bu', 'kir^', 'riki', 'sura', 'rusa']

    trained_words_dict = {'trained': trained_words}
    novel_words_dict = {'novel': novel_words}

    all_test_words = trained_words + novel_words
    all_onsets = [2, 21, 40, 59]

    dfs = []
    all_r1_states = []
    all_r2_states = []

    for k in all_test_words:
        I = allCVCV_dict[k]
        df = model.apply_input(k, I, np.zeros(L_size), training=False, af=act_func)
        dfs.append(df)

        r1states = model.r1_hat_states
        r2states = model.r2_hat_states
        label = model.label

        if k == label:
            print("Tested word: ", k)

        all_r1_states.append(r1states)
        all_r2_states.append(r2states)

    all_test_df = pd.concat(dfs, ignore_index=True)

    print("all_r1_states shape: ", np.array(all_r1_states).shape)
    print("all_r2_states shape: ", np.array(all_r2_states).shape)
    print("all_test_words: ", all_test_words)
    print("all_onsets: ", all_onsets)
    print("trained_words: ", trained_words)
    print("novel_words: ", novel_words)


    # Optional pickle dump all words
    if save_states_words_onsets:
        with open(states_words_onsets_to_cluster, "wb") as cluster_file:
            pickle.dump([all_r1_states, all_r2_states, all_test_words, all_onsets, trained_words_dict, novel_words_dict], cluster_file)
        print(f"Pickle dumped all_r1_states, all_r2_states, all_test_words, all_onsets, trained_words_dict, novel_words_dict (for Phoneme Clustering) to {states_words_onsets_to_cluster}")
    else:
        print("Skipped pickle dump for states/words/onsets (save_states_words_onsets=False).")
    
    if not lpp_plot:
        return

    all_test_df["softmaxd_r2_hat"] = all_test_df.r2_hat.apply(lambda x: softmax(x, c=softmax_c))

    # State Unit Activations
    all_test_df["I_sum"] = all_test_df.I.apply(np.sum)
    all_test_df["r1_hat_sum"] = all_test_df.r1_hat.apply(np.sum)
    all_test_df["r2_hat_sum"] = all_test_df.r2_hat.apply(np.sum)
    all_test_df["r1_hat_abs"] = all_test_df.r1_hat.apply(np.abs)
    all_test_df["r2_hat_abs"] = all_test_df.r2_hat.apply(np.abs)
    all_test_df["r1_hat_abs_sum"] = all_test_df.r1_hat_abs.apply(np.sum)
    all_test_df["r2_hat_abs_sum"] = all_test_df.r2_hat_abs.apply(np.sum)

    # Errors
    all_test_df["e10"] = all_test_df.apply(lambda x: x.I - (model.U1 @ x.r1_hat), axis=1)
    all_test_df["e11"] = all_test_df.apply(lambda x: x.r1_hat - x.r1_bar, axis=1)
    all_test_df["e21"] = all_test_df.apply(lambda x: x.r1_hat - (model.U2 @ x.r2_hat), axis=1)
    all_test_df["e22"] = all_test_df.apply(lambda x: x.r2_hat - x.r2_bar, axis=1)
    all_test_df["e10_sum"] = all_test_df.e10.apply(np.sum)
    all_test_df["e11_sum"] = all_test_df.e11.apply(np.sum)
    all_test_df["e21_sum"] = all_test_df.e21.apply(np.sum)
    all_test_df["e22_sum"] = all_test_df.e22.apply(np.sum)
    all_test_df["e10_abs"] = all_test_df.e10.apply(np.abs)
    all_test_df["e11_abs"] = all_test_df.e11.apply(np.abs)
    all_test_df["e21_abs"] = all_test_df.e21.apply(np.abs)
    all_test_df["e22_abs"] = all_test_df.e22.apply(np.abs)
    all_test_df["e10_abs_sum"] = all_test_df.e10_abs.apply(np.sum)
    all_test_df["e11_abs_sum"] = all_test_df.e11_abs.apply(np.sum)
    all_test_df["e21_abs_sum"] = all_test_df.e21_abs.apply(np.sum)
    all_test_df["e22_abs_sum"] = all_test_df.e22_abs.apply(np.sum)

    # Information Flows
    all_test_df["r1_re"] = all_test_df.apply(lambda x: f(model.V1 @ x.r1_hat, af=act_func) - x.r1_hat, axis=1)
    all_test_df["r2_re"] = all_test_df.apply(lambda x: f(model.V2 @ x.r2_hat, af=act_func) - x.r2_hat, axis=1)
    all_test_df["r1_bu"] = all_test_df.apply(lambda x: model.U1.T @ f_prime(model.U1 @ x.r1_bar, af=act_func) @ (x.I - f(model.U1 @ x.r1_bar, af=act_func)), axis=1)
    all_test_df["r2_bu"] = all_test_df.apply(lambda x: model.U2.T @ f_prime(model.U2 @ x.r2_bar, af=act_func) @ (x.r1_bar - f(model.U2 @ x.r2_bar, af=act_func)), axis=1)
    all_test_df["r1_td"] = all_test_df.apply(lambda x: x.r1_bar - f(model.U2 @ x.r2_bar, af=act_func), axis=1)
    all_test_df["r1_re_sum"] = all_test_df.r1_re.apply(np.sum)
    all_test_df["r2_re_sum"] = all_test_df.r2_re.apply(np.sum)
    all_test_df["r1_bu_sum"] = all_test_df.r1_bu.apply(np.sum)
    all_test_df["r2_bu_sum"] = all_test_df.r2_bu.apply(np.sum)
    all_test_df["r1_td_sum"] = all_test_df.r1_td.apply(np.sum)
    all_test_df["r1_re_abs"] = all_test_df.r1_re.apply(np.abs)
    all_test_df["r2_re_abs"] = all_test_df.r2_re.apply(np.abs)
    all_test_df["r1_bu_abs"] = all_test_df.r1_bu.apply(np.abs)
    all_test_df["r2_bu_abs"] = all_test_df.r2_bu.apply(np.abs)
    all_test_df["r1_td_abs"] = all_test_df.r1_td.apply(np.abs)
    all_test_df["r1_re_abs_sum"] = all_test_df.r1_re_abs.apply(np.sum)
    all_test_df["r2_re_abs_sum"] = all_test_df.r2_re_abs.apply(np.sum)
    all_test_df["r1_bu_abs_sum"] = all_test_df.r1_bu_abs.apply(np.sum)
    all_test_df["r2_bu_abs_sum"] = all_test_df.r2_bu_abs.apply(np.sum)
    all_test_df["r1_td_abs_sum"] = all_test_df.r1_td_abs.apply(np.sum)

    # Global Plot Settings
    mpl.rcdefaults()
    mpl.pyplot.style.use('seaborn-paper')
    mpl.pyplot.rcParams['image.aspect'] = 'auto'
    mpl.pyplot.rcParams['figure.dpi'] = 600
    mpl.pyplot.rcParams['mathtext.fontset'] = 'dejavuserif'
    sns.set_palette("colorblind")

    # Add condition_word column (trained vs. novel)
    all_test_df["condition_word"] = all_test_df["label"].apply(
        lambda word: "trained" if word in trained_words else "novel"
    )

    comparison_plots(
        d=all_test_df,
        subplots=["r1_hat_abs_sum", "r2_hat_abs_sum", "e10_abs_sum", "e21_abs_sum", "e11_abs_sum", "e22_abs_sum"],
        subplots_title=["$\\hat{r}_{1}(t)$", "$\\hat{r}_{2}(t)$", "$e_{1}(t)$", "$e_{2}(t)$", "$e'_{1}(t)$", "$e'_{2}(t)$"],
        ylabel=r"$\|$Prediction error$\|$",
        ncols=2,
        sharey=False,
        sharex=True,
        subplot_xy=(3, 2),
        err_style="band",
        onsets=all_onsets
    )


if __name__ == "__main__":
    
    # Model, inputs
    pkl_path = str(RESULTS_DIR / "trained_rpcc_1.pkl")
    allCVCV_pkl = str(DATA_DIR / "CVCV_all_prepro.pkl")
    
    # The main plot. Set to False to skip the LPP comparison plot (main returns
    # after the pickle dump).
    lpp_plot = True
    # Save state results/words/onset timesteps for phonological clustering
    save_states_words_onsets = True
    states_words_onsets_to_cluster = str(RESULTS_DIR / "LPP_hiddenstates_wordlist_phononsets.pkl")

    main(
        pkl_path=pkl_path,
        allCVCV_pkl=allCVCV_pkl,
        states_words_onsets_to_cluster=states_words_onsets_to_cluster,
        save_states_words_onsets=save_states_words_onsets,              # saves the states/words/onsets pickle
        lpp_plot=lpp_plot
    )
