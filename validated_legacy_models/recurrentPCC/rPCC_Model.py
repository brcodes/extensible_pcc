"""Legacy recurrent predictive-coding classifier (rPCC) model.

Defines the two-layer recurrent predictive-coding classifier used for the
CVCV lexical tasks, plus the softmax/activation helpers, word-recognition
metrics, and plotting utilities that operate on its outputs. This is the
validated legacy reference implementation against which the extensible
models in pypredcoding/ are numerically parity-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import os

def softmax(x: np.ndarray, c: float = 1) -> np.ndarray:
    """Compute a max-shifted (numerically stable) softmax with gain ``c``."""
    x = np.array(x)
    x = x - x.max()
    return np.exp(x*c)/np.sum(np.exp(x*c))

def f(x: np.ndarray, af: str = "linear") -> np.ndarray:
    """Apply the activation function named by ``af`` ("linear" or "tanh")."""
    if af == "linear":
        fx = x
    elif af == "tanh":
        fx = np.tanh(x)
    return fx

def f_prime(x: np.ndarray, af: str = "linear") -> np.ndarray:
    """Return the activation-function derivative as a diagonal Jacobian matrix."""
    if af == "linear":
        dfx = np.eye(x.size)
    elif af == "tanh":
        dfx = np.diag(1 - np.square(np.tanh(x)))
    return dfx

def raw_vs_softmax_plots(df: pd.DataFrame, nrows: int, subplot_yx: tuple, scale: float,
                         groupby: str, value: str, nodes: list, nlines_max: int = 10,
                         c: float = 1, sharex: bool = False, sharey: bool = False,
                         cmap: str | None = None) -> None:
    """Plot raw vs. softmax-transformed activation timecourses, one row per group.

    Args:
        groupby: Column of ``df`` to group by (one subplot row per group key).
        value: Column holding the per-timestep activation vectors.
        nodes: Column labels (word names) for the activation dimensions.
    """
    if cmap == None:
        # note: the nlines_max > 20 branch is unreachable (> 10 matches first)
        if nlines_max > 10:
            cmap="tab20"
        elif nlines_max > 20:
            cmap="turbo"

    df_groupby = df.groupby(groupby)
    group_keys = list(df_groupby.groups.keys())
    
    ncols = 2
    nrows = nrows if nrows <= len(group_keys) else len(group_keys)
    
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey)
    
    for k in group_keys[:nrows]:
        k_idx = group_keys.index(k)

        raw_r = df_groupby.get_group(k).apply(lambda x: pd.Series(x[value]), axis=1).reset_index(drop=True)
        softmaxd_r = df_groupby.get_group(k).apply(lambda x: pd.Series(softmax(x[value], c=c)), axis=1).reset_index(drop=True)

        raw_r.columns = nodes
        softmaxd_r.columns = nodes
        
        if len(nodes) > nlines_max:
            raw_r = raw_r.filter(items=raw_r.tail(1).squeeze().sort_values(ascending=False).head(nlines_max).index)
            softmaxd_r = softmaxd_r.filter(items=softmaxd_r.tail(1).squeeze().sort_values(ascending=False).head(nlines_max).index)

        raw_plot = raw_r.plot(ax=axes[k_idx, 0], title="{}: raw activations".format(k), cmap=cmap);
        softmaxd_plot = softmaxd_r.plot(ax=axes[k_idx, 1], title="{}: softmax'd activations".format(k), cmap=cmap);
        
        raw_plot.legend(loc="lower left");
        softmaxd_plot.legend(loc="lower left");
        
        # thicken target line
        for i, j, l1, l2 in zip(raw_r.columns, softmaxd_r.columns, raw_plot.lines, softmaxd_plot.lines):
            if i == k:
                plt.setp(l1, linewidth=3)
            if j == k:
                plt.setp(l2, linewidth=3)

    plt.tight_layout();
    
def recon_plots(df: pd.DataFrame, ncols: int, nrows: int, subplot_yx: tuple, scale: float,
                groupby: str, value: str, vmin: float | None = None, vmax: float | None = None,
                sharex: bool = True, sharey: bool = True, y_label_list: list | None = None,
                cmap: str = "gray", cbar_shrink: float = 0.5, cbar_pad: float = 0.01) -> None:
    """Show a heatmap of a vector-valued column over timesteps, one panel per group."""
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey, constrained_layout=True)

    df_groupby = df.groupby(groupby)
    group_keys = df_groupby.groups.keys()
    
    if vmax == None:
        vmax = df.apply(lambda x: np.abs(pd.Series(x[value])), axis=1).values.max()
    if vmin == None:
        vmin = -vmax
    
    for k, ax in zip(group_keys, axes.flatten()):
        vall = df_groupby.get_group(k).apply(lambda x: pd.Series(x[value]), axis=1).values.T
        im = ax.imshow(vall, cmap=cmap, vmin=vmin, vmax=vmax)
    
        if y_label_list != None:
            ax.set_yticks(range(len(y_label_list)))
            ax.set_yticklabels(y_label_list)

        ax.set_title(k)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=cbar_shrink, pad=cbar_pad);
    
def argmax_1(x: np.ndarray) -> int | float:
    """Return the argmax index if the maximum is unique, otherwise NaN."""
    x = np.array(x)
    max_marks = [1 if i == x.max() else 0 for i in x]
    argmax_x = np.argmax(max_marks) if np.sum(max_marks) == 1 else np.nan
    return argmax_x

def recog(L: pd.Series, mode: int = 1, value: float = 0) -> pd.Series | float:
    """Operationalize word recognition over a timecourse of activation vectors.

    Args:
        L: Series of per-timestep activation vectors (raw or transformed).
        mode: 1 = most-activated node at the last timestep, if above ``value``;
            2 = first node that stays at the maximum for at least ``value``
            timesteps; 3 = first node exceeding absolute threshold ``value``.

    Returns:
        Recognized node index (length-1 Series) or NaN if no node qualifies.
    """
    if mode == 1:
        L_argmax = L.apply(argmax_1)
        L_max = L.apply(np.max)
        recog_node = L_argmax.tail(1) if L_max.tail(1).squeeze() >= value else np.nan
    elif mode == 2:
        L_argmax = L.apply(lambda x: argmax_1(x))
        X = np.array([[k, sum(1 for i in g)] for k, g in groupby(L_argmax)])
        df = pd.DataFrame(X, columns=["max_node", "steps"])
        max_nodes = df.query("steps >= {}".format(value))["max_node"]
        recog_node = max_nodes.head(1) if len(max_nodes) > 0 else np.nan
    elif mode == 3:
        L_argmax = L.apply(lambda x: argmax_1(x > value))
        X = np.array([[k, sum(1 for i in g)] for k, g in groupby(L_argmax)])
        df = pd.DataFrame(X, columns=["max_node", "steps"])
        max_nodes = df.query("max_node.notna()", engine="python")["max_node"]
        recog_node = max_nodes.head(1) if len(max_nodes) > 0 else np.nan

    return recog_node

def category_given_target(target: str, word: str, cohort_len: int = 1, rhyme_len: int = 1) -> str:
    """Classify ``word`` relative to ``target`` as target/cohort/rhyme/embedded/other."""

    if target == word:
        category = 'target'
    elif target[:cohort_len] == word[:cohort_len]:
        category = 'cohort'
    elif target[-rhyme_len:] == word[-rhyme_len:]:
        category = 'rhyme'
    elif word in target:
        category = 'embedded'
    else:
        category = 'other'

    return category

class Model:
    """Two-layer recurrent predictive-coding classifier.

    Holds top-down prediction weights (U1, U2), recurrent temporal weights
    (V1, V2), variance weightings (s10, s11, s21, s22, s32), and state/weight
    update rates (alpha_*, beta_*, gamma_*); all are plain attributes that
    callers override after construction.
    """

    def __init__(self, I_size: int, r1_size: int, r2_size: int, L_size: int,
                 seed: int | None = None) -> None:
        """Initialize weights from N(0, 0.1) with optional seed; all rates default to 1."""
        self.I_size = I_size
        self.r1_size = r1_size
        self.r2_size = r2_size
        self.L_size = L_size
        
        np.random.seed(seed)
        self.U1 = np.random.normal(loc=0, scale=0.1, size=(I_size, r1_size))
        self.U2 = np.random.normal(loc=0, scale=0.1, size=(r1_size, r2_size))
        self.V1 = np.random.normal(loc=0, scale=0.1, size=(r1_size, r1_size))
        self.V2 = np.random.normal(loc=0, scale=0.1, size=(r2_size, r2_size))
        
        self.s10 = 1
        self.s11 = 1
        self.s21 = 1
        self.s22 = 1
        self.s32 = 1
        
        self.alpha_1 = 1
        self.alpha_2 = 1
        self.beta_1 = 1
        self.beta_2 = 1
        self.gamma_1 = 1
        self.gamma_2 = 1
        
    def apply_input(self, label: str, I: np.ndarray, L: np.ndarray, training: bool = False,
                    af: str = "linear", r2_priming: list | None = None) -> pd.DataFrame:
        """Run one input sequence through the model, optionally updating weights.

        Each timestep first computes "bar" states predicted from the previous
        timestep's "hat" states via V, then "hat" states corrected by
        bottom-up and top-down prediction errors; r2_hat_x additionally
        receives the classification correction (softmax(r2_bar) vs. L).
        With ``training=True``, U and V weights are updated every timestep.

        Args:
            I: Input array with timesteps as columns (shape features x T).
            L: One-hot label vector (zeros when testing).
            r2_priming: Optional initial value for r2_hat and r2_hat_x. Note
                it is compared with ``== None``, so pass a list/tuple rather
                than an ndarray.

        Returns:
            DataFrame with one row per timestep holding copies of the bar/hat
            states plus label/input metadata.
        """
        output = {"training": [], "label": [], "I": [], "L": [], "timestep": [],
                  "r1_bar": [], "r2_bar": [], "r2_bar_x": [],
                  "r1_hat": [], "r2_hat": [], "r2_hat_x": []}
        
        self.r1_hat_states = []
        self.r2_hat_states = []
        
        s10 = self.s10
        s11 = self.s11
        s21 = self.s21
        s22 = self.s22
        s32 = self.s32
        
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2

        # MLi Method (zeros is mode of sparse kurtotic, but this isn't actually sk)
        r1_hat = np.zeros(self.r1_size)

        # # Gaussian instead
        # r1_hat = np.random.normal(size=(self.r1_size,))

        if r2_priming == None:
            # MLi Method (zeros is mode of sparse kurtotic, but this isn't actually sk)
            r2_hat = np.zeros(self.r2_size)
            r2_hat_x = np.zeros(self.r2_size)

            # # Gaussian instead
            # r2_hat = np.random.normal(size=(self.r2_size,))
            # r2_hat_x = np.random.normal(size=(self.r2_size,))

        else:
            r2_hat = np.array(r2_priming)
            r2_hat_x = np.array(r2_priming)
        
        U1_hat = self.U1.copy()
        U2_hat = self.U2.copy()
        
        V1_hat = self.V1.copy()
        V2_hat = self.V2.copy()

        for idx in np.arange(I.shape[1]):
            r1_hat_old = r1_hat.copy()
            r2_hat_old = r2_hat.copy()
            r2_hat_x_old = r2_hat_x.copy()

            U1_bar = U1_hat.copy()
            U2_bar = U2_hat.copy()

            V1_bar = V1_hat.copy()
            V2_bar = V2_hat.copy()
            
            # bar states: temporal predictions of current states from previous hat states via V
            r1_bar = f(V1_bar @ r1_hat_old, af=af)
            r2_bar = f(V2_bar @ r2_hat_old, af=af)
            r2_bar_x = f(V2_bar @ r2_hat_x_old, af=af)

            # hat states: bar states corrected by prediction errors; r2_hat_x adds the classification term
            r1_hat = r1_bar + alpha_1/s10 * U1_bar.T @ f_prime(U1_bar @ r1_bar, af=af) @ (I[:, idx] - f(U1_bar @ r1_bar, af=af)) - alpha_1/s21 * (r1_bar - f(U2_bar @ r2_bar, af=af))
            r2_hat = r2_bar + alpha_2/s21 * U2_bar.T @ f_prime(U2_bar @ r2_bar, af=af) @ (r1_bar - f(U2_bar @ r2_bar, af=af))
            r2_hat_x = r2_hat - 1/2 * alpha_2/s32 * (softmax(r2_bar) - L)

            if training == True:
                U1_hat = U1_bar + beta_1/s10 * f_prime(U1_bar @ r1_hat, af=af) @ np.outer(I[:, idx] - f(U1_bar @ r1_hat, af=af), r1_hat)
                U2_hat = U2_bar + beta_2/s21 * f_prime(U2_bar @ r2_hat_x, af=af) @ np.outer(r1_bar - f(U2_bar @ r2_hat_x, af=af), r2_hat_x)

                V1_hat = V1_bar + gamma_1/s11 * f_prime(V1_bar @ r1_hat_old, af=af) @ np.outer(r1_hat - f(V1_bar @ r1_hat_old, af=af), r1_hat_old)
                V2_hat = V2_bar + gamma_2/s22 * f_prime(V2_bar @ r2_hat_x_old, af=af) @ np.outer(r2_hat_x - f(V2_bar @ r2_hat_x_old, af=af), r2_hat_x_old)
            
            output["training"].append(training)
            output["label"].append(label)
            output["I"].append(I[:, idx])
            output["L"].append(L)
            output["timestep"].append(idx)

            output["r1_bar"].append(r1_bar.copy())
            output["r2_bar"].append(r2_bar.copy())
            output["r2_bar_x"].append(r2_bar_x.copy())
            
            output["r1_hat"].append(r1_hat.copy())
            output["r2_hat"].append(r2_hat.copy())
            output["r2_hat_x"].append(r2_hat_x.copy())
            
            self.r1_hat_states.append(r1_hat)
            self.r2_hat_states.append(r2_hat)
            
        self.U1 = U1_hat.copy()
        self.U2 = U2_hat.copy()
        
        self.V1 = V1_hat.copy()
        self.V2 = V2_hat.copy()
        
        self.r1_hat_states = np.array(self.r1_hat_states)
        self.r2_hat_states = np.array(self.r2_hat_states)

        self.label = label
        
        output_df = pd.DataFrame.from_dict(output)
        return output_df
    
    def save(self, dir_name: str) -> None:
        """Save all weight matrices to ``dir_name``/model.npz."""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            V1=self.V1,
                            V2=self.V2)
        print("saved: {}".format(dir_name))

    def load(self, dir_name: str) -> None:
        """Load previously saved weight matrices from ``dir_name``/model.npz."""
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        
        data = np.load(file_path)
        self.U1 = data["U1"]
        self.U2 = data["U2"]
        self.V1 = data["V1"]
        self.V2 = data["V2"]
        print("loaded: {}".format(dir_name))
