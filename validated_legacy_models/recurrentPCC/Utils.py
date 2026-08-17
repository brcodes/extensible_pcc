"""Shared helpers for the legacy recurrent PCC analysis scripts.

Provides plot reference-line helpers, trained-model pickle loading,
bootstrap confidence intervals, and sparsity metrics. This is part of the
validated legacy reference implementation.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np


def add_vertical_timestep_lines(ax: Any, x_positions: Iterable, *, color: str = "0.75", alpha: float = 0.75,
                                 linestyle: str = "--", linewidth: float = 1.75,
                                 highlight_special: Optional[Sequence] = None,
                                 special_color: str = "0.35", special_linestyle: str = "-.") -> None:
    """Draw vertical reference lines at given x positions on a Matplotlib Axes.

    Optionally highlight specific positions (e.g., timestep 40) with a distinct style.
    """
    highlight = set(highlight_special or [])
    for t in x_positions:
        is_special = t in highlight
        ax.axvline(
            x=t,
            color=special_color if is_special else color,
            alpha=alpha,
            linestyle=special_linestyle if is_special else linestyle,
            linewidth=linewidth,
        )


def load_trained_namespace(pkl_path: str) -> dict:
    """Load a trained_rpcc_*.pkl into a plain dict namespace with param entries expanded."""
    results_dict = pickle.load(open(pkl_path, "rb"))
    ns = dict(results_dict)
    param_df = ns.get("param", None)
    if param_df is None:
        raise KeyError(f"Expected 'param' in {pkl_path} results_dict, but it was missing.")
    for key in param_df.index:
        ns[key] = param_df[key]
    return ns


def compute_bootstrap_ci(data_arr: np.ndarray, stat: str = "mean", nboot: int = 10000, ci: float = 0.95,
                         return_stat: bool = False) -> list[float] | tuple[list[float], float]:
    """Compute bootstrap confidence interval for mean/std/median of 1D data.

    Returns:
        ``[lower, upper]``, or ``([lower, upper], stat)`` if ``return_stat``.
    """
    if stat == "mean":
        stat_func: Callable = np.mean
    elif stat == "std":
        stat_func = np.std
    elif stat == "median":
        stat_func = np.median
    else:
        raise ValueError("stat must be one of: mean, std, median")

    if return_stat:
        bootstrap_ci, orig_stat = bootstrap(data_arr, stat_func, confidence_level=ci, n_resamples=nboot,
                                            random_state=123, return_stat=True)
        lower = bootstrap_ci[0]
        upper = bootstrap_ci[1]
        return [lower, upper], orig_stat

    bootstrap_ci = bootstrap(data_arr, stat_func, confidence_level=ci, n_resamples=nboot,
                             random_state=123, return_stat=False)
    lower = bootstrap_ci[0]
    upper = bootstrap_ci[1]
    return [lower, upper]


def bootstrap(data_array: np.ndarray, stat_func: Callable, confidence_level: float = 0.95, n_resamples: int = 1000,
             random_state: Optional[int] = None,
             return_stat: bool = False) -> list[float] | tuple[list[float], float]:
    """Compute a percentile bootstrap confidence interval for a statistic."""
    if random_state is not None:
        np.random.seed(random_state)

    sample_size = len(data_array)
    resampled_stats = []

    for _ in range(n_resamples):
        resample = np.random.choice(data_array, size=sample_size, replace=True)
        resampled_stat = stat_func(resample)
        resampled_stats.append(resampled_stat)

    alpha = (1 - confidence_level)
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(resampled_stats, lower_percentile)
    upper_bound = np.percentile(resampled_stats, upper_percentile)

    if return_stat:
        orig_stat = stat_func(data_array)
        return [lower_bound, upper_bound], orig_stat

    return [lower_bound, upper_bound]


def inv_part_ratio(v: np.ndarray) -> float:
    """Compute the inverse participation ratio of vector ``v`` (normalized first)."""
    vnorm = np.sqrt(np.dot(v, v))
    v2 = np.power(v / vnorm, 2)
    v4 = np.power(v / vnorm, 4)
    return np.power(v2.sum(), 2) / v4.sum()


def sparsity(v: np.ndarray) -> float:
    """Compute the sparsity metric from Vinje and Gallant (2000)."""
    n = len(v)
    return (1.0 - (v.mean() ** 2) / ((v ** 2).mean())) / (1.0 - 1.0 / n)
