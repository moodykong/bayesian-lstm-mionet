"""Figures comparing reference and predicted trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_comparison(
    x_list: Sequence[np.ndarray],
    y_list: Sequence[np.ndarray],
    legend_list: Sequence[str],
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xlabel: str = "Time $t$",
    ylabel: str = "State $x(t)$",
    color_list: Sequence[str] | None = None,
    linestyle_list: Sequence[str] | None = None,
    fig_path: str | None = None,
    font_size: str = "7",
    save_fig: bool = False,
) -> None:
    """Plot reference and predicted trajectories on one axis."""
    plt.rcParams["font.size"] = font_size
    figure, axe = plt.subplots()

    for x_i, y_i, legend_i, c_i, ls_i in zip(
        x_list, y_list, legend_list, color_list, linestyle_list, strict=False
    ):
        axe.plot(
            x_i.reshape(
                -1,
            ),
            y_i.reshape(
                -1,
            ),
            lw=1.0,
            color=c_i,
            linestyle=ls_i,
            label=legend_i,
        )
    if xlim is not None:
        axe.set_xlim(xlim)
    if ylim is not None:
        axe.set_ylim(ylim)
    axe.set_xlabel(xlabel)
    axe.set_ylabel(ylabel)
    axe.legend()

    if save_fig and fig_path is not None:
        figure.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_comparison_uq(
    y_list: Sequence[np.ndarray],
    y_std: Any,
    legend_list: Sequence[str],
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xlabel: str = "Evaluation point",
    ylabel: str = "State $x(t_n + h)$",
    color_list: Sequence[str] | None = None,
    linestyle_list: Sequence[str] | None = None,
    fig_path: str | None = None,
    font_size: str = "20",
) -> None:
    """Plot the ensemble mean with its 95% confidence band.

    ``y_list`` is ``(truth, ensemble mean, ensemble sample)``; the band is
    drawn around ``y_list[1]``.
    """
    plt.rcParams["font.size"] = font_size
    figure = plt.figure()

    for y_i, legend_i, c_i, ls_i in zip(
        y_list, legend_list, color_list, linestyle_list, strict=False
    ):
        plt.plot(
            y_i.reshape(
                -1,
            ),
            lw=2.0,
            color=c_i,
            linestyle=ls_i,
            label=legend_i,
        )

    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    t = np.arange(y_list[0].shape[0])
    plt.fill(
        np.concatenate([t, t[::-1]]),
        np.concatenate(
            [y_list[1] - 1.9600 * y_std, (y_list[1] + 1.9600 * y_std)[::-1]]
        ),
        alpha=0.8,
        fc="g",
        ec="None",
        label="0.95 confidence interval",
    )
    plt.legend(prop={"size": font_size})
    if fig_path is not None:
        figure.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
