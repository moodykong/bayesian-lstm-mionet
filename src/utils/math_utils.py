#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky
from scipy.interpolate import CubicSpline

import torch
from utils.data_utils import dotdict

from typing import Any


# L2 relative error
def L2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> Any:
    return 100 * np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


# L1 relative error
def L1_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> Any:
    return 100 * np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)


def plot_comparison(
    x_list: list,
    y_list: list,
    legend_list: list,
    xlim: list = None,
    ylim: list = None,
    xlabel: str = "Position $t$",
    ylabel: str = "State $x(t)$",
    figsize: list = (14, 10),
    color_list: list = None,
    linestyle_list: list = None,
    fig_path: str = None,
    font_size: str = "7",
    save_fig: bool = False,
):
    plt.rcParams["font.size"] = font_size
    axe = plt.subplot()

    for x_i, y_i, legend_i, c_i, ls_i in zip(
        x_list, y_list, legend_list, color_list, linestyle_list
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
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_comparison_UQ(
    y_list: list,
    y_std: Any,
    legend_list: list,
    xlim: list = None,
    ylim: list = None,
    xlabel: str = "Position $x$",
    ylabel: str = "Target $f(x)$",
    figsize: list = (14, 10),
    color_list: list = None,
    linestyle_list: list = None,
    fig_path: str = None,
    font_size: str = "20",
) -> None:
    plt.rcParams["font.size"] = font_size
    plt.figure()

    for y_i, legend_i, c_i, ls_i in zip(
        y_list, legend_list, color_list, linestyle_list
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
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


# initialize nn parameters
def init_params(net: torch.nn) -> list:
    list_pars = []
    for p in net.parameters():
        temp = torch.zeros_like(p.data)
        list_pars.append(temp.cpu().detach().numpy())
    return list_pars


def grf_1d(a=0.01, nu=1):
    # Correlation function
    def rho(h, a=a, nu=nu):
        return np.exp(-((h / a) ** (2 * nu)))

    # Space discretization
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]

    # Distance matrix
    H = np.abs(x[:, np.newaxis] - x)

    # Covariance matrix
    sigma = rho(H)
    sigma += 1e-6 * np.eye(sigma.shape[0])

    # Cholesky factorization
    L = cholesky(sigma, lower=True)

    # Independent standard Gaussian random variables
    z = np.random.normal(size=x.size)

    # Gaussian random field
    y = np.dot(L, z)
    spline = CubicSpline(x, y)
    return spline


def runge_kutta(f, x, u, h):
    """f(x, u) -> next_x."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)
    next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
    return next_x


def integrate(method, f, control, x0, h, N):
    soln = dotdict()
    soln.x = []
    soln.t = []
    soln.u = []

    x = x0

    t = 0 * h
    u = control(t, x)
    soln.x.append(x)

    for n in range(1, N):
        # log previous control
        soln.t.append(t)
        soln.u.append(u)
        # compute next state
        x_next = runge_kutta(f, x, u, h)
        # log next state
        soln.x.append(x_next)
        x = x_next
        t = n * h
        u = control(t, x)

    soln.x = np.vstack(soln.x)
    soln.t = np.vstack(soln.t)
    soln.u = np.vstack(soln.u)

    return soln
