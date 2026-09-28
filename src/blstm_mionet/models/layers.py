"""Shared building blocks: MLPs, the LSTM encoder and activation functions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Sin(nn.Module):
    """Sine activation (formerly ``utils.torch_utils.sin_act``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ReLUSin(nn.Module):
    """Rectified sine activation (formerly ``utils.torch_utils.Rsin``).

    The original implementation had an unreachable ``return torch.sin(x)``
    after the rectified branch; it has been removed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), torch.sin(x))


def get_activation(identifier: str) -> nn.Module:
    """Return a fresh activation module for ``identifier``."""
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "leaky": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "Rrelu": nn.RReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "Mish": nn.Mish,
        "sin": Sin,
        "relu_sin": ReLUSin,
    }
    try:
        return activations[identifier]()
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {identifier!r}; available: {sorted(activations)}"
        ) from exc


class MLP(nn.Module):
    """Fully connected network with a layer norm before the output layer."""

    def __init__(
        self, in_features: int, layer_size: list[int], activation: str
    ) -> None:
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(in_features, layer_size[0], bias=True))
        for k in range(len(layer_size) - 2):
            self.net.append(nn.Linear(layer_size[k], layer_size[k + 1], bias=True))
            self.net.append(get_activation(activation))
        self.net.append(nn.LayerNorm([layer_size[-2]]))
        self.net.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
        self.net.apply(self._init_weights)

    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net)):
            y = self.net[k](y)
        return y


class LSTM_MLP(nn.Module):
    """Encoder of a length-variant input function.

    A per-time-step MLP lifts the scalar input, an LSTM consumes the packed
    (zero-padded) sequence and a second MLP maps the last hidden state to the
    branch output.  Histories are right-padded with zeros (paper Sec. 3.4): the
    length of a sequence is the position of its last non-zero step, so an exact
    zero *inside* a history is kept as data.  An all-zero history is invalid.
    """

    def __init__(
        self, layer_size: list[int], lstm_size: int, lstm_layer: int, activation: str
    ) -> None:
        super().__init__()
        self.net_1 = nn.ModuleList()
        self.net_1.append(nn.Linear(1, layer_size[0], bias=True))
        for k in range(0, len(layer_size) - 2):
            self.net_1.append(nn.Linear(layer_size[k], layer_size[k + 1], bias=True))
            self.net_1.append(get_activation(activation))
        self.net_1.append(nn.LayerNorm([layer_size[-2]]))
        self.net_1.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))

        self.lstm = nn.LSTM(
            layer_size[-1], lstm_size, lstm_layer, batch_first=True, dropout=0.0
        )
        self.net_2 = nn.ModuleList()
        self.net_2.append(nn.Linear(lstm_size, layer_size[0], bias=True))
        for k in range(0, len(layer_size) - 2):
            self.net_2.append(nn.Linear(layer_size[k], layer_size[k + 1], bias=True))
            self.net_2.append(get_activation(activation))
        self.net_2.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))

    @staticmethod
    def sequence_lengths(x: torch.Tensor) -> torch.Tensor:
        """Length of every zero-padded history in ``x`` (``[batch, time, 1]``)."""
        present = (x != 0).any(dim=-1)  # [batch, time]
        steps = torch.arange(1, x.shape[1] + 1, device=x.device)
        return (present * steps).amax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net_1)):
            y = self.net_1[k](y)
        y = pack_padded_sequence(
            y, self.sequence_lengths(x).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(y)
        y = self.net_2[0](h_n[-1])
        for k in range(1, len(self.net_2)):
            y = self.net_2[k](y)
        return y
