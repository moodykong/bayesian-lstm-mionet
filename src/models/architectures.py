#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from typing import Any
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


# LSTM-MIONet Static
class LSTM_MIONet_Static(nn.Module):
    def __init__(self):
        super(LSTM_MIONet_Static, self).__init__()
        d_latent = 150
        d_rnn = 10
        self.layernorm = nn.LayerNorm([d_latent])
        self.layernorm_x_n = nn.LayerNorm([d_latent])
        self.layernorm_delta_t = nn.LayerNorm([d_latent])
        self.lstm = nn.LSTM(d_latent, d_rnn, 2, batch_first=True, dropout=0.0)

        input_mlp = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        cell_mlp = nn.Sequential(
            nn.Linear(d_rnn, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        hidden_mlp = nn.Sequential(
            nn.Linear(d_rnn, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        x_n_mlp = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        delta_t_mlp = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        self.cell_mlp = cell_mlp

        self.input_mlp = input_mlp
        self.delta_t_mlp = delta_t_mlp
        self.x_n_mlp = x_n_mlp
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        input_data, x_n, t_params = x
        x = input_data
        x_n = x_n.unsqueeze(1)
        h = t_params[:, [1]]

        latent_x = self.input_mlp(x)
        latent_x_n = self.x_n_mlp(x_n)

        delta_t = h
        delta_t = delta_t.unsqueeze(1)

        # normalize
        latent_x = self.layernorm(latent_x)

        # define the mask for the padded sequence
        mask = (x != 0.0).type(torch.bool)
        mask_len = mask.sum(axis=(-1, -2)).type(torch.int64)
        latent_x_packed = (
            pack_padded_sequence(
                latent_x, mask_len.cpu(), batch_first=True, enforce_sorted=False
            )
            if mask_len is not None
            else latent_x
        )

        output, (h_n, c_n) = self.lstm(latent_x_packed)
        memory = self.cell_mlp(c_n[-1]).unsqueeze(1)

        latent_delta_t = self.delta_t_mlp(delta_t)
        latent_delta_t = self.layernorm_delta_t(latent_delta_t)
        latent_x_n = self.layernorm_x_n(latent_x_n)
        pred = (memory * latent_x_n * latent_delta_t).sum(dim=-1).view(
            -1, 1
        ) + self.bias

        return pred


# LSTM-MIONet
class LSTM_MIONet(nn.Module):
    def __init__(
        self, branch_1: dict, branch_2: dict, trunk: dict, use_bias: bool = True
    ) -> None:
        super(LSTM_MIONet, self).__init__()

        self.branch_1 = MLP(
            in_features=branch_1["state_feature_num"],
            layer_size=branch_1["layer_size_list"],
            activation=branch_1["activation"],
        )
        self.branch_2 = LSTM_MLP(
            layer_size=branch_2["layer_size_list"],
            lstm_size=branch_2["lstm_size"],
            lstm_layer=branch_2["lstm_layer_num"],
            activation=branch_2["activation"],
        )
        self.trunk = MLP(
            in_features=1,
            layer_size=trunk["layer_size_list"],
            activation=trunk["activation"],
        )

        self.use_bias = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: list) -> list:
        input_data, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch_1(x_n) * self.branch_2(input_data)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next


# Vanilla DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super(DeepONet, self).__init__()

        self.branch = MLP(
            in_features=branch["state_feature_num"],
            layer_size=branch["layer_size_list"],
            activation=branch["activation"],
        )
        self.trunk = MLP(
            in_features=1,
            layer_size=trunk["layer_size_list"],
            activation=trunk["activation"],
        )

        self.use_bias = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: list) -> list:
        _, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch(x_n)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next


# Local DeepONet
class DeepONet_Local(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super(DeepONet_Local, self).__init__()

        self.branch_1 = MLP(
            in_features=branch["state_feature_num"],
            layer_size=branch["layer_size_list"],
            activation=branch["activation"],
        )
        self.branch_2 = MLP(
            in_features=branch["state_feature_num"],
            layer_size=branch["layer_size_list"],
            activation=branch["activation"],
        )
        self.trunk = MLP(
            in_features=1,
            layer_size=trunk["layer_size_list"],
            activation=trunk["activation"],
        )

        self.use_bias = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: list) -> list:
        input_data, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch_1(x_n) * self.branch_2(input_data)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next


# MLP
class MLP(nn.Module):
    def __init__(self, in_features: int, layer_size: list, activation: str) -> None:
        super(MLP, self).__init__()
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


# LSTM-MLP
class LSTM_MLP(nn.Module):
    def __init__(
        self, layer_size: list, lstm_size: int, lstm_layer: int, activation: str
    ) -> None:
        super(LSTM_MLP, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net_1)):
            y = self.net_1[k](y)
        # Define the mask for the padded sequence
        mask = (x != 0).type(torch.bool)
        mask_len = mask.sum(axis=(-1, -2)).type(torch.int64)
        y = pack_padded_sequence(
            y, mask_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(y)
        y = self.net_2[0](h_n[-1])
        for k in range(1, len(self.net_2)):
            y = self.net_2[k](y)
        return y


# get activation function from str
def get_activation(identifier: str) -> Any:
    """get activation function."""
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "sigmoid": nn.Sigmoid(),
        "leaky": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
        "Rrelu": nn.RReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "Mish": nn.Mish(),
    }[identifier]
