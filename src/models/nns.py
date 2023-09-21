#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from typing import Any
from torch.nn.utils.rnn import pack_padded_sequence


# LSTM-MIONet
class LSTM_MIONet(nn.Module):
    def __init__(self, branch_1: dict, branch_2: dict, trunk: dict, use_bias: bool=True) -> None:
        super(LSTM_MIONet, self).__init__()

        self.branch_1 = MLP(branch_1["layer_size_list"], branch_1["activation"])
        self.branch_2 = LSTM_MLP(branch_2["layer_size_list"],branch_2["lstm_size"],branch_2["lstm_layer_num"], branch_2["activation"])
        self.trunk = MLP(trunk["layer_size_list"], trunk["activation"])

        self.use_bias  = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True) 

    def forward(self, x: list) -> list:
        input_data, x_tn, t_local = x
        h = t_local[:, 1]
        B = torch.einsum("bi, bi -> b", self.branch_1(x_tn), self.branch_2(input_data))
        T = self.trunk(h)
        
        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)
        
        return pred_x_next + self.tau if self.use_bias else pred_x_next

# MLP
class MLP(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        for k in range(len(layer_size) - 2):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
            self.net.append(get_activation(activation))
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
    def __init__(self, layer_size: list, lstm_size: int, lstm_layer: int, activation: str) -> None:
        super(LSTM_MLP, self).__init__()
        self.net_1 = nn.ModuleList()
        for k in range(1, len(layer_size) - 2):
            self.net_1.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
            self.net_1.append(get_activation(activation))
        self.net_1.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
        self.lstm = nn.LSTM(layer_size[-1], lstm_size, lstm_layer, batch_first=True, dropout=0.)
        self.net_2 = nn.ModuleList()
        self.net_2.append(nn.Linear(lstm_size, layer_size[0], bias=True))
        for k in range(1, len(layer_size) - 2):
            self.net_2.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
            self.net_2.append(get_activation(activation))
        self.net_2.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
        self.layernorm=nn.LayerNorm([layer_size[-1]])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net_1)):
            y = self.net_1[k](y)
        y = y.unsqueeze(1)
        # Normalize
        y = self.layernorm(y)
        # Define the mask for the padded sequence
        mask = x.sum(axis=-1) != 0
        mask_len = mask.sum(axis=-1).type(torch.int64).cpu()
        y = pack_padded_sequence(y, mask_len, batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(y)
        y = self.net_2[0](h_n[-1])
        for k in range(1, len(self.net_2)):
            y = self.net_2[k](y)
        return y


# get activation function from str
def get_activation(identifier: str) -> Any:
    """get activation function."""
    return{
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sin": sin_act(),
            "rsin": Rsin(),
            "softplus": nn.Softplus(),
            "Rrelu": nn.RReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "Mish": nn.Mish(),
    }[identifier]

