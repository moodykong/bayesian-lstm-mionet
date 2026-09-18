"""The vanilla LSTM baseline of Section 5.1 of the paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from blstm_mionet.models.layers import LSTM_MLP, MLP


class LSTM_DeepONet(nn.Module):
    """LSTM branch x time trunk: no explicit state branch.

    The current state has to be inferred from the input function history alone,
    which is the ablation the paper compares LSTM-MIONet against.
    """

    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super().__init__()

        self.branch = LSTM_MLP(
            layer_size=branch["layer_size_list"],
            lstm_size=branch["lstm_size"],
            lstm_layer=branch["lstm_layer_num"],
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

    def forward(self, x: list) -> torch.Tensor:
        input_data, _, t_params = x
        h = t_params[:, [1]]
        B = self.branch(input_data)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next
