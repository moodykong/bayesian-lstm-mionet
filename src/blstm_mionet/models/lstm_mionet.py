"""The LSTM-MIONet operator of the paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from blstm_mionet.models.layers import LSTM_MLP, MLP


class LSTM_MIONet(nn.Module):
    """Multiple-input operator: state branch x memory branch x time trunk.

    ``branch_1`` encodes the current state ``x_n``, ``branch_2`` (an LSTM)
    encodes the length-variant input function and the trunk encodes the time
    spacing ``h``.
    """

    def __init__(
        self, branch_1: dict, branch_2: dict, trunk: dict, use_bias: bool = True
    ) -> None:
        super().__init__()

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

    def forward(self, x: list) -> torch.Tensor:
        input_data, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch_1(x_n) * self.branch_2(input_data)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next
