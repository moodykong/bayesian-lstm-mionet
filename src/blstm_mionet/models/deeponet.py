"""DeepONet baselines that do not use the input function history."""

from __future__ import annotations

import torch
import torch.nn as nn

from blstm_mionet.models.layers import MLP


class DeepONet(nn.Module):
    """Vanilla DeepONet: state branch x time trunk (no input function)."""

    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super().__init__()

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

    def forward(self, x: list) -> torch.Tensor:
        _, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch(x_n)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next


class DeepONet_Local(nn.Module):
    """Local MIONet: current state x control at the prediction time x trunk.

    It is trained on the "future local" dataset, i.e. the control sampled at
    ``t_n + h`` rather than the past input function.
    """

    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super().__init__()

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

    def forward(self, x: list) -> torch.Tensor:
        input_data, x_n, t_params = x
        h = t_params[:, [1]]
        B = self.branch_1(x_n) * self.branch_2(input_data)
        T = self.trunk(h)

        pred_x_next = torch.einsum("bi, bi -> b", B, T)
        pred_x_next = torch.unsqueeze(pred_x_next, dim=-1)

        return pred_x_next + self.tau if self.use_bias else pred_x_next
