"""Model registry.

Every architecture is paired with the dataset preparation routine it expects,
which replaces the ``match config["architecture"]`` blocks that used to be
duplicated in ``train.py``, ``infer.py`` and ``infer_ensemble.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch.nn as nn

from blstm_mionet.config import ModelConfig
from blstm_mionet.data.masking import (
    prepare_future_local_predict_dataset,
    prepare_local_predict_dataset,
)
from blstm_mionet.models.baselines import LSTM_DeepONet
from blstm_mionet.models.deeponet import DeepONet, DeepONet_Local
from blstm_mionet.models.layers import LSTM_MLP, MLP, ReLUSin, Sin, get_activation
from blstm_mionet.models.lstm_mionet import LSTM_MIONet

__all__ = [
    "ARCHITECTURES",
    "DeepONet",
    "DeepONet_Local",
    "LSTM_DeepONet",
    "LSTM_MIONet",
    "LSTM_MLP",
    "MLP",
    "ReLUSin",
    "Sin",
    "build_model",
    "dataset_preparer_for",
    "get_activation",
]


class Architecture(NamedTuple):
    """One entry of the registry."""

    #: the ``nn.Module`` subclass
    cls: type[nn.Module]
    #: which branches the constructor takes, in order
    branches: tuple[str, ...]
    #: the dataset preparation routine the architecture is trained on
    preparer: Callable[..., tuple]


ARCHITECTURES: dict[str, Architecture] = {
    "LSTM_MIONet": Architecture(
        LSTM_MIONet, ("branch_state", "branch_memory"), prepare_local_predict_dataset
    ),
    "LSTM_DeepONet": Architecture(
        LSTM_DeepONet, ("branch_memory",), prepare_local_predict_dataset
    ),
    "DeepONet": Architecture(
        DeepONet, ("branch_state",), prepare_local_predict_dataset
    ),
    "DeepONet_Local": Architecture(
        DeepONet_Local, ("branch_state",), prepare_future_local_predict_dataset
    ),
}


def dataset_preparer_for(architecture: str) -> Callable[..., tuple]:
    """Return the dataset preparation routine required by ``architecture``."""
    return _lookup(architecture).preparer


def build_model(config: ModelConfig, state_feature_num: int) -> nn.Module:
    """Instantiate the architecture named by ``config`` .

    ``state_feature_num`` is the number of state features produced by the
    dataset preparation step and therefore known only at run time.
    """
    entry = _lookup(config.architecture)

    branch_state = {
        "layer_size_list": [config.branch_state.width] * config.branch_state.depth,
        "activation": config.branch_state.activation,
        "state_feature_num": state_feature_num,
    }
    branch_memory = {
        "layer_size_list": [config.branch_memory.width] * config.branch_memory.depth,
        "lstm_size": config.branch_memory.lstm_size,
        "lstm_layer_num": config.branch_memory.lstm_layer_num,
        "activation": config.branch_memory.activation,
    }
    trunk = {
        "layer_size_list": [config.trunk.width] * config.trunk.depth,
        "activation": config.trunk.activation,
    }

    branches = {"branch_state": branch_state, "branch_memory": branch_memory}
    args = [branches[name] for name in entry.branches]
    return entry.cls(*args, trunk, use_bias=config.use_bias)


def _lookup(architecture: str) -> Architecture:
    try:
        return ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(
            f"invalid architecture {architecture!r}; "
            f"available: {sorted(ARCHITECTURES)}"
        ) from exc
