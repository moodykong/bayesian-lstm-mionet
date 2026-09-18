"""The architecture registry, the layers and the forward passes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from blstm_mionet.config import ARCHITECTURE_CHOICES
from blstm_mionet.data.masking import (
    prepare_future_local_predict_dataset,
    prepare_local_predict_dataset,
)
from blstm_mionet.models import (
    ARCHITECTURES,
    LSTM_MLP,
    MLP,
    DeepONet,
    DeepONet_Local,
    LSTM_DeepONet,
    LSTM_MIONet,
    ReLUSin,
    Sin,
    build_model,
    dataset_preparer_for,
    get_activation,
)
from conftest import tiny_model_config

BATCH_SIZE = 4


@pytest.fixture(params=sorted(ARCHITECTURES), ids=sorted(ARCHITECTURES))
def architecture(request) -> str:
    return request.param


@pytest.fixture
def batch(architecture: str, pendulum_dataset: Path, make_torch_dataset):
    """One collated batch prepared exactly like training does."""
    dataset, _, state_feature_num = make_torch_dataset(
        pendulum_dataset, architecture=architecture, search_num=3, search_len=2
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    inputs, targets = next(iter(loader))
    return list(inputs), targets, state_feature_num


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_matches_the_configuration_choices() -> None:
    assert tuple(ARCHITECTURES) == ARCHITECTURE_CHOICES


def test_registry_entries() -> None:
    assert ARCHITECTURES["LSTM_MIONet"].cls is LSTM_MIONet
    assert ARCHITECTURES["LSTM_DeepONet"].cls is LSTM_DeepONet
    assert ARCHITECTURES["DeepONet"].cls is DeepONet
    assert ARCHITECTURES["DeepONet_Local"].cls is DeepONet_Local
    assert ARCHITECTURES["LSTM_MIONet"].branches == ("branch_state", "branch_memory")
    assert ARCHITECTURES["LSTM_DeepONet"].branches == ("branch_memory",)
    assert ARCHITECTURES["DeepONet"].branches == ("branch_state",)
    assert ARCHITECTURES["DeepONet_Local"].branches == ("branch_state",)


@pytest.mark.parametrize(
    ("name", "preparer"),
    [
        ("LSTM_MIONet", prepare_local_predict_dataset),
        ("LSTM_DeepONet", prepare_local_predict_dataset),
        ("DeepONet", prepare_local_predict_dataset),
        ("DeepONet_Local", prepare_future_local_predict_dataset),
    ],
)
def test_dataset_preparer_for(name: str, preparer) -> None:
    assert dataset_preparer_for(name) is preparer
    assert ARCHITECTURES[name].preparer is preparer


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid architecture 'LSTM_MIONet_Static'"):
        dataset_preparer_for("LSTM_MIONet_Static")
    with pytest.raises(ValueError) as excinfo:
        build_model(tiny_model_config("LSTM_MIONet_Static"), 1)
    assert "available: " in str(excinfo.value)
    for choice in ARCHITECTURE_CHOICES:
        assert choice in str(excinfo.value)


# --------------------------------------------------------------------------- #
# build_model / forward passes
# --------------------------------------------------------------------------- #
def test_build_every_architecture(architecture: str) -> None:
    model = build_model(tiny_model_config(architecture), state_feature_num=1)
    assert isinstance(model, ARCHITECTURES[architecture].cls)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_forward_pass_output_shape(architecture: str, batch) -> None:
    inputs, targets, state_feature_num = batch
    model = build_model(tiny_model_config(architecture), state_feature_num)
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (BATCH_SIZE, 1) == targets.shape
    assert torch.isfinite(output).all()


def test_forward_pass_is_differentiable(architecture: str, batch) -> None:
    inputs, targets, state_feature_num = batch
    model = build_model(tiny_model_config(architecture), state_feature_num)
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_use_bias_adds_the_tau_parameter(architecture: str) -> None:
    config = tiny_model_config(architecture)
    with_bias = build_model(config, 1)
    assert hasattr(with_bias, "tau")
    assert with_bias.tau.shape == (1,)

    config.use_bias = False
    without_bias = build_model(config, 1)
    assert not hasattr(without_bias, "tau")
    assert (
        sum(p.numel() for p in without_bias.parameters())
        == sum(p.numel() for p in with_bias.parameters()) - 1
    )


def test_width_and_depth_reach_the_layers() -> None:
    small = tiny_model_config("LSTM_MIONet")
    wide = tiny_model_config("LSTM_MIONet")
    wide.branch_state.width = 32
    assert sum(p.numel() for p in build_model(wide, 1).parameters()) > sum(
        p.numel() for p in build_model(small, 1).parameters()
    )


def test_state_feature_num_reaches_the_state_branch() -> None:
    model = build_model(tiny_model_config("DeepONet"), state_feature_num=3)
    first_linear = model.branch.net[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == 3


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
def test_mlp_shapes() -> None:
    mlp = MLP(in_features=3, layer_size=[8, 5], activation="relu")
    assert mlp(torch.zeros(7, 3)).shape == (7, 5)


def test_lstm_mlp_encodes_a_variable_length_sequence() -> None:
    encoder = LSTM_MLP(layer_size=[8, 6], lstm_size=4, lstm_layer=1, activation="relu")
    sequence = torch.zeros(3, 10, 1)
    # Three histories of different (non-zero) lengths; zeros are the padding.
    sequence[0, :4, 0] = torch.linspace(1.0, 2.0, 4)
    sequence[1, :7, 0] = torch.linspace(1.0, 2.0, 7)
    sequence[2, :10, 0] = torch.linspace(1.0, 2.0, 10)
    output = encoder(sequence)
    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()


def test_lstm_mlp_ignores_the_zero_padding() -> None:
    """Appending zeros to a history must not change the encoding."""
    encoder = LSTM_MLP(layer_size=[8, 6], lstm_size=4, lstm_layer=1, activation="relu")
    encoder.eval()
    short = torch.zeros(1, 6, 1)
    short[0, :3, 0] = torch.tensor([1.0, 2.0, 3.0])
    padded = torch.zeros(1, 12, 1)
    padded[0, :3, 0] = torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():
        assert torch.allclose(encoder(short), encoder(padded), atol=1e-6)


@pytest.mark.parametrize(
    "identifier",
    [
        "elu",
        "relu",
        "selu",
        "sigmoid",
        "leaky",
        "tanh",
        "softplus",
        "Rrelu",
        "gelu",
        "silu",
        "Mish",
        "sin",
        "relu_sin",
    ],
)
def test_get_activation_returns_a_fresh_module(identifier: str) -> None:
    first = get_activation(identifier)
    second = get_activation(identifier)
    assert isinstance(first, torch.nn.Module)
    assert first is not second


def test_get_activation_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation 'swish'"):
        get_activation("swish")


def test_sin_and_relu_sin() -> None:
    x = torch.tensor([-2.0, 0.0, 0.5, 3.0])
    assert torch.allclose(Sin()(x), torch.sin(x))
    assert torch.allclose(ReLUSin()(x), torch.clamp(torch.sin(x), min=0.0))
    assert torch.all(ReLUSin()(x) >= 0.0)
