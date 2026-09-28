"""Adam training, replica exchange SGLD and the MLflow tracking helpers."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from mlflow.exceptions import MlflowException

from blstm_mionet.config import LangevinConfig, TrainConfig
from blstm_mionet.models import build_model
from blstm_mionet.training import tracking
from blstm_mionet.training.resgld import langevin_step
from blstm_mionet.training.trainer import train_adam
from conftest import (
    execute_training,
    prepare_training_dataset,
    tiny_bayesian_config,
    tiny_model_config,
    tiny_train_config,
)


def _metric_history(run_id: str, key: str) -> list:
    """Distinct ``(step, value)`` points logged under ``key``.

    MLflow 3 re-associates the run metrics with every model logged afterwards,
    so ``get_metric_history`` returns the same point several times.
    """
    history = mlflow.tracking.MlflowClient().get_metric_history(run_id, key)
    return sorted({(point.step, point.value) for point in history})


def _artifact_paths(run_id: str, path: str | None = None) -> list[str]:
    return [
        item.path
        for item in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=path)
    ]


# --------------------------------------------------------------------------- #
# train_adam
# --------------------------------------------------------------------------- #
def test_train_adam_logs_train_loss(adam_run) -> None:
    history = _metric_history(adam_run["run_id"], "train_loss")
    assert len(history) == 2  # one point per epoch
    assert [step for step, _ in history] == [0, 1]
    assert all(np.isfinite(value) for _, value in history)


def test_train_adam_logs_val_loss(adam_run) -> None:
    history = _metric_history(adam_run["run_id"], "val_loss")
    assert [step for step, _ in history] == [0, 1]
    assert all(np.isfinite(value) for _, value in history)


def test_train_adam_returns_its_history(adam_run) -> None:
    history = adam_run["history"]
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert np.isfinite(history["best_metric"]["train_loss"])
    assert np.isfinite(history["best_metric"]["val_loss"])
    assert history["best_epoch"] in (0, 1)
    assert history["model_uri"]


def test_train_adam_logs_a_loadable_model(adam_run, cpu_device) -> None:
    model = tracking.load_model(adam_run["model_uri"], cpu_device)
    assert isinstance(model, torch.nn.Module)
    assert type(model).__name__ == "LSTM_MIONet"

    # The reloaded model reproduces the trained weights.
    reference = adam_run["model"]
    for (name, left), (_, right) in zip(
        model.state_dict().items(), reference.state_dict().items(), strict=True
    ):
        assert torch.allclose(left, right), name


def test_train_adam_logged_model_predicts(adam_run, cpu_device) -> None:
    model = tracking.load_model(adam_run["model_uri"], cpu_device)
    dataset = adam_run["dataset"]
    inputs = [dataset.input_data[:4], dataset.x_n[:4], dataset.t_params[:4]]
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
    assert prediction.shape == (4, 1)
    assert torch.isfinite(prediction).all()


def test_train_adam_can_skip_saving_the_model(
    shared_tracking, lorentz_dataset: Path
) -> None:
    result = execute_training(
        tiny_train_config(
            lorentz_dataset,
            epochs=1,
            save_model=False,
            monitor_metric="train_loss",
            run_name="no_save",
        ),
        tiny_model_config("LSTM_MIONet"),
    )
    assert result["history"]["model_uri"] is None
    assert result["history"]["model_uris"] == []
    with pytest.raises(MlflowException):
        tracking.load_model(f"runs:/{result['run_id']}/model", torch.device("cpu"))
    assert len(_metric_history(result["run_id"], "train_loss")) == 1


# --------------------------------------------------------------------------- #
# train_resgld
# --------------------------------------------------------------------------- #
def test_train_resgld_logs_the_ensemble(resgld_run) -> None:
    members = _artifact_paths(resgld_run["run_id"], tracking.ENSEMBLE_ARTIFACT_PATH)
    assert members == ["ensemble/member_0000.pt", "ensemble/member_0001.pt"]
    assert resgld_run["history"]["n_ensemble"] == 2


def test_train_resgld_logs_the_model_and_checkpoint(resgld_run, cpu_device) -> None:
    # MLflow 3 stores logged models outside the run artifacts; the exploit model
    # is still reachable through the run scoped URI.
    assert "checkpoints" in _artifact_paths(resgld_run["run_id"])
    assert _artifact_paths(resgld_run["run_id"], "checkpoints") == [
        "checkpoints/best_replica_model.pt"
    ]
    model = tracking.load_model(resgld_run["model_uri"], cpu_device)
    assert type(model).__name__ == "LSTM_MIONet"


def test_train_resgld_logs_the_swap_diagnostics(resgld_run) -> None:
    run_id = resgld_run["run_id"]
    for key in ("ge_exploit", "ge_explore", "swap_probability", "swap"):
        history = _metric_history(run_id, key)
        assert [step for step, _ in history] == [0, 1, 2, 3, 4]  # one per epoch
        assert all(np.isfinite(value) for _, value in history)
    probabilities = _metric_history(run_id, "swap_probability")
    assert all(0.0 <= value <= 1.0 for _, value in probabilities)

    assert _metric_history(run_id, "n_ensemble_members") == [(0, 2.0)]


def test_train_resgld_history(resgld_run) -> None:
    history = resgld_run["history"]
    assert len(history["ge exploit"]) == 5
    assert len(history["ge explore"]) == 5
    assert len(history["switches"]) == 5
    assert np.isfinite(history["best_ge"])
    assert history["model_uri"]


def test_resgld_members_load_into_the_model(resgld_run, cpu_device) -> None:
    local_dir = tracking.download_artifacts(
        resgld_run["run_id"], tracking.ENSEMBLE_ARTIFACT_PATH
    )
    member_paths = sorted(Path(local_dir).glob("member_*.pt"))
    assert len(member_paths) == 2

    model = tracking.load_model(resgld_run["model_uri"], cpu_device)
    for path in member_paths:
        checkpoint = torch.load(path, map_location=cpu_device, weights_only=False)
        assert "state_dict" in checkpoint
        model.load_state_dict(checkpoint["state_dict"])  # must not raise


def _single_parameter_net(n: int, grad: float) -> torch.nn.Module:
    net = torch.nn.Linear(n, 1, bias=False)
    torch.nn.init.zeros_(net.weight)
    net.weight.grad = torch.full_like(net.weight, grad)
    return net


def test_langevin_step_deterministic_part() -> None:
    """With zero temperature the update is plain momentum SGD on ``grad / sigma``."""
    chain = LangevinConfig(tau=0.0, eta=0.1, alpha=0.25, v=0.1)
    assert chain.scale == 0.0
    net = _single_parameter_net(5, grad=2.0)
    velocity = [torch.full_like(net.weight, 1.0)]

    langevin_step(net, velocity, chain, sigma=4.0)

    # v = (1 - 0.25) * 1.0 - 0.1 * 2.0 / 4.0 = 0.7; theta = 0 + v
    assert torch.allclose(velocity[0], torch.full_like(net.weight, 0.7))
    assert torch.allclose(net.weight.detach(), torch.full_like(net.weight, 0.7))


def test_langevin_step_noise_is_independent_per_entry() -> None:
    """Every parameter entry receives its own N(0, scale^2) kick."""
    chain = LangevinConfig(tau=1.0, eta=1e-2, alpha=0.5, v=0.1)
    net = _single_parameter_net(20_000, grad=0.0)
    velocity = [torch.zeros_like(net.weight)]

    torch.manual_seed(0)
    langevin_step(net, velocity, chain, sigma=1.0)

    kicks = net.weight.detach().flatten()
    # A single scalar per tensor would make every entry identical.
    assert kicks.unique().numel() > 1000
    assert kicks.mean().abs() < 5 * chain.scale / np.sqrt(kicks.numel())
    assert kicks.std().item() == pytest.approx(chain.scale, rel=0.05)


def test_train_adam_fails_loudly_when_resume_model_is_missing(
    shared_tracking, lorentz_dataset: Path
) -> None:
    """A warm start that cannot be loaded must not silently train from scratch."""
    config: TrainConfig = tiny_train_config(
        lorentz_dataset, epochs=1, resume_model="runs:/0123456789abcdef/model"
    )
    model_config = tiny_model_config("LSTM_MIONet")
    dataset, _, state_feature_num = prepare_training_dataset(config, model_config)
    model = build_model(model_config, state_feature_num)
    with tracking.start_run("blstm_mionet_tests", "bad_resume"):
        with pytest.raises(MlflowException):
            train_adam(config, model, dataset, torch.device("cpu"))


def test_resgld_burn_in_controls_the_member_count() -> None:
    bayesian = tiny_bayesian_config(n_ensemble=2)
    # burn_in = epochs - (n_ensemble + 1); members are collected for epoch > burn_in.
    assert bayesian.burn_in(5) == 2
    assert len([e for e in range(5) if e > bayesian.burn_in(5)]) == 2


# --------------------------------------------------------------------------- #
# tracking helpers
# --------------------------------------------------------------------------- #
def test_configure_tracking_prefers_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "from_env"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(target))
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    assert tracking.configure_tracking("ignored") == str(target)
    assert mlflow.get_tracking_uri() == str(target)
    # A file store needs this flag on recent MLflow releases.
    import os as _os

    assert _os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_configure_tracking_falls_back_to_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    target = tmp_path / "from_config"
    assert tracking.configure_tracking(str(target)) == str(target)
    assert mlflow.get_tracking_uri() == str(target)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("runs:/abc123", "abc123"),
        ("runs:/abc123/model", "abc123"),
        ("runs:/abc123/ensemble/member_0000.pt", "abc123"),
        ("abc123", "abc123"),
        ("  abc123  ", "abc123"),
    ],
)
def test_run_id_from_uri(uri: str, expected: str) -> None:
    assert tracking.run_id_from_uri(uri) == expected


def test_log_source_snapshot_and_state_dict(shared_tracking, cpu_device) -> None:
    model = torch.nn.Linear(2, 1)
    with tracking.start_run("blstm_mionet_tests", "snapshot") as active_run:
        run_id = active_run.info.run_id
        tracking.log_source_snapshot()
        tracking.log_state_dict(model.state_dict(), "weights.pt", "checkpoints")

    assert "source" in _artifact_paths(run_id)
    assert "source/blstm_mionet" in _artifact_paths(run_id, "source")
    package_files = _artifact_paths(run_id, "source/blstm_mionet")
    assert "source/blstm_mionet/config.py" in package_files
    assert not any("__pycache__" in path for path in package_files)

    assert _artifact_paths(run_id, "checkpoints") == ["checkpoints/weights.pt"]
    tags = mlflow.tracking.MlflowClient().get_run(run_id).data.tags
    assert tags["blstm_mionet.version"] == "1.0.0"


def test_log_model_unwraps_data_parallel(shared_tracking, cpu_device) -> None:
    inner = torch.nn.Linear(3, 1)
    wrapped = torch.nn.DataParallel(inner)
    with tracking.start_run("blstm_mionet_tests", "data_parallel") as active_run:
        run_id = active_run.info.run_id
        tracking.log_model(wrapped, "model")
    loaded = tracking.load_model(f"runs:/{run_id}/model", cpu_device)
    assert isinstance(loaded, torch.nn.Linear)
    assert torch.allclose(loaded.weight, inner.weight)
