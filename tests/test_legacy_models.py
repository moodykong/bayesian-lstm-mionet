"""Models trained with the research code, and MLflow stores moved between machines.

The pretrained runs published with the paper were written by the research code,
which ran from ``src/`` and pickled its models against ``models.architectures``;
MLflow also records absolute paths in every ``meta.yaml``.  These tests rebuild
both situations and check that the package still loads such a model and
reproduces its predictions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from mlflow.exceptions import MlflowException

from blstm_mionet.cli.main import main as cli_main
from blstm_mionet.models import LSTM_MIONet
from blstm_mionet.training import tracking

RESEARCH_CODE = Path(__file__).parent / "data" / "research_code"

#: Logs one model with the research code's classes, the way its train.py did.
LOG_LEGACY_MODEL = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import mlflow, numpy as np, torch
    from models import architectures
    from blstm_mionet.training import tracking

    torch.manual_seed(0)
    state = {"layer_size_list": [16] * 3, "activation": "relu", "state_feature_num": 1}
    memory = {"layer_size_list": [16] * 3, "lstm_size": 8, "lstm_layer_num": 1,
              "activation": "relu"}
    trunk = {"layer_size_list": [16] * 3, "activation": "relu"}
    model = architectures.LSTM_MIONet(state, memory, trunk)
    history = torch.cat([torch.rand(4, 6, 1) + 0.1, torch.zeros(4, 4, 1)], dim=1)
    inputs = [history, torch.rand(4, 1), torch.rand(4, 3)]
    np.save(sys.argv[2] + "/expected.npy", model(inputs).detach().numpy())
    torch.save(inputs, sys.argv[2] + "/inputs.pt")

    tracking.configure_tracking(sys.argv[2] + "/mlruns")
    with tracking.start_run("lorentz") as run:
        tracking.log_model(model, "model", registered_model_name="lorentz")
    open(sys.argv[2] + "/run_id.txt", "w").write(run.info.run_id)
    """)


def _write_legacy_store(directory: Path) -> None:
    """Run the research code in a separate interpreter to write ``mlruns``."""
    subprocess.run(
        [sys.executable, "-c", LOG_LEGACY_MODEL, str(RESEARCH_CODE), str(directory)],
        check=True,
        capture_output=True,
        env={**_clean_env(), "MLFLOW_ALLOW_FILE_STORE": "true"},
    )


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}


def _predict(model: torch.nn.Module, directory: Path) -> np.ndarray:
    model.eval()
    inputs = torch.load(directory / "inputs.pt", weights_only=False)
    with torch.no_grad():
        return model(inputs).numpy()


@pytest.fixture(scope="module")
def legacy_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An archive of a research-code store whose original location is gone.

    The store is written under ``<dir>/PENDULUM/src`` and then moved, so its
    recorded paths point nowhere, as on any machine other than the author's.
    """
    base = tmp_path_factory.mktemp("legacy")
    written = base / "PENDULUM" / "src"
    written.mkdir(parents=True)
    _write_legacy_store(written)
    archive = base / "archive"
    shutil.move(written, archive)
    return archive


@pytest.fixture
def moved_legacy_store(
    legacy_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """The research-code store, copied away from where it was written."""
    moved = tmp_path / "elsewhere"
    shutil.copytree(legacy_store, moved)
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(moved / "mlruns"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking.configure_tracking()
    yield moved
    # set_tracking_uri is process-global; do not leak this store into other tests
    mlflow.set_tracking_uri(previous_uri)


def test_a_moved_store_does_not_resolve_until_relocated(moved_legacy_store) -> None:
    with pytest.raises(MlflowException):
        tracking.load_model("models:/lorentz/latest", torch.device("cpu"))


@pytest.mark.parametrize("kind", ["registered", "run"])
def test_a_research_code_model_loads_after_relocation(
    moved_legacy_store: Path, kind: str
) -> None:
    assert tracking.relocate_file_store(moved_legacy_store / "mlruns")
    run_id = (moved_legacy_store / "run_id.txt").read_text()
    uri = "models:/lorentz/latest" if kind == "registered" else f"runs:/{run_id}/model"

    model = tracking.load_model(uri, torch.device("cpu"))

    # The pickle names models.architectures.LSTM_MIONet; it becomes ours.
    assert type(model) is LSTM_MIONet
    expected = np.load(moved_legacy_store / "expected.npy")
    assert np.array_equal(_predict(model, moved_legacy_store), expected)
    # The stand-in modules are gone again after the load.
    assert "models" not in sys.modules
    assert "models.architectures" not in sys.modules


def test_relocation_is_idempotent(moved_legacy_store: Path) -> None:
    assert tracking.relocate_file_store(moved_legacy_store / "mlruns")
    assert tracking.relocate_file_store(moved_legacy_store / "mlruns") == []


def test_relocate_mlruns_command(moved_legacy_store: Path, capsys) -> None:
    assert cli_main(["relocate-mlruns", str(moved_legacy_store / "mlruns")]) == 0
    assert "Rewrote the paths" in capsys.readouterr().out
    model = tracking.load_model("models:/lorentz/latest", torch.device("cpu"))
    assert type(model) is LSTM_MIONet


def test_relocation_keeps_file_uris_and_quotes(tmp_path: Path) -> None:
    """Older MLflow releases wrote ``file://`` URIs; unrelated values stay put."""
    store = tmp_path / "mlruns"
    (store / "1" / "abc").mkdir(parents=True)
    (store / "1" / "meta.yaml").write_text(
        "artifact_location: file:///PENDULUM/src/mlruns/1\nname: lorentz\n"
    )
    (store / "1" / "abc" / "meta.yaml").write_text(
        "artifact_uri: 'file:///PENDULUM/src/mlruns/1/abc/artifacts'\n"
        "source: models:/m-0123\n"
        "run_name: /not/a/store/path\n"
    )

    rewritten = tracking.relocate_file_store(store)

    assert len(rewritten) == 2
    root = str(store.resolve())
    assert (store / "1" / "meta.yaml").read_text() == (
        f"artifact_location: file://{root}/1\nname: lorentz\n"
    )
    assert (store / "1" / "abc" / "meta.yaml").read_text() == (
        f"artifact_uri: 'file://{root}/1/abc/artifacts'\n"
        "source: models:/m-0123\n"
        "run_name: /not/a/store/path\n"
    )


def test_relocation_handles_a_store_moved_more_than_once(tmp_path: Path) -> None:
    """The published store: experiments under one old root, most runs under another."""
    store = tmp_path / "mlruns"
    (store / "7" / "run1").mkdir(parents=True)
    (store / "7" / "run2").mkdir(parents=True)
    (store / "7" / "meta.yaml").write_text(
        "artifact_location: file:///LSTM-MIONet/src/mlruns/7\n"
    )
    (store / "7" / "run1" / "meta.yaml").write_text(
        "artifact_uri: file:///LSTM-MIONet/src/mlruns/7/run1/artifacts\n"
    )
    (store / "7" / "run2" / "meta.yaml").write_text(
        "artifact_uri: file:///PENDULUM/src/mlruns/7/run2/artifacts\n"
    )

    assert len(tracking.relocate_file_store(store)) == 3
    root = str(store.resolve())
    assert (store / "7" / "run2" / "meta.yaml").read_text() == (
        f"artifact_uri: file://{root}/7/run2/artifacts\n"
    )
    for meta in store.rglob("meta.yaml"):
        assert "LSTM-MIONet" not in meta.read_text()
        assert "PENDULUM" not in meta.read_text()


def test_relocation_rejects_a_missing_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tracking.relocate_file_store(tmp_path / "missing")
