"""MLflow helpers.

The tracking URI comes from the ``tracking.uri`` configuration key and is
overridden by the ``MLFLOW_TRACKING_URI`` environment variable.  Relative URIs
(the default ``mlruns``) are resolved against the current working directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import torch

ENSEMBLE_ARTIFACT_PATH = "ensemble"
"""Artifact directory holding ``member_XXXX.pt`` state dicts of a reSGLD run."""

MODEL_ARTIFACT_NAME = "model"
"""Artifact name of the model logged with ``mlflow.pytorch.log_model``."""


def configure_tracking(uri: str = "mlruns") -> str:
    """Point MLflow at ``uri`` (or ``MLFLOW_TRACKING_URI``) and return it."""
    resolved = os.environ.get("MLFLOW_TRACKING_URI") or uri
    if _is_file_store(resolved):
        # MLflow >= 3.13 refuses the filesystem backend unless this is set; the
        # published ``mlruns`` directory of the paper is a file store.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(resolved)
    return resolved


def _is_file_store(uri: str) -> bool:
    scheme = urlparse(uri).scheme
    return scheme in ("", "file")


def start_run(experiment_name: str, run_name: str | None = None) -> Any:
    """Set the experiment and start a run (context manager)."""
    mlflow.set_experiment(experiment_name=experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_config(config: Any) -> None:
    """Log a configuration object as flat MLflow parameters."""
    params = {key: value for key, value in config.to_flat_dict().items()}
    mlflow.log_params(params)


def log_source_snapshot(artifact_path: str = "source") -> None:
    """Snapshot the installed ``blstm_mionet`` package into the run.

    Replaces the hardcoded ``mlflow.log_artifact("train.py", ...)`` calls of the
    original trainer.  The current git commit is recorded as a tag when the
    package is used from a checkout.
    """
    import blstm_mionet

    package_dir = Path(blstm_mionet.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / package_dir.name
        shutil.copytree(
            package_dir,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        mlflow.log_artifacts(
            str(destination), artifact_path=f"{artifact_path}/{package_dir.name}"
        )

    mlflow.set_tag("blstm_mionet.version", getattr(blstm_mionet, "__version__", ""))
    sha = _git_sha(package_dir)
    if sha:
        mlflow.set_tag("blstm_mionet.git_sha", sha)


def _git_sha(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def log_model(
    model: torch.nn.Module,
    name: str,
    registered_model_name: str | None = None,
) -> str:
    """Log a PyTorch model and return a URI that can be loaded back.

    MLflow 3 renamed ``artifact_path`` to ``name`` and made the traced ``pt2``
    serialisation the default; the pickled format is requested explicitly so
    that models whose ``forward`` takes a list of tensors keep working.  Both
    keywords are attempted so the code also runs on MLflow 2.x.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Stating the requirements explicitly skips MLflow's environment inference,
    # which otherwise re-imports the model in a subprocess (several seconds per
    # call, once per improving epoch during training).
    pip_requirements = [
        f"torch=={torch.__version__.split('+')[0]}",
        f"blstm-mionet=={_package_version()}",
    ]

    for kwargs in (
        {"name": name, "serialization_format": "pickle"},
        {"name": name},
        {"artifact_path": name},
    ):
        try:
            info = mlflow.pytorch.log_model(
                model,
                registered_model_name=registered_model_name,
                pip_requirements=pip_requirements,
                **kwargs,
            )
        except TypeError:
            continue
        return getattr(info, "model_uri", None) or f"runs:/{_active_run_id()}/{name}"
    raise RuntimeError("mlflow.pytorch.log_model rejected every supported signature")


def _package_version() -> str:
    from blstm_mionet import __version__

    return __version__


def log_state_dict(state_dict: dict, filename: str, artifact_path: str) -> None:
    """Write ``state_dict`` to a temporary file and log it as a run artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        torch.save({"state_dict": state_dict}, path)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def load_model(uri: str, device: torch.device) -> torch.nn.Module:
    """Load a model from any MLflow URI and move it to ``device``.

    Handles ``models:/<name>/latest``, ``models:/<model id>`` and
    ``runs:/<run id>/<artifact path>``; models saved through
    ``torch.nn.DataParallel`` are unwrapped.
    """
    model = mlflow.pytorch.load_model(uri, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.to(device)


def run_id_from_uri(uri: str) -> str:
    """Extract a run id from ``runs:/<id>``, ``runs:/<id>/<path>`` or a bare id."""
    text = uri.strip()
    if text.startswith("runs:/"):
        text = text[len("runs:/") :]
    return text.strip("/").split("/")[0]


def download_artifacts(run_id: str, artifact_path: str) -> str:
    """Download a run artifact directory and return the local path."""
    return mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )


def list_artifacts(run_id: str, artifact_path: str) -> list[str]:
    """List the artifact paths stored under ``artifact_path`` of ``run_id``."""
    return [
        item.path
        for item in mlflow.artifacts.list_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
    ]


def _active_run_id() -> str:
    run = mlflow.active_run()
    return run.info.run_id if run is not None else ""
