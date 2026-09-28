"""MLflow helpers.

The tracking URI comes from the ``tracking.uri`` configuration key and is
overridden by the ``MLFLOW_TRACKING_URI`` environment variable.  Relative URIs
(the default ``mlruns``) are resolved against the current working directory.

Two helpers keep the models trained with the research code usable: pickles that
reference its module layout (``models.architectures``) are loaded into the
classes of this package, and :func:`relocate_file_store` rewrites the absolute
paths an MLflow file store records, so a store can be moved to another machine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterator
from contextlib import contextmanager
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
    ``torch.nn.DataParallel`` are unwrapped.  Models pickled by the research
    code, which referenced ``models.architectures.LSTM_MIONet`` and friends,
    are loaded into the equivalent classes of this package.
    """
    try:
        model = mlflow.pytorch.load_model(uri, map_location=device)
    except ModuleNotFoundError as exc:
        if exc.name not in LEGACY_MODULE_NAMES:
            raise
        with _legacy_module_aliases():
            model = mlflow.pytorch.load_model(uri, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.to(device)


#: Modules the research code pickled its models from (it ran from ``src/``).
LEGACY_MODULE_NAMES = ("models", "models.architectures", "utils", "utils.torch_utils")


def _legacy_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for the research code's modules, backed by this package."""
    from blstm_mionet.models import (
        LSTM_MLP,
        MLP,
        DeepONet,
        DeepONet_Local,
        LSTM_DeepONet,
        LSTM_MIONet,
        ReLUSin,
        Sin,
        get_activation,
    )

    architectures = types.ModuleType("models.architectures")
    for obj in (
        LSTM_MIONet,
        LSTM_DeepONet,
        DeepONet,
        DeepONet_Local,
        MLP,
        LSTM_MLP,
        get_activation,
    ):
        setattr(architectures, obj.__name__, obj)
    torch_utils = types.ModuleType("utils.torch_utils")
    torch_utils.sin_act = Sin
    torch_utils.Rsin = ReLUSin

    models = types.ModuleType("models")
    models.__path__ = []
    models.architectures = architectures
    utils = types.ModuleType("utils")
    utils.__path__ = []
    utils.torch_utils = torch_utils
    return {
        "models": models,
        "models.architectures": architectures,
        "utils": utils,
        "utils.torch_utils": torch_utils,
    }


@contextmanager
def _legacy_module_aliases() -> Iterator[None]:
    """Register the stand-in modules for the duration of one load."""
    added = []
    for name, module in _legacy_modules().items():
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


#: ``meta.yaml`` keys under which an MLflow file store records absolute paths.
_PATH_KEYS = ("artifact_location", "artifact_uri", "storage_location", "source")
_PATH_LINE = re.compile(
    r"^(?P<key>" + "|".join(_PATH_KEYS) + r"): (?P<quote>['\"]?)"
    r"(?P<scheme>file://)?(?P<path>/[^'\"]*)(?P=quote)\s*$"
)


def relocate_file_store(store: str | Path) -> list[Path]:
    """Point the absolute paths of a moved MLflow file store at its new location.

    A file store writes the absolute location of every experiment, run, logged
    model and registered model version into its ``meta.yaml`` files, so after
    the directory is copied elsewhere ``runs:/`` and ``models:/`` URIs resolve
    to the old place.  Every ``meta.yaml`` records its own location, e.g. a
    run's ``artifact_uri`` is ``<old root>/<experiment>/<run>/artifacts``, so
    the old root is recovered file by file (a store that was moved before can
    hold several) and replaced by the store's current absolute path.  Returns
    the rewritten files; running it again is a no-op.
    """
    root = Path(store).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no MLflow file store at {root}")

    metas = sorted(root.rglob("meta.yaml"))
    old_roots = set()
    for meta in metas:
        own = "/" + meta.parent.relative_to(root).as_posix()
        for line in meta.read_text(encoding="utf-8").splitlines():
            match = _PATH_LINE.match(line)
            if not match:
                continue
            path = match["path"].rstrip("/")
            for suffix in (own, own + "/artifacts"):
                if path.endswith(suffix) and len(path) > len(suffix):
                    old_roots.add(path[: -len(suffix)])
    old_roots.discard(str(root))
    # the longest root first, in case one old root is nested inside another
    old_roots = sorted(old_roots, key=len, reverse=True)

    rewritten = []
    for meta in metas:
        lines = meta.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            match = _PATH_LINE.match(line.rstrip("\n"))
            if not match:
                continue
            for old in old_roots:
                path = match["path"]
                if path == old or path.startswith(old + "/"):
                    new_path = str(root) + path[len(old) :]
                    lines[i] = (
                        f"{match['key']}: {match['quote']}{match['scheme'] or ''}"
                        f"{new_path}{match['quote']}\n"
                    )
                    changed = True
                    break
        if changed:
            meta.write_text("".join(lines), encoding="utf-8")
            rewritten.append(meta)
    return rewritten


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
