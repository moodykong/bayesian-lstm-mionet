"""Shared fixtures for the ``blstm_mionet`` test suite.

Everything here is CPU only, deterministic and confined to ``tmp_path`` /
``tmp_path_factory``: no test is allowed to write into the repository or to
touch the network.  The datasets are deliberately tiny (a couple of short
trajectories) so that the whole suite finishes in a few minutes.
"""

from __future__ import annotations

import os

# OpenBLAS/OpenMP oversubscribe badly on the matrices used here (the Gaussian
# random field factorises a 2000 x 2000 covariance, the models are tiny), which
# makes the suite several times slower on many-core machines.  The limit has to
# be in place before NumPy/SciPy/torch are imported, hence the early os.environ
# block and the noqa markers below.
_THREADS = str(min(4, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import datetime  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from blstm_mionet.cli.main import main as cli_main  # noqa: E402
from blstm_mionet.config import (  # noqa: E402
    BayesianConfig,
    BranchMemoryConfig,
    BranchStateConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrunkConfig,
)
from blstm_mionet.data.ausgrid import select_ausgrid_data  # noqa: E402
from blstm_mionet.data.datasets import (  # noqa: E402
    prepare_torch_dataset,
    split_dataset,
)
from blstm_mionet.data.generate import (  # noqa: E402
    generate_trajectories,
    load_dataset,
    save_dataset,
)
from blstm_mionet.models import build_model, dataset_preparer_for  # noqa: E402
from blstm_mionet.utils.seed import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
BAYESIAN_CONFIG_DIR = CONFIG_DIR / "bayesian"

#: ``t_max = 1.0`` with ``step_size = 0.01`` gives 99 stored time steps.
TINY_T_MAX = 1.0
TINY_STEP = 0.01
TINY_N_TIME = int(TINY_T_MAX / TINY_STEP) - 1

LORENTZ_INIT_PTS = [[-17.0, 20.0], [-23.0, 28.0], [0.0, 50.0]]
PENDULUM_INIT_PTS = [[-np.pi, np.pi], [-8.0, 8.0]]

#: Half-hour column labels of the Ausgrid "solar home" files: each reading is
#: labelled with the end of its half hour, "0:30", "1:00", ..., "23:30", "0:00".
HALF_HOUR_COLUMNS = [f"{(k + 1) // 2 % 24}:{30 * ((k + 1) % 2):02d}" for k in range(48)]


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker (slow tests stay enabled by default)."""
    config.addinivalue_line(
        "markers",
        "slow: end-to-end command line test; enabled by default, deselect with "
        "-m 'not slow'",
    )


# --------------------------------------------------------------------------- #
# Paths of the shipped configuration files
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def lorentz_config_path() -> Path:
    return CONFIG_DIR / "lorentz.yaml"


@pytest.fixture(scope="session")
def pendulum_config_path() -> Path:
    return CONFIG_DIR / "pendulum.yaml"


@pytest.fixture(scope="session")
def ausgrid_config_path() -> Path:
    return CONFIG_DIR / "ausgrid.yaml"


@pytest.fixture(scope="session")
def bayesian_lorentz_config_path() -> Path:
    return BAYESIAN_CONFIG_DIR / "lorentz.yaml"


# --------------------------------------------------------------------------- #
# Tiny datasets
# --------------------------------------------------------------------------- #
def _lorentz_data_config(n_sample: int) -> DataConfig:
    return DataConfig(
        system="lorentz",
        t_max=TINY_T_MAX,
        step_size=TINY_STEP,
        n_sample=n_sample,
        x_init_pts=LORENTZ_INIT_PTS,
        control=None,
        seed=999,
        verbose=False,
    )


def _pendulum_data_config(n_sample: int) -> DataConfig:
    return DataConfig(
        system="pendulum",
        t_max=TINY_T_MAX,
        step_size=TINY_STEP,
        n_sample=n_sample,
        x_init_pts=PENDULUM_INIT_PTS,
        control="gaussian",
        seed=999,
        verbose=False,
    )


@pytest.fixture(scope="session")
def lorentz_data() -> dict[str, Any]:
    """Six short Lorenz trajectories as an in-memory dataset dictionary."""
    set_seed(999)
    return generate_trajectories(_lorentz_data_config(6))


@pytest.fixture(scope="session")
def pendulum_data() -> dict[str, Any]:
    """Four short pendulum trajectories driven by a Gaussian random field."""
    set_seed(999)
    return generate_trajectories(_pendulum_data_config(4))


@pytest.fixture(scope="session")
def lorentz_dataset(tmp_path_factory: pytest.TempPathFactory, lorentz_data) -> Path:
    """``.npy`` file holding :func:`lorentz_data`."""
    path = tmp_path_factory.mktemp("datasets") / "lorentz_tiny.npy"
    return save_dataset(lorentz_data, path)


@pytest.fixture(scope="session")
def pendulum_dataset(tmp_path_factory: pytest.TempPathFactory, pendulum_data) -> Path:
    """``.npy`` file holding :func:`pendulum_data`."""
    path = tmp_path_factory.mktemp("datasets") / "pendulum_tiny.npy"
    return save_dataset(pendulum_data, path)


@pytest.fixture(scope="session")
def single_trajectory_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One Lorenz trajectory: what the recursive rollout expects."""
    set_seed(999)
    data = generate_trajectories(_lorentz_data_config(1))
    path = tmp_path_factory.mktemp("datasets") / "lorentz_one.npy"
    return save_dataset(data, path)


# --------------------------------------------------------------------------- #
# Synthetic Ausgrid CSV (the real files are licensed and not redistributable)
# --------------------------------------------------------------------------- #
def write_ausgrid_csv(
    path: Path,
    customers: tuple[int, ...] = (1, 2, 3),
    n_days: int = 41,
    start: datetime.date = datetime.date(2010, 7, 1),
    category: str = "GG",
    dead_customers: tuple[int, ...] = (),
) -> Path:
    """Write a file shaped like an Ausgrid "solar home half-hour data" export.

    One title row, then the header row, then one row per customer and day: five
    metadata columns (``Customer``, ``Generator Capacity``, ``Postcode``,
    ``Consumption Category``, ``date``), the 48 half-hour readings and a
    ``Row Quality`` column, i.e. 54 columns, so that the ``iloc[:, 18:39]``
    daylight window used by the loader exists.  Dates use the Australian
    day-first format (``1/07/2010``).  Customers listed in ``dead_customers``
    only report zeros and must be dropped by the non-zero filter.
    """
    header = [
        "Customer",
        "Generator Capacity",
        "Postcode",
        "Consumption Category",
        "date",
        *HALF_HOUR_COLUMNS,
        "Row Quality",
    ]
    assert len(header) == 54
    lines = ["Solar home half-hour data (synthetic test fixture)" + "," * 4]
    lines.append(",".join(header))
    for customer in customers:
        for day_index in range(n_days):
            day = start + datetime.timedelta(days=day_index)
            readings = []
            for column in range(48):
                if customer in dead_customers:
                    readings.append(0.0)
                    continue
                hour = 0.5 * column
                bell = float(np.exp(-((hour - 12.0) ** 2) / 18.0))
                readings.append(
                    round(
                        0.6 * bell * (1.0 + 0.05 * customer) + 0.01 * (day_index % 5), 4
                    )
                )
            row = [
                str(customer),
                "2.5",
                "2000",
                category,
                f"{day.day}/{day.month:02d}/{day.year}",
                *(f"{value}" for value in readings),
                "A",
            ]
            lines.append(",".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def ausgrid_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic Ausgrid CSV: customers 1-3 plus an all-zero customer 4."""
    path = tmp_path_factory.mktemp("ausgrid") / "solar_home_synthetic.csv"
    return write_ausgrid_csv(path, customers=(1, 2, 3, 4), dead_customers=(4,))


@pytest.fixture(scope="session")
def ausgrid_csv_writer():
    """The CSV writer itself, for tests that need a differently shaped file."""
    return write_ausgrid_csv


@pytest.fixture(scope="session")
def ausgrid_dataset(
    tmp_path_factory: pytest.TempPathFactory, ausgrid_csv: Path
) -> Path:
    """``.npy`` dataset selected from the synthetic Ausgrid CSV."""
    data = select_ausgrid_data(
        [str(ausgrid_csv)],
        cust_id=[1, 2],
        start_date="2010-07-01",
        end_date="2010-07-10",
        category="GG",
        verbose=False,
    )
    path = tmp_path_factory.mktemp("datasets") / "ausgrid_tiny.npy"
    return save_dataset(data, path)


# --------------------------------------------------------------------------- #
# Models and prepared torch datasets
# --------------------------------------------------------------------------- #
def tiny_model_config(architecture: str = "LSTM_MIONet") -> ModelConfig:
    """A minuscule version of the paper architecture (fast on CPU)."""
    return ModelConfig(
        architecture=architecture,
        branch_state=BranchStateConfig(width=8, depth=2, activation="relu"),
        branch_memory=BranchMemoryConfig(
            width=8, depth=2, activation="relu", lstm_size=6, lstm_layer_num=1
        ),
        trunk=TrunkConfig(width=8, depth=2, activation="relu"),
        use_bias=True,
    )


@pytest.fixture
def model_config():
    """Factory returning :func:`tiny_model_config`."""
    return tiny_model_config


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")


def build_torch_dataset(
    dataset_path: Path | str,
    *,
    architecture: str = "LSTM_MIONet",
    search_len: int = 2,
    search_num: int = 4,
    search_random: bool = False,
    scale_mode: str = "",
    test_size: float = 1.0,
    state_component: int = 0,
):
    """Load a ``.npy`` dataset and prepare it exactly like the CLI does."""
    raw = load_dataset(dataset_path)
    train_split, test_split = split_dataset(raw, test_size=test_size, verbose=False)
    split = test_split if test_size > 0.0 else train_split
    return prepare_torch_dataset(
        split,
        dataset_preparer_for(architecture),
        state_component=state_component,
        search_len=search_len,
        search_num=search_num,
        search_random=search_random,
        offset=0.0,
        t_max=None,
        scale_mode=scale_mode,
        device=torch.device("cpu"),
        verbose=False,
    )


@pytest.fixture
def make_torch_dataset():
    """Factory fixture wrapping :func:`build_torch_dataset`."""
    return build_torch_dataset


# --------------------------------------------------------------------------- #
# MLflow tracking
# --------------------------------------------------------------------------- #
@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test MLflow file store under ``tmp_path``."""
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tracking_dir))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return tracking_dir


@pytest.fixture(scope="session")
def _shared_tracking_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One MLflow store shared by the (expensive) training fixtures."""
    return tmp_path_factory.mktemp("shared_mlruns")


@pytest.fixture
def shared_tracking(
    monkeypatch: pytest.MonkeyPatch, _shared_tracking_dir: Path
) -> Path:
    """Point MLflow at the shared store for the duration of one test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(_shared_tracking_dir))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return _shared_tracking_dir


@pytest.fixture(scope="session")
def training_run_cache() -> dict[str, Any]:
    """Memo for the session-wide training runs (see ``test_training.py``)."""
    return {}


# --------------------------------------------------------------------------- #
# Command line helper
# --------------------------------------------------------------------------- #
RUN_ID_PATTERN = re.compile(r"MLflow run id:\s*(\S+)")
MODEL_URI_PATTERN = re.compile(r"Model uri \(run scoped\):\s*(\S+)")


def parse_run_id(output: str) -> str:
    """Extract the run id printed by ``blstm-mionet train``."""
    match = RUN_ID_PATTERN.search(output)
    assert match is not None, f"no run id in the training output:\n{output}"
    return match.group(1)


def parse_model_uri(output: str) -> str:
    """Extract the run scoped model URI printed by ``blstm-mionet train``."""
    match = MODEL_URI_PATTERN.search(output)
    assert match is not None, f"no model uri in the training output:\n{output}"
    return match.group(1)


@pytest.fixture
def cli_workdir(tmp_path: Path) -> Path:
    """``tmp_path``, made to look like this repository's uv project.

    The command line tests have to run outside the repository (they write
    datasets, figures and an MLflow store into the working directory), but when
    ``mlflow.pytorch.log_model`` cannot find a ``uv.lock`` next to the working
    directory it falls back to re-importing the model in a subprocess to infer
    its pip requirements, which costs seconds per logged model.  Copying the two
    project files restores the fast path without changing what the tests
    exercise; if ``uv`` is unavailable MLflow simply falls back again.
    """
    for name in ("pyproject.toml", "uv.lock"):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copyfile(source, tmp_path / name)
    return tmp_path


@pytest.fixture
def run_cli(
    monkeypatch: pytest.MonkeyPatch, cli_workdir: Path, mlflow_tracking_uri: Path
):
    """Run ``blstm-mionet`` in-process with ``tmp_path`` as working directory."""

    def _run(*argv: Any, expected_code: int = 0) -> int:
        monkeypatch.chdir(cli_workdir)
        args = [str(item) for item in argv]
        code = cli_main(args)
        assert code == expected_code, (
            f"`blstm-mionet {' '.join(args)}` returned {code}, "
            f"expected {expected_code}"
        )
        return code

    return _run


# --------------------------------------------------------------------------- #
# Shared (expensive) training runs
# --------------------------------------------------------------------------- #
def tiny_train_config(datafile: Path | str, **overrides: Any) -> TrainConfig:
    """A training configuration that converges to nothing but runs in seconds."""
    config = TrainConfig(
        datafile=str(datafile),
        state_component=0,
        search_len=2,
        search_num=4,
        search_random=True,
        offset=0.0,
        t_max=None,
        scale_mode="",
        holdout_size=0.2,
        validation_split=0.25,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        validate_freq=1,
        use_scheduler=False,
        early_stopping_epochs=100,
        monitor_metric="val_loss",
        save_model=True,
        registered_model_name=None,
        plot_trajs=False,
        figure_dir="figures",
        device="cpu",
        experiment_name="blstm_mionet_tests",
        run_name="tiny",
        verbose=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def tiny_bayesian_config(**overrides: Any) -> BayesianConfig:
    config = BayesianConfig(
        n_ensemble=2,
        use_grad_norm=True,
        grad_norm=50.0,
        print_every=1,
        level=15.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def prepare_training_dataset(train_config: TrainConfig, model_config: ModelConfig):
    """The ``train`` sub-command's dataset pipeline, without the CLI."""
    raw = load_dataset(train_config.datafile)
    train_split, _ = split_dataset(
        raw, test_size=train_config.holdout_size, verbose=False
    )
    return prepare_torch_dataset(
        train_split,
        dataset_preparer_for(model_config.architecture),
        state_component=train_config.state_component,
        search_len=train_config.search_len,
        search_num=train_config.search_num,
        search_random=train_config.search_random,
        offset=train_config.offset,
        t_max=train_config.t_max,
        scale_mode=train_config.scale_mode,
        device=torch.device("cpu"),
        verbose=False,
    )


def execute_training(
    train_config: TrainConfig,
    model_config: ModelConfig,
    bayesian: BayesianConfig | None = None,
) -> dict[str, Any]:
    """Run one training inside a fresh MLflow run and report what it produced."""
    import copy as _copy

    from blstm_mionet.training import tracking
    from blstm_mionet.training.resgld import train_resgld
    from blstm_mionet.training.trainer import train_adam

    dataset, _, state_feature_num = prepare_training_dataset(train_config, model_config)
    model = build_model(model_config, state_feature_num)
    device = torch.device("cpu")

    tracking.configure_tracking("mlruns")
    with tracking.start_run(
        train_config.experiment_name, train_config.run_name
    ) as active_run:
        run_id = active_run.info.run_id
        if bayesian is None:
            history = train_adam(train_config, model, dataset, device)
        else:
            history = train_resgld(
                config=train_config,
                bayesian=bayesian,
                model_exploit=model,
                model_explore=_copy.deepcopy(model),
                dataset=dataset,
                device=device,
            )
    return {
        "run_id": run_id,
        "history": history,
        "model": model,
        "model_config": model_config,
        "train_config": train_config,
        "state_feature_num": state_feature_num,
        "dataset": dataset,
        "model_uri": f"runs:/{run_id}/model",
    }


@pytest.fixture
def adam_run(shared_tracking, training_run_cache, lorentz_dataset: Path):
    """A two-epoch Adam run on the tiny Lorenz dataset (built once per session)."""
    if "adam" not in training_run_cache:
        set_seed(999)
        training_run_cache["adam"] = execute_training(
            tiny_train_config(lorentz_dataset, run_name="adam"),
            tiny_model_config("LSTM_MIONet"),
        )
    return training_run_cache["adam"]


@pytest.fixture
def resgld_run(shared_tracking, training_run_cache, lorentz_dataset: Path):
    """A five-epoch reSGLD run collecting two posterior members."""
    if "resgld" not in training_run_cache:
        set_seed(999)
        training_run_cache["resgld"] = execute_training(
            tiny_train_config(lorentz_dataset, epochs=5, run_name="resgld"),
            tiny_model_config("LSTM_MIONet"),
            bayesian=tiny_bayesian_config(n_ensemble=2),
        )
    return training_run_cache["resgld"]
