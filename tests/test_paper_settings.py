"""The shipped configurations reproduce the experimental settings of the paper.

Values the paper states (Sections 3.5 and 4) are checked against its text; the
network sizes, which the paper does not state, against the published models of
the data release.  A change that breaks one of these is a change of experiment,
not a refactoring.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from blstm_mionet.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _load(name: str, bayesian: bool = False):
    return load_config(
        CONFIGS / f"{name}.yaml",
        bayesian_path=CONFIGS / "bayesian" / f"{name}.yaml" if bayesian else None,
    )


def test_lorentz_matches_section_4_1() -> None:
    config = _load("lorentz")
    data, training, inference = config.data, config.training, config.inference
    # 5000 initial values from X0 = [-17, 20] x [-23, 28] x [0, 50]
    assert data.n_sample == 5000
    assert data.x_init_pts == [[-17.0, 20.0], [-23.0, 28.0], [0.0, 50.0]]
    assert data.control is None  # autonomous
    # RK-4 at Delta = 0.01 s on [0, 20] s
    assert (data.step_size, data.t_max) == (0.01, 20.0)
    # h_max = 0.02 s; x replicated 4 times -> N_train = 20000
    assert training.search_len * data.step_size == pytest.approx(0.02)
    assert training.search_num * data.n_sample == 20000
    # tested on 100 trajectories replicated 200 times, at h = 0.01 s
    assert inference.search_num == 200
    assert not inference.search_random
    assert 0.5 * inference.search_len * data.step_size == pytest.approx(0.01)


def test_pendulum_matches_section_4_2() -> None:
    config = _load("pendulum")
    data, training, inference = config.data, config.training, config.inference
    assert data.n_sample == 5000
    assert data.x_init_pts == [[-math.pi, math.pi], [-8.0, 8.0]]
    assert data.control == "gaussian"
    assert (data.step_size, data.t_max) == (0.01, 10.0)
    # h_max = 0.02 s; u replicated 10 times -> N_train = 50000
    assert training.search_len * data.step_size == pytest.approx(0.02)
    assert training.search_num * data.n_sample == 50000
    assert inference.search_num == 200
    assert 0.5 * inference.search_len * data.step_size == pytest.approx(0.01)


def test_ausgrid_matches_section_4_3() -> None:
    config = _load("ausgrid")
    ausgrid, training, inference = (
        config.data.ausgrid,
        config.training,
        config.inference,
    )
    # gross generation of customers 1-50, 2010-07-01 to 2011-06-30
    assert ausgrid.category == "GG"
    assert ausgrid.customer_id == list(range(1, 51))
    assert (str(ausgrid.start_date), str(ausgrid.end_date)) == (
        "2010-07-01",
        "2011-06-30",
    )
    # columns 18-38 of each day, interpolated at h = 0.05 hours
    assert (ausgrid.column_start, ausgrid.column_end) == (18, 39)
    step = ausgrid.delta_t_idxs * ausgrid.sample_interval_hours
    assert step == pytest.approx(0.05)
    # h_n in [0, 0.5] hours; every profile replicated 5 times
    assert training.search_len * step == pytest.approx(0.5)
    assert training.search_num == 5
    # tested with 100 sub-sequences per day, at h_n = 0.25 hours
    assert inference.search_num == 100
    assert 0.5 * inference.search_len * step == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["lorentz", "pendulum", "ausgrid"])
def test_bayesian_ensemble_matches_section_3_5(name: str) -> None:
    config = _load(name, bayesian=True)
    # M = 300 posterior members are evaluated, out of those collected
    assert config.inference.n_ensemble == 300
    assert config.bayesian.n_ensemble >= config.inference.n_ensemble
    # the explore chain runs at a higher temperature than the exploit chain
    assert config.bayesian.explore.tau > config.bayesian.exploit.tau


@pytest.mark.parametrize(
    ("name", "lstm_size", "offset"),
    [("lorentz", 10, 0.02), ("pendulum", 10, 0.02), ("ausgrid", 100, 0.0)],
)
def test_network_matches_the_published_model(
    name: str, lstm_size: int, offset: float
) -> None:
    """Read off models:/{lorentz,pendulum,Ausgrid}/latest of the data release."""
    config = _load(name)
    model = config.model
    assert model.architecture == "LSTM_MIONet"
    for branch in (model.branch_state, model.branch_memory, model.trunk):
        assert (branch.width, branch.depth, branch.activation) == (200, 3, "relu")
    assert model.branch_memory.lstm_size == lstm_size
    assert model.branch_memory.lstm_layer_num == 2
    assert model.use_bias
    assert config.training.offset == pytest.approx(offset)
