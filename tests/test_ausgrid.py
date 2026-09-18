"""Selection and resampling of the (synthetic) Ausgrid solar-home CSV files."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from blstm_mionet.data.ausgrid import _as_datetime, select_ausgrid_data
from blstm_mionet.data.datasets import split_dataset

#: 21 daylight columns (18:39) resampled every 0.1 index -> 210 points.
N_TIME = 210
#: Sampling interval of the raw columns, in hours.
RAW_INTERVAL = 0.5
DELTA_T_IDXS = 0.1

START = "2010-07-01"
END = "2010-08-10"


def _select(csv_paths, **kwargs):
    defaults = dict(
        cust_id=[1, 2, 3],
        start_date=START,
        end_date=END,
        category="GG",
        verbose=False,
    )
    defaults.update(kwargs)
    return select_ausgrid_data([str(path) for path in csv_paths], **defaults)


# --------------------------------------------------------------------------- #
# Shapes and time axis
# --------------------------------------------------------------------------- #
def test_select_shapes_and_time_axis(ausgrid_csv: Path) -> None:
    data = _select([ausgrid_csv])

    assert set(data) == {"x", "t"}
    # [N_time, C, N_traj]; 3 customers x 41 days of profiles.
    assert data["x"].shape == (N_TIME, 1, 123)
    assert data["t"].shape == (N_TIME,)
    assert np.isfinite(data["x"]).all()
    assert (data["x"] > 0.0).all()

    # ``t`` is in hours: 0.1 index steps of 0.5 h columns = 0.05 h.
    assert data["t"][0] == pytest.approx(0.0)
    assert np.allclose(np.diff(data["t"]), DELTA_T_IDXS * RAW_INTERVAL)
    assert data["t"][-1] == pytest.approx((N_TIME - 1) * DELTA_T_IDXS * RAW_INTERVAL)


def test_selected_dataset_splits_into_trajectories(ausgrid_csv: Path) -> None:
    """``split_dataset`` turns the selection into [N_traj, N_time, C]."""
    data = _select([ausgrid_csv])
    _, (u_test, x_test, t_test) = split_dataset(data, test_size=1.0, verbose=False)
    assert u_test is None
    assert x_test.shape == (123, N_TIME, 1)
    assert t_test.shape == (N_TIME,)


def test_interpolated_values_track_the_raw_readings(ausgrid_csv: Path) -> None:
    """Every 10th interpolated point is a raw half-hour reading."""
    data = _select([ausgrid_csv], cust_id=[1], delta_t_idxs=DELTA_T_IDXS)
    profile = data["x"][:, 0, 0]
    knots = profile[:: int(round(1 / DELTA_T_IDXS))]
    assert knots.size == 21
    # Five metadata columns precede the 48 half-hour readings, so the 18:39
    # window is the half-hour columns 13 to 33, i.e. 06:30 to 16:30.  The
    # synthetic profile is a bell centred on noon (half-hour column 24), which
    # lands on offset 24 - 13 = 11 of the window.
    assert knots.argmax() == 11
    assert profile.min() > 0.0
    # Unimodal: rising to the peak, falling afterwards.
    assert np.all(np.diff(knots[:12]) > 0)
    assert np.all(np.diff(knots[11:]) < 0)


def test_delta_t_idxs_controls_the_resampling(ausgrid_csv: Path) -> None:
    coarse = _select([ausgrid_csv], cust_id=[1], delta_t_idxs=1.0)
    assert coarse["x"].shape[0] == 21
    assert np.allclose(np.diff(coarse["t"]), RAW_INTERVAL)

    fine = _select([ausgrid_csv], cust_id=[1], delta_t_idxs=0.5)
    assert fine["x"].shape[0] == 42


def test_column_window_controls_the_number_of_readings(ausgrid_csv: Path) -> None:
    data = _select([ausgrid_csv], cust_id=[1], column_start=20, column_end=30)
    assert data["x"].shape[0] == 10 * int(round(1 / DELTA_T_IDXS))


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_customer_filter(ausgrid_csv: Path) -> None:
    one = _select([ausgrid_csv], cust_id=[2])
    two = _select([ausgrid_csv], cust_id=[2, 3])
    assert one["x"].shape[-1] == 41
    assert two["x"].shape[-1] == 82


def test_date_filter(ausgrid_csv: Path) -> None:
    """Dates are day-first (``1/07/2010``); the range is inclusive."""
    week = _select([ausgrid_csv], cust_id=[1], start_date=START, end_date="2010-07-07")
    assert week["x"].shape[-1] == 7

    single = _select(
        [ausgrid_csv], cust_id=[1], start_date="2010-07-15", end_date="2010-07-15"
    )
    assert single["x"].shape[-1] == 1


def test_date_filter_accepts_date_objects(ausgrid_csv: Path) -> None:
    """YAML parses an unquoted date into ``datetime.date``."""
    data = _select(
        [ausgrid_csv],
        cust_id=[1],
        start_date=datetime.date(2010, 7, 1),
        end_date=datetime.date(2010, 7, 3),
    )
    assert data["x"].shape[-1] == 3


def test_category_filter_is_case_insensitive(ausgrid_csv: Path) -> None:
    data = _select([ausgrid_csv], cust_id=[1], category="GG")
    assert data["x"].shape[-1] == 41
    with pytest.raises(ValueError, match="no Ausgrid records matched the selection"):
        _select([ausgrid_csv], cust_id=[1], category="GC")


def test_records_that_are_mostly_zero_are_dropped(ausgrid_csv: Path) -> None:
    """Customer 4 of the fixture reports only zeros and must disappear."""
    with pytest.raises(ValueError, match="no Ausgrid records matched the selection"):
        _select([ausgrid_csv], cust_id=[4])

    with_dead = _select([ausgrid_csv], cust_id=[1, 4])
    assert with_dead["x"].shape[-1] == 41


def test_min_nonzero_fraction_can_keep_them(ausgrid_csv: Path) -> None:
    kept = _select([ausgrid_csv], cust_id=[1, 4], min_nonzero_fraction=0.0)
    assert kept["x"].shape[-1] == 82


def test_no_customer_filter_selects_every_customer(ausgrid_csv: Path) -> None:
    data = _select([ausgrid_csv], cust_id=None)
    assert data["x"].shape[-1] == 123  # customer 4 is still dropped as all-zero


def test_multiple_csv_files_are_concatenated(
    tmp_path: Path, ausgrid_csv: Path, ausgrid_csv_writer
) -> None:
    second = ausgrid_csv_writer(
        tmp_path / "second_year.csv",
        customers=(7,),
        n_days=5,
        start=datetime.date(2010, 7, 1),
    )
    data = _select([ausgrid_csv, second], cust_id=[1, 7])
    assert data["x"].shape[-1] == 41 + 5


def test_empty_csv_paths_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one Ausgrid CSV file"):
        select_ausgrid_data([], verbose=False)


def test_verbose_prints_a_summary(ausgrid_csv: Path, capsys) -> None:
    _select([ausgrid_csv], verbose=True)
    printed = capsys.readouterr().out
    assert "GG category selected" in printed
    assert "customer 1 - 3" in printed
    assert "num_trajs= 123" in printed


# --------------------------------------------------------------------------- #
# Date coercion helper
# --------------------------------------------------------------------------- #
def test_as_datetime_accepts_the_three_supported_forms() -> None:
    expected = datetime.datetime(2010, 7, 1)
    assert _as_datetime(None) is None
    assert _as_datetime("2010-07-01") == expected
    assert _as_datetime(datetime.date(2010, 7, 1)) == expected
    assert _as_datetime(expected) is expected
