"""Selection and resampling of the Ausgrid solar-home half-hour dataset.

The CSV files are licensed by Ausgrid and are **not** redistributed with this
repository; their locations are given by ``data.ausgrid.csv_paths`` in the
experiment configuration and are resolved relative to the current working
directory.  Each file is expected to have one title row followed by a header
row containing at least the columns ``Customer``, ``Consumption Category`` and
``date`` plus the 48 half-hour reading columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

DateLike = str | date | datetime | None


def _as_datetime(value: DateLike) -> datetime | None:
    """Accept ``None``, ``YYYY-MM-DD`` strings and (YAML) date objects."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.strptime(value, "%Y-%m-%d")


def select_ausgrid_data(
    csv_paths: Sequence[str],
    cust_id: Iterable[int] | None = None,
    start_date: DateLike = None,
    end_date: DateLike = None,
    category: str | None = "GG",
    delta_t_idxs: float = 0.1,
    column_start: int = 18,
    column_end: int = 39,
    min_nonzero_fraction: float = 0.8,
    sample_interval_hours: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Select, filter and cubically resample Ausgrid daily profiles.

    Parameters mirror the original ``select_Ausgrid_data`` except that the CSV
    locations are now arguments instead of hardcoded relative paths.
    """
    if not csv_paths:
        raise ValueError("csv_paths must list at least one Ausgrid CSV file")

    start_date = _as_datetime(start_date)
    end_date = _as_datetime(end_date)

    # Concatenate the data from multiple files along the row axis
    data = pd.concat(
        [pd.read_csv(filepath, header=1) for filepath in csv_paths], axis=0
    )
    # Convert the date column from string to datetime.  The releases disagree:
    # 2010-2011 writes "1-Jul-10", the two "v2" files write the Australian
    # day-first "1/07/2011", so every element is parsed on its own.
    data["date"] = pd.to_datetime(data["date"], format="mixed", dayfirst=True)
    data_len = data.shape[0]
    idxs_query_time = (
        (data["date"] >= start_date) & (data["date"] <= end_date)
        if (start_date is not None) and (end_date is not None)
        else np.ones(data_len, dtype=bool)
    )

    cust_id = list(cust_id) if cust_id is not None else None
    if cust_id:
        idxs_query_cust = data["Customer"].isin(cust_id)
    else:
        idxs_query_cust = np.ones(data_len, dtype=bool)

    idxs_query_category = (
        data["Consumption Category"].str.upper() == category
        if category is not None
        else np.ones(data_len, dtype=bool)
    )
    idxs_query = idxs_query_time & idxs_query_cust & idxs_query_category
    data_select = data[idxs_query]
    # The selected columns are the ones that are mostly non-zero.
    data_select = data_select.iloc[:, column_start:column_end].values + 1e-7

    # Remove the daily records with too many zeros
    idxs_nonzero = (data_select > 1e-5).sum(axis=1) >= (
        data_select.shape[1] * min_nonzero_fraction
    )
    data_select = data_select[idxs_nonzero]
    if data_select.shape[0] == 0:
        raise ValueError(
            "no Ausgrid records matched the selection; check the customer ids, "
            "dates and category"
        )

    # Interpolate the data every delta_t_idxs points
    data_select_len = data_select.shape[0]
    t_idxs = np.arange(data_select.shape[1])
    t_idxs_interp = np.arange(0, t_idxs.size, delta_t_idxs)
    data_select_interp = np.zeros((data_select_len, t_idxs_interp.size))

    for i in range(data_select_len):
        spline = interp1d(
            t_idxs, data_select[i], kind="cubic", fill_value=1e-6, bounds_error=False
        )
        data_select_interp[i] = spline(t_idxs_interp)

    if verbose:
        customers = f"{cust_id[0]} - {cust_id[-1]}" if cust_id else "all customers"
        print(
            f"Ausgrid data with {category} category selected from {start_date} to {end_date} for customer {customers}."
        )
        print(
            f"Shape of the selected data: num_trajs= {data_select_interp.shape[0]}, num_time = {data_select_interp.shape[1]}"
        )

    return {
        "x": np.expand_dims(data_select_interp, axis=2).transpose(1, 2, 0),
        # the raw columns are sampled every ``sample_interval_hours`` hours
        "t": t_idxs_interp * sample_interval_hours,
    }
