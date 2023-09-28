import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils.math_utils import grf_1d, integrate, runge_kutta
from utils.args_parser import add_data_args, args_to_config
from config import data_config
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser


def run(config):
    if config["ode_system"] == "Ausgrid":
        if isinstance(config["Ausgrid_start_date"], str):
            config["Ausgrid_start_date"] = datetime.strptime(
                config["Ausgrid_start_date"], "%Y-%m-%d"
            )
        if isinstance(config["Ausgrid_end_date"], str):
            config["Ausgrid_end_date"] = datetime.strptime(
                config["Ausgrid_end_date"], "%Y-%m-%d"
            )
        data = select_Ausgrid_data(
            config["Ausgrid_customer_id"],
            config["Ausgrid_start_date"],
            config["Ausgrid_end_date"],
            config["Ausgrid_category"],
            config["Ausgrid_delta_t_idxs"],
            config["verbose"],
        )
        with open(config["datafile_path"], "wb") as f:
            np.save(f, data)
        print(f'Data saved in {config["datafile_path"]} .')
        return None
    else:
        pass

    if config["ode_system"] == "pendulum":
        ode_system = data_config.pendulum
    elif config["ode_system"] == "lorentz":
        ode_system = data_config.lorentz
    else:
        raise ValueError("Invalid ode_system.")
    # Define the time parameters
    T = config["t_max"]
    h = config["step_size"]
    N_time = int(T / h)
    N_sample = config["n_sample"]
    # Define the initial states
    x_init_pts = config["x_init_pts"]
    x_init_pts = np.array(x_init_pts).reshape(-1, 2)
    x_num = x_init_pts.shape[0]
    x_init = np.random.uniform(x_init_pts[:, 0], x_init_pts[:, 1], (N_sample, x_num))

    # Generate the data
    data = dict()
    i = 0
    progress_bar = tqdm(
        total=N_sample, desc=f"Generating {N_sample} data ...", dynamic_ncols=True
    )
    while i < N_sample:
        # Define the control
        if config["ctr_func"] == "designate":
            control = data_config.control_formula
        elif config["ctr_func"] == "gaussian":
            grf = grf_1d(a=0.01)
            control = data_config.control_grf(grf)
        else:
            raise ValueError("Invalid ctr_func.")

        soln = integrate(runge_kutta, ode_system, control, x_init[i], h, N_time)

        if np.isnan(soln.x).sum() + np.isnan(soln.u).sum() > 0:
            continue
        x = (
            np.dstack((x, soln.x[:-1, :]))
            if i > 0
            else np.expand_dims(soln.x[:-1, :], 2)
        )
        u = np.dstack((u, soln.u)) if i > 0 else np.expand_dims(soln.u, 2)
        i += 1
        progress_bar.update(1)
    progress_bar.close()
    # Save the data
    data["x"] = x
    data["t"] = soln.t
    if config["ode_system"] == "pendulum":
        data["u"] = u
    else:
        pass

    with open(config["datafile_path"], "wb") as f:
        np.save(f, data)
    print(f'Data saved in {config["datafile_path"]} .')


def main():
    config = data_config.get_config()
    parser = ArgumentParser()
    parser = add_data_args(parser)
    args_config = args_to_config(parser)
    config.update(args_config)
    run(config)


# Ausgrid data selector
def select_Ausgrid_data(
    cust_id: list = np.arange(1, 51).tolist(),
    start_date: datetime = datetime(2010, 7, 1),
    end_date: datetime = datetime(2011, 6, 30),
    category: str = "GG",
    delta_t_idxs: float = 0.1,
    verbose: bool = True,
) -> dict:
    # Read the data from the csv file
    datafile_paths = [
        "data/Ausgrid/Solar home half-hour data - 1 July 2010 to 30 June 2011/2010-2011 Solar home electricity data.csv",
        "data/Ausgrid/Solar home half-hour data - 1 July 2011 to 30 June 2012/2011-2012 Solar home electricity data v2.csv",
        "data/Ausgrid/Solar home half-hour data - 1 July 2012 to 30 June 2013/2012-2013 Solar home electricity data v2.csv",
    ]
    # Concatenate the data from multiple files along the row axis
    data = pd.concat(
        [pd.read_csv(filepath, header=1) for filepath in datafile_paths], axis=0
    )
    # Convert the date column from string to datetime
    data["date"] = pd.to_datetime(data["date"])
    data_len = data.shape[0]
    idxs_query_time = (
        (data["date"] >= start_date) & (data["date"] <= end_date)
        if (start_date is not None) and (start_date is not None)
        else np.ones(data_len, dtype=bool)
    )

    if isinstance(cust_id, list):
        for i, cust_id_i in enumerate(cust_id):
            if i == 0:
                idxs_query_cust = data["Customer"] == cust_id_i
            else:
                idxs_query_cust = idxs_query_cust | (data["Customer"] == cust_id_i)
    elif isinstance(cust_id, (int, float)):
        idxs_query_cust = data["Customer"] == cust_id
    elif cust_id is None:
        idxs_query_cust = np.ones(data_len, dtype=bool)

    idxs_query_category = (
        data["Consumption Category"].str.upper() == category
        if category is not None
        else np.ones(data_len, dtype=bool)
    )
    idxs_query = idxs_query_time & idxs_query_cust & idxs_query_category
    data_select = data[idxs_query]
    data_select = (
        data_select.iloc[:, 18:39].values + 1e-7
    )  # 18:39 is the columns of the data those are mostly non-zero

    # Interpolate the data every 0.1 points
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
        print(
            "Ausgrid data with {} category selected from {} to {} for customer {} - {}.".format(
                category, start_date, end_date, cust_id[0], cust_id[-1]
            )
        )
        print(
            "Shape of the selected data: num_trajs= {}, num_time = {}".format(
                data_select_interp.shape[0], data_select_interp.shape[1]
            )
        )

    data_select_dict = dict()
    data_select_dict["x"] = np.expand_dims(data_select_interp, axis=2).transpose(
        1, 2, 0
    )
    data_select_dict["t"] = (
        t_idxs_interp * 0.5
    )  # 0.5 (hrs) is the original time interval

    return data_select_dict


if __name__ == "__main__":
    main()
