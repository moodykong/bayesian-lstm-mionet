import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from utils.math_utils import grf_1d, integrate, runge_kutta
from utils.args_parser import add_data_args, args_to_config
import config.data_config as conf
from tqdm import tqdm
from argparse import ArgumentParser


def run(config):
    if config["ode_system"] == "pendulum":
        ode_system = conf.pendulum
    elif config["ode_system"] == "lorentz":
        ode_system = conf.lorentz
    else:
        raise ValueError("Invalid ode_system.")
    # Define the time parameters
    T = config["t_max"]
    h = config["step_size"]
    N_time = int(T / h)
    N_sample = config["n_sample"]
    # Define the initial states
    x_init_pts = config["x_init_pts"]
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
            control = conf.control_formula
        elif config["ctr_func"] == "gaussian":
            grf = grf_1d(a=0.01)
            control = conf.control_grf(grf)
        else:
            raise ValueError("Invalid ctr_func.")

        soln = integrate(runge_kutta, ode_system, control, x_init[i], h, N_time)

        if np.isnan(soln.x).sum() + np.isnan(soln.u).sum() > 0:
            continue
        x = np.dstack((x, soln.x[:-1, :])) if i > 0 else soln.x[:-1, :]
        u = np.dstack((u, soln.u)) if i > 0 else soln.u
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
    config = conf.get_config()
    parser = ArgumentParser()
    parser = add_data_args(parser)
    args_config = args_to_config(parser)
    config.update(args_config)
    run(config)


if __name__ == "__main__":
    main()
