import numpy as np
from datetime import datetime

# Define the dynamic system
ODE_SYSTEM = "Ausgrid"  # 'pendulum' or 'lorentz' or 'Ausgrid'

# For pendulum system or lorentz system
T_MAX = 10
STEP_SIZE = 0.01
N_SAMPLE = 10
X_INIT_PTS = np.array([[-np.pi, np.pi], [-8.0, 8.0]])  # initial states range
CONTROL_FUNC = "gaussian"  # 'designate' or 'gaussian'

# For Ausgrid system
CUSTOMER_ID = np.arange(1, 51).tolist()
START_DATE = datetime(2010, 7, 1)
END_DATE = datetime(2011, 6, 30)
CATEGORY = "GG"
DELTA_T_IDXS = 0.1
VERBOSE = True

if ODE_SYSTEM == "pendulum" or ODE_SYSTEM == "lorentz":
    DATAFILE_PATH = (
        f"data/{ODE_SYSTEM}_ctr_{CONTROL_FUNC}_N_{N_SAMPLE}_T{T_MAX}_h{STEP_SIZE}.npy"
    )
elif ODE_SYSTEM == "Ausgrid":
    DATAFILE_PATH = "data/Ausgrid_select.npy"
else:
    DATAFILE_PATH = None


def get_config():
    config = {
        "ode_system": ODE_SYSTEM,
        "t_max": T_MAX,
        "step_size": STEP_SIZE,
        "n_sample": N_SAMPLE,
        "x_init_pts": X_INIT_PTS,
        "ctr_func": CONTROL_FUNC,
        "Ausgrid_customer_id": CUSTOMER_ID,
        "Ausgrid_start_date": START_DATE,
        "Ausgrid_end_date": END_DATE,
        "Ausgrid_category": CATEGORY,
        "Ausgrid_delta_t_idxs": DELTA_T_IDXS,
        "verbose": VERBOSE,
        "datafile_path": DATAFILE_PATH,
    }
    return config


# Define the pendulum's vector field
def pendulum(x, u):
    m = 1
    l = 1
    I = (1 / 12) * m * (l**2)
    b = 0.01
    g = 9.81
    theta, omega = x
    f_theta = omega
    f_omega = (1 / (0.25 * m * (l**2) + I)) * (
        u - b * omega - 0.5 * m * l * g * np.sin(theta)
    )
    return np.array([f_theta, f_omega])


# Define the Lorentz system
def lorentz(x, u):
    sigma = 10
    rho = 28
    beta = 8 / 3
    x, y, z = x
    f_x = sigma * (y - x)
    f_y = x * (rho - z) - y
    f_z = x * y - beta * z
    return np.array([f_x, f_y, f_z])


# Define the control
def control_formula(t, x):
    theta = x[0]
    theta_dot = x[1]
    formula = -0.80 * theta_dot
    return formula


# Define the control
def control_grf(func):
    def control(t, x):
        theta = x[0]
        theta_dot = x[1]
        return func(theta_dot)

    return control
