import numpy as np

ODE_SYSTEM = "pendulum"  # 'pendulum' or 'lorentz'
T_MAX = 20
STEP_SIZE = 0.01
N_SAMPLE = 10
X_INIT_PTS = np.array([[-np.pi, np.pi], [-8.0, 8.0]])  # initial states range
CONTROL_FUNC = "gaussian"  # 'designate' or 'gaussian'
DATAFILE_PATH = f"data/{ODE_SYSTEM}_ctr_{CONTROL_FUNC}_N_{N_SAMPLE}_T{T_MAX}.npy"


def get_config():
    config = {
        "ode_system": ODE_SYSTEM,
        "t_max": T_MAX,
        "step_size": STEP_SIZE,
        "n_sample": N_SAMPLE,
        "x_init_pts": X_INIT_PTS,
        "ctr_func": CONTROL_FUNC,
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
