import numpy as np

###################################
# Define training parameters
###################################
n_ensemble = 400
train_params = dict()
train_params["use grad norm"] = True
train_params["grad norm"] = 0.5e2
train_params["print every"] = 1

###################################
# Define model exploit parameters
###################################
level = 2.0
v_exploit = 0.1

eta = 1e-4

exploit_params = dict()
exploit_params["tau"] = 1e-7
exploit_params["eta"] = eta
beta_exploit = 0.5 * v_exploit * exploit_params["eta"]
exploit_params["alpha"] = 0.1
self_sigma_exploit = np.sqrt(
    2.0 * (exploit_params["alpha"] - beta_exploit) * exploit_params["eta"]
)
exploit_params["scale"] = self_sigma_exploit * np.sqrt(exploit_params["tau"])
exploit_params["sigma"] = 2 * (level**2)

###################################
# Define model explore parameters
###################################
v_explore = 0.1

explore_params = dict()
explore_params["tau"] = 2e-7
explore_params["eta"] = eta
beta_explore = 0.5 * v_explore * explore_params["eta"]
explore_params["alpha"] = 0.1
self_sigma_explore = np.sqrt(
    2.0 * (explore_params["alpha"] - beta_explore) * explore_params["eta"]
)
explore_params["scale"] = self_sigma_explore * np.sqrt(explore_params["tau"])
explore_params["sigma"] = 2 * (level**2)

###################################
# Define replica parameters
###################################
replica_params = dict()
replica_params["tau delta"] = (1 / exploit_params["tau"]) - (1 / explore_params["tau"])
replica_params["sigma uniform"] = 1.0
replica_params["F adj"] = 90000000


###################################
# Return parameters
###################################
def get_config():
    config = {
        "n_ensemble": n_ensemble,
        "train_params": train_params,
        "exploit_params": exploit_params,
        "explore_params": explore_params,
        "replica_params": replica_params,
    }
    return config
