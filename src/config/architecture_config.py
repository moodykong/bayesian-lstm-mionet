## Architecture configurations ##
BRANCH_STATE = {"width": 200, "depth": 3, "activation": "relu"}
BRANCH_MEMORY = {
    "width": 200,
    "depth": 3,
    "lstm_size": 100,
    "lstm_layer_num": 2,
    "activation": "relu",
}
TRUNK = {"width": 200, "depth": 3, "activation": "relu"}


def get_config():
    config = {
        "branch_state_width": BRANCH_STATE["width"],
        "branch_state_depth": BRANCH_STATE["depth"],
        "branch_state_activation": BRANCH_STATE["activation"],
        "branch_memory_width": BRANCH_MEMORY["width"],
        "branch_memory_depth": BRANCH_MEMORY["depth"],
        "branch_memory_lstm_size": BRANCH_MEMORY["lstm_size"],
        "branch_memory_lstm_layer_num": BRANCH_MEMORY["lstm_layer_num"],
        "branch_memory_activation": BRANCH_MEMORY["activation"],
        "trunk_width": TRUNK["width"],
        "trunk_depth": TRUNK["depth"],
        "trunk_activation": TRUNK["activation"],
    }
    return config
