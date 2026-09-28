"""Device resolution.

The original code kept a module level global ``device`` that was mutated by
``init_gpu``.  Here the device is resolved once and passed explicitly to every
function that needs it.
"""

from __future__ import annotations

import torch

DeviceSpec = int | str
"""A GPU index or ``"cpu"``.

``"parallel"`` is still accepted, for configuration files written for the
original code, as an alias of the current CUDA device.  The model is *not*
replicated across GPUs: the operators are small enough for a single device.
"""


def resolve_device(spec: DeviceSpec = "cpu", verbose: bool = False) -> torch.device:
    """Translate a configuration device specification into a ``torch.device``.

    Parameters
    ----------
    spec:
        ``int``  -> ``cuda:<spec>`` when CUDA is available, otherwise CPU.
        ``"parallel"`` -> the current CUDA device when CUDA is available,
        otherwise CPU (a legacy alias; no multi-GPU replication).  ``"cpu"`` (or anything else) -> CPU.
    verbose:
        Print the resolved device, mirroring the message of the original
        ``utils.torch_utils.init_gpu``.
    """
    if isinstance(spec, bool):  # ``bool`` is a subclass of ``int``
        raise TypeError("device specification must be a GPU index or 'cpu'")

    if torch.cuda.is_available() and isinstance(spec, int):
        device = torch.device(f"cuda:{spec}")
        if verbose:
            print(f"Using GPU ID {spec}.")
    elif torch.cuda.is_available() and spec == "parallel":
        device = torch.device("cuda")
        if verbose:
            print("Using the current CUDA device.")
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU is not used, defaulting to CPU.")
    return device


def parse_device_spec(value: str) -> DeviceSpec:
    """Parse a command line device string into a :data:`DeviceSpec`."""
    if value in ("parallel", "cpu"):
        return value
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - argparse formats the message
        raise ValueError(
            f"invalid device {value!r}: expected a GPU index or 'cpu'"
        ) from exc
