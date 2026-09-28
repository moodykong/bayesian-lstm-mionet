"""Global seeding.

The published experiments were run with seed 999.  The sub-sequence sampling in
:mod:`blstm_mionet.data.masking` uses its own generator seeded with the same
value, so it is reproducible independently of the global state set here.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_SEED = 999


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed NumPy and PyTorch (CPU and all CUDA devices)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
