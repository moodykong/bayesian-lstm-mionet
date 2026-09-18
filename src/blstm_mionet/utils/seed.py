"""Global seeding.

The published experiments were run with seed 999; the dataset preparation
routines in :mod:`blstm_mionet.data.masking` re-seed with the same value so
that the sub-sequence sampling is reproducible independently of the caller.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_SEED = 999


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed NumPy and PyTorch (CPU and all CUDA devices)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
