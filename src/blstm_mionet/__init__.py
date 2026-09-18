"""B-LSTM-MIONet: Bayesian LSTM-based neural operators.

Reference implementation for Kong, Mollaali, Moya, Lu and Lin,
*B-LSTM-MIONet: Bayesian LSTM-based Neural Operators for Learning the Response
of Complex Dynamical Systems to Length-Variant Multiple Input Functions*
(arXiv:2311.16519).
"""

from blstm_mionet.config import ExperimentConfig, load_config
from blstm_mionet.models import build_model

__version__ = "1.0.0"

__all__ = ["ExperimentConfig", "__version__", "build_model", "load_config"]
