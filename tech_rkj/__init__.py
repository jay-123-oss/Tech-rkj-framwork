"""
tech-rkj: High-Performance AVX2/NEON Closed-Form & Continual Learning Subspace Framework.
"""

from .classifier import SubspaceClassifier, AICoreClassifier
from .regressor import SubspaceRegressor, AICoreRegressor
from .persistence import save, load

try:
    from .torch import TorchSubspaceHead
except ImportError:
    TorchSubspaceHead = None  # type: ignore

__version__ = "0.2.0"
__all__ = [
    "SubspaceClassifier",
    "AICoreClassifier",
    "SubspaceRegressor",
    "AICoreRegressor",
    "TorchSubspaceHead",
    "save",
    "load",
    "__version__",
]
