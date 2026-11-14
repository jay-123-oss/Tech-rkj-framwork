"""
tech-rkj: High-Performance AVX2 Closed-Form & Continual Learning Subspace Engine in Rust.
"""

from .classifier import SubspaceClassifier, AICoreClassifier
from .regressor import SubspaceRegressor, AICoreRegressor

__version__ = "0.1.0"
__all__ = [
    "SubspaceClassifier",
    "AICoreClassifier",
    "SubspaceRegressor",
    "AICoreRegressor",
    "__version__",
]
