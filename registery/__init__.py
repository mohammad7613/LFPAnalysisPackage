# registry/__init__.py
from .base import REGISTRIES, register, get_registry
from .preprocessors import PREPROCESSOR_REGISTRY
from .features import FEATURE_REGISTRY
from .visualizers import VISUALIZER_REGISTRY

__all__ = [
    "REGISTRIES",
    "register",
    "get_registry",
    "PREPROCESSOR_REGISTRY",
    "FEATURE_REGISTRY",
    "VISUALIZER_REGISTRY",
]
