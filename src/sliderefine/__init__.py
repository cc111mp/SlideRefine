"""SlideRefine research workspace; the preserved tsclahe backend is version 0.2.0."""
__version__ = "0.2.0"
from .contracts import Region, RegionPixels
from .registry import available_methods, require_method, MethodUnavailableError
__all__ = ["Region", "RegionPixels", "available_methods", "require_method", "MethodUnavailableError"]
