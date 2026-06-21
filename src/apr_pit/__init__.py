"""APR-PiT: residual-adaptive physics-informed Transformer."""

from .config import load_config
from .model import APRPiT
from .physics import TunnelFirePhysics
from .sampling import APRController, CollocationPool

__all__ = [
    "APRPiT",
    "APRController",
    "CollocationPool",
    "TunnelFirePhysics",
    "load_config",
]

__version__ = "0.1.0"

