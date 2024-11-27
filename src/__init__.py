"""Signal Analyzer package for ATF file analysis."""

from .signal_processor import ProcessingParams, SignalProcessor
from .signal_visualizer import SignalVisualizer

__version__ = "0.1.0"

# Export main classes for easier imports
__all__ = [
    "SignalProcessor",
    "ProcessingParams",
    "SignalVisualizer",
]
