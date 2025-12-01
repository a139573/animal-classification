"""
Helper scripts and entry points for the Animal Classification project.

This package serves as a container for executable scripts and dashboard interfaces
used to interact with the trained models and datasets. It is designed to be
used primarily via command-line entry points or the interactive web UI.

Available Submodules:

* **dashboard**: Implements the interactive Gradio dashboard for data reduction, training,
    and inference visualization.
"""

# --- Recommended: Define __all__ ---
# This tells pdoc which modules from 'scripts' to display

__all__ = [
    "dashboard"
]