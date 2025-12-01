"""
This package contains all modules related to Machine Learning
for the project.

This includes the model architecture definitions, the training 
process, and functions for performing inference.

Submodules:

* **inference**: Functions for loading a trained model and 
                 making predictions on new data.
* **train**: Contains the pipeline and loops for training
             and evaluating the models.
"""

# --- Recommended: Define __all__ ---
# This tells pdoc which modules to document

__all__ = [
    "inference",
    "train"
]