"""
Animal Classification Project (animal_classification)

This is the main package for the project. It contains all modules
required for the Machine Learning pipeline.

### Submodules

* **dataset**: Functions for loading and processing the dataset.
* **download_data**: Script to download the initial data.
* **modeling**: Model definition, training, and evaluation.
* **plots**: Functions for generating visualizations.
* **reduce_data**: Scripts for dimensionality reduction.
* **scripts**: Other helper scripts.
"""

__docformat__ = "numpy"

# This tells pdoc which modules to include in the documentation
# Make sure these match your .py files
__all__ = [
    "dataset", 
    "download_data", 
    "modeling", 
    "plots", 
    "reduce_data", 
    "scripts"
]