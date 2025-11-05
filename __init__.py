"""
my_projects Package

This is the main module for the animal classification project.
It provides functionality to train models, process data, etc.

Available submodules:

- `dataset`: Data loading and processing.
- `download_data`: Scripts to download the data.
- `modeling`: Model definition and training.
- `plots`: Generation of plots.
- `reduce_data`: Data dimensionality reduction.
- `scripts`: Helper scripts.
"""

# Esto le dice a pdoc qué módulos debe exportar públicamente

__all__ = [
    "dataset", 
    "download_data", 
    "modeling", 
    "plots", 
    "reduce_data", 
    "scripts"
]
