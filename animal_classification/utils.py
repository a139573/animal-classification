# animal_classification/utils.py
import torch
import importlib.resources as pkg_resources
from pathlib import Path

# --- Safe device selection, avoid CUDA issues ---
def get_accelerator():
    """
    Determines the hardware accelerator to use (GPU or CPU).

    This function checks if CUDA is available and performs a sanity check
    by attempting to access device properties. If CUDA is reported as
    available but fails the check (e.g., due to driver issues), it gracefully
    falls back to CPU and prints a warning.

    Returns
    -------
    str
        Returns "gpu" if a CUDA device is available and functioning,
        otherwise returns "cpu".
    """
    # CUDA available?
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            torch.cuda.get_device_properties(0)
            return "gpu"
        except Exception as e:
            print(f"⚠️ CUDA available but broken: {e}")
            return "cpu"
    # no CUDA at all
    return "cpu"

def get_packaged_mini_data_path():
    """
    Locates the 'mini_animals' dataset directory.

    This function attempts to find the dataset inside the installed package
    (site-packages) using `importlib.resources`. If the package is not
    installed (e.g., during local development), it falls back to the
    relative local path.

    Returns
    -------
    pathlib.Path
        The path object pointing to the 'mini_animals/animals' directory.
    """
    # Use files() to correctly reference the path inside site-packages
    try:
        return pkg_resources.files('animal_classification').joinpath('data/mini_animals/animals')
    except Exception:
        # Fallback for local dev setup where package is not installed as a wheel
        return Path("data/mini_animals/animals")