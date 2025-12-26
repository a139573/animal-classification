"""
# 🔍 Model Inference & Evaluation

This module executes the inference pipeline using a trained model checkpoint.

It handles:
1.  **Data Loading:** Prepares the validation/test set.
2.  **Model Loading:** Instantiates the VGG architecture and loads weights.
3.  **Prediction:** Runs the forward pass to get probabilities.
4.  **Reporting:** Calculates metrics (Accuracy, F1, AUC) and generates plots.

## 💻 CLI Usage
Run evaluation on the latest checkpoint:
```bash
python -m animal_classification.modeling.inference --architecture vgg16 --dataset-choice mini
```
Run on a specific model file using the full dataset:

```bash
python -m animal_classification.modeling.inference --model-path models/vgg16-best.ckpt --dataset-choice full
```
"""

import argparse
import glob
import re
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

# --- Local Project Imports ---
from animal_classification.modeling.architecture import VGGNet
from animal_classification.modeling.metrics import ( plot_calibration_curve, plot_confusion_matrix, plot_roc_curves, )
from animal_classification.preprocessing import AnimalsDataModule
from animal_classification.utils import get_packaged_mini_data_path

# Configuration Defaults
DEFAULT_MODEL_PATTERN = "models/*.ckpt"
DEFAULT_OUTPUT_PATH = "reports/inference"

def run_inference( model_path: Path = None, data_dir: Path = None, architecture: str = "vgg16", trained_model: pl.LightningModule = None, output_path: Path = None, batch_size: int = 16, num_workers: int = 2, is_demo: bool = False):
    """ Executes the full inference pipeline on the validation dataset.

    Parameters
    ----------
    model_path : Path, optional.
        Path to the `.ckpt` checkpoint file. Required if `trained_model` is None.

    data_dir : Path, optional.
        Path to the dataset directory.

    architecture : str.
        Model architecture name ('vgg16' or 'vgg11').

    trained_model : pl.LightningModule, optional
        An in-memory model object (used by the Dashboard). If provided, `model_path` is ignored.

    output_path : Path, optional.
        Directory to save results (arrays and plots). Ignored in demo mode.

    batch_size : int.
        Batch size for inference.

    num_workers : int.
        Number of data loading workers.

    is_demo : bool.
        If `True`, returns raw objects for the UI instead of saving to disk.

    Returns
    -------
    tuple (if is_demo=True).
        (probs, labels, accuracy, f1_macro, confusion_matrix_img, roc_img, calibration_img)

    dict (if is_demo=False).
        Dictionary containing metrics and the output path.
    """
    architecture = architecture.lower()

    if output_path:
        output_path = Path(output_path).absolute()
        output_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Setup
    if is_demo:
        # Use the robust utility to find data in the package or locally
        data_dir = get_packaged_mini_data_path()

    if data_dir is None:
        raise ValueError("Data directory must be provided if not in demo mode.")

    data_module = AnimalsDataModule(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # 2. Model Initialization
    if trained_model is None:
        print(f"🔄 Loading model from checkpoint: {model_path}")
        
        # Check for class count mismatch between checkpoint and current data_dir
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            ckpt_hparams = checkpoint.get("hyper_parameters", {})
            ckpt_num_classes = ckpt_hparams.get("num_classes")
            
            if ckpt_num_classes and ckpt_num_classes != num_classes:
                print(f"⚠️ Warning: Checkpoint has {ckpt_num_classes} classes, but current data has {num_classes}.")
                print(f"🔄 Adjusting model to {ckpt_num_classes} classes to prevent size mismatch.")
                num_classes = ckpt_num_classes
        except Exception as e:
            print(f"⚠️ Could not inspect checkpoint metadata: {e}")

        # Use Lightning's native loader which handles strict matching automatically
        trained_model = VGGNet.load_from_checkpoint(
            checkpoint_path=model_path,
            architecture=architecture,
            num_classes=num_classes,
            strict=False # Allow minor mismatches
        )

    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    # 3. Prediction Loop
    all_probs, all_labels = [], []
    val_dataset = val_loader.dataset

    try:
        if hasattr(val_dataset, 'samples'): # Standard ImageFolder
            all_paths = [str(Path(s[0]).resolve()) for s in val_dataset.samples]
        elif hasattr(val_dataset, 'dataset') and hasattr(val_dataset, 'indices'): # Subset
            all_paths = [str(Path(val_dataset.dataset.samples[i][0]).resolve()) for i in val_dataset.indices]
        else:
            all_paths = []
    except Exception as e:
        print(f"⚠️ Warning: Image path extraction failed: {e}")
        all_paths = []

    print(f"🚀 Running inference on {len(val_dataset)} samples...")
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            outputs = trained_model(batch_images.to(device))
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(batch_labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    predictions = all_probs.argmax(axis=1)

    # 4. Persistence
    if not is_demo and output_path:
        np.save(output_path / "val_probs.npy", all_probs)
        np.save(output_path / "val_labels.npy", all_labels)
        if all_paths:
            np.save(output_path / "val_paths.npy", np.array(all_paths))

    # 5. Metrics and Diagnostics
    metrics = {
        "accuracy": accuracy_score(all_labels, predictions),
        "f1_macro": f1_score(all_labels, predictions, average="macro")
    }

    cm_img = plot_confusion_matrix(all_labels, predictions, architecture, output_path, is_demo)
    roc_img, auc_score = plot_roc_curves(all_labels, all_probs, num_classes, architecture, output_path, is_demo)
    cal_img = plot_calibration_curve(all_labels, all_probs, num_classes, architecture, output_path, is_demo)

    metrics["auc_macro"] = auc_score

    if is_demo:
        return all_probs, all_labels, metrics["accuracy"], metrics["f1_macro"], cm_img, roc_img, cal_img

    return {**metrics, "output_path": output_path}

def extract_version(filepath): 
    """Parses version numbers from checkpoint filenames (e.g., v2 -> 2).""" 
    match = re.search(r'-v(\d+).ckpt$', filepath)
    return int(match.group(1)) if match else 0

def main(): 
    parser = argparse.ArgumentParser(description="Run inference on trained animal classification models.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATTERN, help="Path to checkpoint or glob pattern.") 
    parser.add_argument("--dataset-choice", type=str, default="mini", choices=["mini", "full"], help="Choice of dataset (mini or full).")
    parser.add_argument("--data-dir", type=str, default=None, help="Manual override for the data directory.") 
    parser.add_argument("--architecture", type=str.lower, default="vgg16", choices=["vgg16", "vgg11"], help="Model architecture.") 
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH, help="Output directory for reports.") 
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.") 
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers.") 
    args = parser.parse_args()

    # 1. Resolve Data Path
    if args.data_dir:
        data_path = Path(args.data_dir)
    elif args.dataset_choice == "mini":
        data_path = get_packaged_mini_data_path()
    else: # full
        # Assumes standard structure created by the download script
        data_path = Path("data/animals/animals")

    if not data_path.exists():
        raise FileNotFoundError(f"Selected data directory not found: {data_path}")

    # 2. Locate Checkpoint
    if "*" in args.model_path:
        model_files = glob.glob(args.model_path)
        if not model_files:
            raise FileNotFoundError(f"No checkpoints found matching: {args.model_path}")
        try:
            target_checkpoint = max(model_files, key=extract_version)
        except Exception:
            model_files.sort(key=os.path.getmtime, reverse=True)
            target_checkpoint = model_files[0]
    else:
        target_checkpoint = args.model_path

    # 3. Execute
    results = run_inference(
        model_path=Path(target_checkpoint),
        data_dir=data_path,
        architecture=args.architecture,
        output_path=Path(args.output_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"\n✅ Evaluation complete for: {Path(target_checkpoint).name}")
    print(f"📊 Dataset: {args.dataset_choice} ({data_path})")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_macro']:.4f}")
    print(f"AUC Score: {results['auc_macro']:.4f}")
    print(f"Results saved to: {results['output_path']}")

if __name__ == "__main__":
    main()