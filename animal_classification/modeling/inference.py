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
    model_path : Path, optional\n
        Path to the `.ckpt` checkpoint file. Required if `trained_model` is None.\n
    data_dir : Path, optional.\n
        Path to the dataset directory\n
    architecture : str.\n
        Model architecture name ('vgg16' or 'vgg11')\n
    trained_model : pl.LightningModule, optional\n
        An in-memory model object (used by the Dashboard). If provided, `model_path` is ignored.\n
    output_path : Path, optional\n
        Directory to save results (arrays and plots). Ignored in demo mode.\n
    batch_size : int\n
        Batch size for inference.\n
    num_workers : int\n
        Number of data loading workers.\n
    is_demo : bool\n
        If `True`, returns raw objects for the UI instead of saving to disk.\n
        
    Returns
    -------
    tuple (if is_demo=True)\n
        (probs, labels, accuracy, f1_macro, confusion_matrix_img, roc_img, calibration_img)\n
    dict (if is_demo=False).\n
        Dictionary containing metrics and the output path.\n
    """
    architecture = architecture.lower()

    if output_path:
        output_path = Path(output_path).absolute()
        output_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Setup
    if is_demo or data_dir is None:
        data_dir = get_packaged_mini_data_path()

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
        
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            ckpt_hparams = checkpoint.get("hyper_parameters", {})
            ckpt_num_classes = ckpt_hparams.get("num_classes")
            
            if ckpt_num_classes and ckpt_num_classes != num_classes:
                print(f"⚠️ Class mismatch: Checkpoint ({ckpt_num_classes}) vs Data ({num_classes}). Adjusting...")
                num_classes = ckpt_num_classes
        except Exception as e:
            print(f"⚠️ Metadata inspection failed: {e}")

        trained_model = VGGNet.load_from_checkpoint(
            checkpoint_path=model_path,
            architecture=architecture,
            num_classes=num_classes,
            strict=False 
        )

    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    # 3. Prediction Loop
    all_probs, all_labels = [], []
    print(f"🚀 Running inference on {len(val_loader.dataset)} samples...")
    
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

    # 5. Metrics
    acc = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions, average="macro")

    cm_img = plot_confusion_matrix(all_labels, predictions, architecture, output_path, is_demo)
    roc_img, auc_score = plot_roc_curves(all_labels, all_probs, num_classes, architecture, output_path, is_demo)
    cal_img = plot_calibration_curve(all_labels, all_probs, num_classes, architecture, output_path, is_demo)

    if is_demo:
        return all_probs, all_labels, acc, f1, cm_img, roc_img, cal_img

    return {"accuracy": acc, "f1_macro": f1, "auc_macro": auc_score, "output_path": output_path}

def extract_version(filepath): 
    """
    Extracts the version number from a PyTorch Lightning checkpoint filename.

    PyTorch Lightning often appends '-v1', '-v2', etc., to checkpoints if 
    multiple versions exist. This function parses that integer to help 
    identify the most recent or 'highest' versioned file.

    Parameters
    ----------
    filepath : str\n
        The full path or filename of the checkpoint.

    Returns
    -------
    int\n
        The version number extracted (e.g., 2 from 'vgg16-v2.ckpt'). 
        Returns 0 if no version pattern is found.
    """
    match = re.search(r'-v(\d+).ckpt$', filepath)
    return int(match.group(1)) if match else 0

def main(): 
    parser = argparse.ArgumentParser(description="Run inference on trained animal classification models.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATTERN) 
    parser.add_argument("--dataset-choice", type=str, default="mini", choices=["mini", "full"])
    parser.add_argument("--data-dir", type=str, default=None) 
    parser.add_argument("--architecture", type=str.lower, default="vgg16", choices=["vgg16", "vgg11"]) 
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH) 
    parser.add_argument("--batch-size", type=int, default=16) 
    parser.add_argument("--num-workers", type=int, default=2) 
    args = parser.parse_args()

    # 1. Resolve Data Path
    if args.data_dir:
        data_path = Path(args.data_dir)
    elif args.dataset_choice == "mini":
        data_path = get_packaged_mini_data_path()
    else: 
        data_path = Path("data/animals/animals")

    # 2. Locate and Filter Checkpoint
    if "*" in args.model_path:
        model_files = glob.glob(args.model_path)
        
        # --- FIX: FILTER BY ARCHITECTURE ---
        # Only keep checkpoints that have 'vgg16' or 'vgg11' in their filename
        filtered_files = [f for f in model_files if args.architecture in Path(f).name.lower()]
        
        if not filtered_files:
            print(f"⚠️ No checkpoints found for {args.architecture}. Checking all files in {args.model_path}...")
            filtered_files = model_files # Fallback to any file if specific arch not found
            
        if not filtered_files:
            raise FileNotFoundError(f"No checkpoints found matching: {args.model_path}")
        
        # Pick the best among the filtered files
        try:
            target_checkpoint = max(filtered_files, key=extract_version)
        except Exception:
            filtered_files.sort(key=os.path.getmtime, reverse=True)
            target_checkpoint = filtered_files[0]
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
    print(f"Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()