import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from ..dataset import AnimalsDataModule
from ..modeling.train import VGGNet
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, RocCurveDisplay
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import tempfile
import argparse


"""
Model Inference and Evaluation Script.

This script runs inference using a trained model checkpoint on a specified
dataset. It calculates key performance metrics, saves raw predictions,
and generates diagnostic plots such as the confusion matrix, ROC curve,
and calibration plot.
"""


import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, RocCurveDisplay
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import tempfile

from ..dataset import AnimalsDataModule
from ..modeling.train import VGGNet


def run_inference(
    trained_model=None,
    model_path: Path = None,
    data_dir: Path = None,
    architecture: str = None,
    output_path: Path = None,
    batch_size: int = 16,
    is_demo: bool = False,
):
    """
    Run model inference on the validation dataset.  
    Uses an in-memory model if provided, otherwise loads from checkpoint.

    Parameters
    ----------
    trained_model : torch.nn.Module, optional
        A trained model instance already loaded in memory.
    model_path : Path, optional
        Path to a saved model state dictionary (.pth).
    data_dir : Path, optional
        Path to the dataset directory.
    architecture : str, optional
        Model architecture name ("vgg16" or "vgg11").
    output_path : Path, optional
        Where to save predictions and plots.
    batch_size : int, optional
        Validation batch size.
    is_demo : bool, optional
        Whether to use a temporary folder for outputs (auto-cleaned).

    Returns
    -------
    dict
        { "val_acc": float, "f1_score": float, "confusion_matrix": np.ndarray }
    """

    # --- Handle defaults / validation ---
    if trained_model is None and (model_path is None or architecture is None):
        raise ValueError(
            "You must provide either a `trained_model` instance or both `model_path` and `architecture`."
        )

    if data_dir is None:
        data_dir = Path("data/mini_animals/animals")
        print(f"⚙️ Defaulting data_dir to: {data_dir}")

    print("\n🔍 Starting inference")
    print(f"📁 Dataset: {data_dir}")
    print(f"🧠 Architecture: {architecture or 'provided model'}")

    # --- Handle output directory ---
    if is_demo:
        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name)
        print(f"⚙️ Running in demo mode — temporary dir: {output_path}")
    else:
        output_path = Path(output_path or "inference_outputs").absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {output_path}")

    # --- Setup DataModule ---
    data_module = AnimalsDataModule(data_dir=data_dir, batch_size=batch_size)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # --- Load or use provided model ---
    if trained_model is not None:
        print("🧩 Using provided trained model instance.")
        model = trained_model
        model.eval()
    else:
        print(f"📦 Loading model from checkpoint: {model_path}")
        model = VGGNet(architecture=architecture, num_classes=num_classes, pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    # --- Move to device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Collect predictions ---
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = all_probs.argmax(axis=1)

    # --- Compute metrics ---
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds)

    # --- Save predictions ---
    np.save(output_path / "val_probs.npy", all_probs)
    np.save(output_path / "val_labels.npy", all_labels)

    # --- Confusion matrix ---
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

    # --- ROC curve ---
    try:
        plt.figure()
        y_true_onehot = np.eye(num_classes)[all_labels]
        RocCurveDisplay.from_predictions(y_true_onehot.ravel(), all_probs.ravel())
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(output_path / "roc_curve.png")
        plt.close()
    except Exception as e:
        print("⚠️ ROC plot failed:", e)

    # --- Calibration plot ---
    try:
        plt.figure(figsize=(6, 6))
        y_true = (preds == all_labels).astype(int)
        y_prob = all_probs.max(axis=1)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.title("Calibration Plot")
        plt.tight_layout()
        plt.savefig(output_path / "calibration_plot.png")
        plt.close()
    except Exception as e:
        print("⚠️ Calibration plot failed:", e)

    print(f"\n✅ Saved predictions and plots to: {output_path}")
    print(f"Validation Accuracy: {acc:.3f}, F1-score: {f1:.3f}")

    results = {
        "val_acc": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "output_path": str(output_path),
    }

    if is_demo:
        temp_dir.cleanup()
        print("🧹 Cleaned up temporary demo files.")

    return results



# -------------------------------------------------------------------
# CLI USAGE
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference and evaluation.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (.pth).")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset (e.g. data/mini_animals/animals).")
    parser.add_argument("--architecture", default="vgg16", choices=["vgg16", "vgg11"])
    parser.add_argument("--output-path", default="inference_outputs", help="Output directory (ignored if --demo).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (temporary files only).")

    args = parser.parse_args()

    run_inference(
        model_path=Path(args.model_path),
        data_dir=Path(args.data_dir),
        architecture=args.architecture,
        output_path=Path(args.output_path),
        batch_size=args.batch_size,
        is_demo=args.demo,
    )