import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from ..dataset import AnimalsDataModule
from ..modeling.train import VGGNet
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, RocCurveDisplay, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from PIL import Image
import io

"""
Model Inference and Evaluation Script.

This script runs inference using a trained model checkpoint on a specified
dataset. It calculates key performance metrics, saves raw predictions,
and generates diagnostic plots such as the confusion matrix, ROC curve,
and calibration plot.
"""

# Helper to convert matplotlib figure to PIL Image
def fig_to_image():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def run_inference(model_path: Path = None, data_dir: Path = None, architecture: str = None, trained_model: pl.LightningModule = None, output_path: Path = None, batch_size: int = 16, num_workers: int = 2, is_demo: bool = False):
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
    if model_path is not None:
        print(f"\n🔍 Loading model: {model_path}")
    else:
        print("Model loaded from training")
    print(f"📁 Using dataset: {data_dir}")
    print(f"🧠 Architecture: {architecture}")

    # --- Ensure absolute path exists ---
    if output_path is not None: # if it is None, we don't want to store (demo)
        output_path = Path(output_path).absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Plots and predictions will be saved to: {output_path}")

    # --- Setup DataModule ---
    data_module = AnimalsDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # --- Load model ---
    if trained_model is None:
        trained_model = VGGNet(architecture=architecture, num_classes=num_classes, pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")
        trained_model.load_state_dict(state_dict)
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    # --- Collect predictions ---
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = trained_model(xb)
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
    if not is_demo:
        np.save(output_path / f"{architecture}_val_probs.npy", all_probs)
        np.save(output_path / f"{architecture}_val_labels.npy", all_labels)
    else:
        # Keep in memory for demo
        val_probs = all_probs
        val_labels = all_labels

    # --- Confusion matrix ---
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_img = fig_to_image() if is_demo else plt.savefig(output_path / f"{architecture}_confusion_matrix.png")

    # --- Macro-average ROC curve ---
    try:
        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(all_labels)
        y_true_onehot = np.eye(num_classes)[all_labels]

        # Keep only the columns corresponding to present classes
        y_true_onehot_present = y_true_onehot[:, unique_labels]
        all_probs_present = all_probs[:, unique_labels]

        auc_score = roc_auc_score(
            y_true_onehot_present,
            all_probs_present,
            average="macro",
            multi_class="ovr"
        )
        print(f"AUC score is {auc_score:.3f}")


        # Compute per-class ROC points and aggregate via interpolation
        fpr_interp = np.linspace(0, 1, 100)
        tpr_all = []

        for i in range(num_classes):
            y_true_i = y_true_onehot[:, i]
            y_prob_i = all_probs[:, i]

            # Skip class if it has only one label present
            if np.sum(y_true_i) == 0 or np.sum(y_true_i) == len(y_true_i):
                continue

            display = RocCurveDisplay.from_predictions(
                y_true_i,
                y_prob_i,
                name=None,
                color='blue',
                alpha=0.1
            )
            # Interpolate TPR at fixed FPR points
            tpr_interp = np.interp(fpr_interp, display.fpr, display.tpr)
            tpr_all.append(tpr_interp)

        # Average TPR across classes
        if tpr_all:
            tpr_mean = np.mean(tpr_all, axis=0)
            plt.plot(fpr_interp, tpr_mean, color='red', label=f'Macro-average ROC (AUC={auc_score:.3f})')

        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (macro-average)")
        plt.legend()
        plt.tight_layout()
        roc_img = fig_to_image() if is_demo else plt.savefig(output_path / f"{architecture}_roc_curve.png")
    except Exception as e:
        print("⚠️ ROC plot failed:", e)
        roc_img = None

    # --- Average Calibration Curve ---
    try:
        plt.figure(figsize=(6, 6))
        fixed_bins = np.linspace(0, 1, 10)
        prob_true_all = []

        for i in range(num_classes):
            y_true_i = y_true_onehot[:, i]
            y_prob_i = all_probs[:, i]

            # Skip classes without positive samples
            if np.sum(y_true_i) == 0:
                continue

            prob_true, prob_pred = calibration_curve(y_true_i, y_prob_i, n_bins=10)
            # Interpolate to fixed bins
            prob_true_interp = np.interp(fixed_bins, prob_pred, prob_true)
            prob_true_all.append(prob_true_interp)

        if prob_true_all:
            prob_true_mean = np.mean(prob_true_all, axis=0)
            plt.plot(fixed_bins, prob_true_mean, marker="o", color="blue", label="Average calibration")

        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.title("Multiclass Calibration Plot (average)")
        plt.legend()
        plt.tight_layout()
        cal_img = fig_to_image() if is_demo else plt.savefig(output_path / f"{architecture}_calibration.png")
    except Exception as e:
        print("⚠️ Calibration plot failed:", e)
        cal_img = None




    # --- Now you can return these images directly to Gradio ---
    if is_demo:
        return val_probs, val_labels, acc, f1, cm_img, roc_img, cal_img
    else:
        return {"output_path": output_path, "val_acc": acc, "f1_score": f1, "confusion_matrix": cm}
