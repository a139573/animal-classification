import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from ..dataset import AnimalsDataModule
from ..modeling.train import VGGNet
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, RocCurveDisplay
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

"""
Model Inference and Evaluation Script.

This script runs inference using a trained model checkpoint on a specified
dataset. It calculates key performance metrics, saves raw predictions,
and generates diagnostic plots such as the confusion matrix, ROC curve,
and calibration plot.
"""

def run_inference(model_path: Path, data_dir: Path, architecture: str, output_path: Path, batch_size: int = 16):
    """
    Runs model inference, calculates metrics, and saves plots.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the saved model state dictionary (.pth file).
    data_dir : pathlib.Path
        Path to the root data directory (e.g., '.../mini_animals/animals').
    architecture : str
        The model architecture name (e.g., "vgg16") to instantiate.
    output_path : pathlib.Path
        Directory where predictions (.npy) and plots (.png) will be saved.
    batch_size : int, optional
        Batch size for the validation DataLoader (default is 16).

    Returns
    -------
    dict
        A dictionary containing key validation metrics:
        - "val_acc" (float): Validation accuracy.
        - "f1_score" (float): Macro F1-score.
        - "confusion_matrix" (np.ndarray): The confusion matrix.
    """
    print(f"\n🔍 Loading model: {model_path}")
    print(f"📁 Using dataset: {data_dir}")
    print(f"🧠 Architecture: {architecture}")

    # --- Ensure absolute path exists ---
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Plots and predictions will be saved to: {output_path}")

    # --- Setup data module ---
    data_module = AnimalsDataModule(data_dir=data_dir, batch_size=batch_size)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # --- Load model ---
    model = VGGNet(architecture=architecture, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
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
    np.save(output_path / f"{architecture}_val_probs.npy", all_probs)
    np.save(output_path / f"{architecture}_val_labels.npy", all_labels)

    # --- Confusion matrix ---
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"{architecture}_confusion_matrix.png")
    plt.close()

    # --- ROC curve ---
    try:
        plt.figure()
        y_true_onehot = np.eye(num_classes)[all_labels]
        RocCurveDisplay.from_predictions(y_true_onehot.ravel(), all_probs.ravel())
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(output_path / f"{architecture}_roc_curve.png")
        plt.close()
    except Exception as e:
        print("⚠️ ROC plot failed:", e)

    # --- Calibration plot (top-class correctness) ---
    try:
        plt.figure(figsize=(6,6))
        y_true = (preds == all_labels).astype(int)  # 1 if top prediction correct
        y_prob = all_probs.max(axis=1)              # probability of top prediction
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.title("Calibration Plot (Top-Class Correctness)")
        plt.tight_layout()
        plt.savefig(output_path / f"{architecture}_calibration.png")
        plt.close()
    except Exception as e:
        print("⚠️ Calibration plot failed:", e)

    print(f"\n✅ Saved validation predictions and plots to: {output_path}")
    print(f"Validation Accuracy: {acc:.3f}, F1-score: {f1:.3f}")
    return {"val_acc": acc, "f1_score": f1, "confusion_matrix": cm}
