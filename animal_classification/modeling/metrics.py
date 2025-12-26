"""
# 📊 Model Visualization & Metrics

This module contains helper functions to generate diagnostic plots for model evaluation.
It supports generating:
- **Confusion Matrices**: To see where the model is confusing classes.
- **ROC Curves**: To evaluate the trade-off between sensitivity and specificity.
- **Calibration Curves**: To check if the model's confidence scores are reliable.

These functions handle the logic of switching between saving to disk (CLI mode) 
and returning PIL images (Dashboard mode).
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

def fig_to_image(fig=None):
    """
    Converts a Matplotlib figure into a PIL Image.
    
    This is used primarily by the Gradio dashboard to display plots 
    in memory without saving them to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional.
        The figure to convert. If None, uses the current active figure.
    
    Returns
    -------
    PIL.Image.Image
        The plot rendered as an image object.
    """
    buf = io.BytesIO()
    if fig:
        fig.savefig(buf, format="png", bbox_inches='tight')
    else:
        plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig if fig else "all")
    return img

def plot_training_curves(train_loss, val_loss, val_acc, architecture_name="Model", output_path=None, is_demo=False):
    """
    Generates dual-axis convergence plots for training/validation loss and accuracy.
    """
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(len(train_loss))
    
    # Subplot 1: Loss Convergence
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_loss, 's--', color='#ff7f0e', linewidth=2, label='Val Loss')
    ax1.set_title(f'Loss Convergence ({architecture_name.upper()})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Subplot 2: Accuracy Development
    ax2.plot(epochs, val_acc, 'D-', color='#2ca02c', linewidth=2, label='Val Accuracy')
    ax2.set_title(f'Accuracy Development ({architecture_name.upper()})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    if is_demo:
        return fig_to_image(fig)
    if output_path:
        fig.savefig(output_path / f"{architecture_name.lower()}_convergence.png", dpi=300)
    plt.close(fig)
    return None

def plot_confusion_matrix(y_true, y_pred, architecture_name="Model", output_path=None, is_demo=False):
    """
    Generates a Confusion Matrix to visualize misclassifications.

    Parameters
    ----------
    y_true : array-like.
        Ground truth (correct) target values.

    y_pred : array-like.
        Estimated targets as returned by a classifier.

    architecture_name : str.
        Name of the model (for the plot title).

    output_path : Path, optional.
        Directory to save the `.png` file if `is_demo=False`.

    is_demo : bool.
        If True, returns a PIL Image instead of saving to disk.

    Returns
    -------
    PIL.Image.Image or numpy.ndarray

        Returns the PIL Image if `is_demo=True`.

        Returns the raw confusion matrix array if `is_demo=False`.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({architecture_name.upper()})")
    fig.colorbar(im)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    
    if is_demo:
        return fig_to_image(fig)
    if output_path:
        fig.savefig(output_path / f"{architecture_name.lower()}_confusion_matrix.png")
    plt.close(fig)
    return cm

def plot_roc_curves(y_true, y_probs, num_classes, architecture_name="Model", output_path=None, is_demo=False):
    """
    Generates Macro-Average ROC Curves.
    
    Calculates the Receiver Operating Characteristic (ROC) curve for each class 
    and computes the macro-average AUC score.

    Parameters
    ----------
    y_true : array-like.
        Ground truth labels.

    y_probs : array-like.
        Probability estimates of the positive class.

    num_classes : int.
        Total number of classes.

    architecture_name : str.
        Name of the model.

    output_path : Path, optional.
        Path to save the plot.

    is_demo : bool.
        If True, returns the plot as an image and the AUC score.

    Returns
    -------
    tuple
    
        (PIL.Image.Image, float) if `is_demo=True`.

        (None, float) if `is_demo=False`.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        y_true_onehot = np.eye(num_classes)[y_true]
        
        # Calculate AUC score
        unique_labels = np.unique(y_true)
        auc_score = roc_auc_score(
            y_true_onehot[:, unique_labels],
            y_probs[:, unique_labels],
            average="macro",
            multi_class="ovr"
        )

        fpr_interp = np.linspace(0, 1, 100)
        tpr_all = []
        
        for i in range(num_classes):
            y_true_i = y_true_onehot[:, i]
            y_prob_i = y_probs[:, i]
            # Skip classes with no positive samples in this batch
            if np.sum(y_true_i) == 0 or np.sum(y_true_i) == len(y_true_i):
                continue
            
            fpr_i, tpr_i, _ = roc_curve(y_true_i, y_prob_i)
            ax.plot(fpr_i, tpr_i, color='blue', alpha=0.05)  # Very faint individual lines
            tpr_all.append(np.interp(fpr_interp, fpr_i, tpr_i))

        if tpr_all:
            tpr_mean = np.mean(tpr_all, axis=0)
            ax.plot(fpr_interp, tpr_mean, color='red', linewidth=2, label=f'Macro-average ROC (AUC={auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve ({architecture_name.upper()})")
        ax.legend()
        fig.tight_layout()
        
        if is_demo:
            return fig_to_image(fig), auc_score
        if output_path:
            fig.savefig(output_path / f"{architecture_name.lower()}_roc_curve.png")
        plt.close(fig)
        return None, auc_score
    except Exception as e:
        print(f"⚠️ ROC plot failed: {e}")
        return None, 0.0

def plot_calibration_curve(y_true, y_probs, num_classes, architecture_name="Model", output_path=None, is_demo=False):
    """
    Generates a Reliability Diagram (Calibration Curve).
    
    Compares the model's predicted probability against the actual frequency of the class.
    A perfectly calibrated model will follow the diagonal line.

    Parameters
    ----------
    y_true : array-like.
        True labels.

    y_probs : array-like.
        Predicted probabilities.

    num_classes : int.
        Number of classes.

    architecture_name : str.
        Name of the model.

    output_path : Path, optional
        Save location.

    is_demo : bool.
        Return image object if True.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        y_true_onehot = np.eye(num_classes)[y_true]
        fixed_bins = np.linspace(0, 1, 10)
        prob_true_all = []

        for i in range(num_classes):
            if np.sum(y_true_onehot[:, i]) == 0:
                continue
            prob_true, prob_pred = calibration_curve(y_true_onehot[:, i], y_probs[:, i], n_bins=10)
            prob_true_all.append(np.interp(fixed_bins, prob_pred, prob_true))

        if prob_true_all:
            prob_true_mean = np.mean(prob_true_all, axis=0)
            ax.plot(fixed_bins, prob_true_mean, marker="o", color="blue", label="Avg Calibration")

        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Plot ({architecture_name.upper()})")
        ax.legend()
        fig.tight_layout()
        
        if is_demo:
            return fig_to_image(fig)
        if output_path:
            fig.savefig(output_path / f"{architecture_name.lower()}_calibration.png")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Calibration plot failed: {e}")
        return None