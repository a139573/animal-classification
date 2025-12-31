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
import random
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

# --- DATA EXPLORATION PLOTS ---

def get_dataset_stats(data_dir: Path):
    """
    Calculates basic statistics for the dataset exploration tab.

    Inspects the provided directory to count classes and filter valid image files
    to provide a summary of the dataset's composition.

    Parameters
    ----------
    data_dir : Path\n
        The root directory containing animal class subfolders.

    Returns
    -------
    tuple
        A tuple containing (pd.DataFrame of counts per species, 
        int total_number_of_classes, int total_image_count, 
        list_of_species_names).
    """
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    counts = {d.name: len([f for f in d.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]) for d in class_folders}
    df = pd.DataFrame(list(counts.items()), columns=['Species', 'Image Count'])
    return df, len(class_folders), sum(counts.values()), [d.name for d in class_folders]

def get_species_samples(data_dir: Path, species_name: str, num_samples=8):
    """
    Returns a list of random PIL images for a specific animal species. 

    This function filters for common image extensions to prevent loading 
    metadata or hidden system files and provides samples for UI visualization.

    Parameters
    ----------
    data_dir : Path\n
        The root directory of the dataset.\n
    species_name : str\n
        The folder name of the target species.\n
    num_samples : int, optional\n
        Maximum number of samples to retrieve (default: 8).

    Returns
    -------
    list
        A list of PIL.Image objects corresponding to the sampled images.
    """
    species_path = data_dir / species_name
    if not species_path.exists():
        return []
    
    # Filter for valid image formats only
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_paths = [p for p in species_path.glob('*') if p.suffix.lower() in valid_extensions]
    
    selected = random.sample(img_paths, min(num_samples, len(img_paths)))
    return [Image.open(p).convert("RGB") for p in selected]

def plot_color_analysis(data_dir: Path):
    """
    Performs RGB profile analysis on a subset of classes for visualization.

    Computes the average R, G, and B channel intensities across a sample of 
    species to visualize color dominance in different animal groups. 
    Limited to 25 species for plot readability.

    Parameters
    ----------
    data_dir : Path\n
        The directory containing the class folders.

    Returns
    -------
    PIL.Image.Image\n
        The generated stacked bar plot as a PIL image object.
    """
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    rgb_means = []
    names = []
    
    # We take a representative sample of 25 species for UI legibility
    for folder in class_folders[:25]: 
        imgs = [f for f in folder.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        imgs = imgs[:10] # Sample 10 images for average
        
        if not imgs: continue
        
        class_rgb = []
        for p in imgs:
            with Image.open(p) as img:
                arr = np.array(img.convert("RGB"))
                class_rgb.append(arr.mean(axis=(0,1)))
        
        rgb_means.append(np.mean(class_rgb, axis=0))
        names.append(folder.name)
    
    df = pd.DataFrame(rgb_means, columns=['R', 'G', 'B'], index=names) / 255.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71', '#3498db'], ax=ax, alpha=0.8)
    ax.set_title("Average Color Profiles (Subset of 25 Species)")
    ax.set_ylabel("Mean Intensity")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    return fig_to_image(fig)


# --- MODEL EVALUATION PLOTS ---

def fig_to_image(fig=None):
    """
    Converts a Matplotlib figure into a PIL Image.
    
    This is used primarily by the Gradio dashboard to display plots 
    in memory without saving them to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional\n
        The figure to convert. If None, uses the current active figure.
    
    Returns
    -------
    PIL.Image.Image\n
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
    Generates dual-axis convergence plots for training/validation performance.

    Visualizes how loss decreases and accuracy increases over training epochs,
    providing immediate feedback on model learning and potential overfitting.

    Parameters
    ----------
    train_loss : list
        Historical values of training loss per epoch.\n
    val_loss : list\n
        Historical values of validation loss per epoch.\n
    val_acc : list\n
        Historical values of validation accuracy per epoch.\n
    architecture_name : str, optional\n
        Name of the model backbone used (default: "Model").\n
    output_path : Path, optional\n
        Filesystem path to save the generated image (ignored if is_demo is True).\n
    is_demo : bool, optional\n
        If True, returns the plot as a PIL Image object (default: False).

    Returns
    -------
    PIL.Image.Image or None
        Returns the image object if in demo mode, otherwise None.
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
    y_true : array-like\n
        Ground truth (correct) target values.\n
    y_pred : array-like\n
        Estimated targets as returned by a classifier.\n
    architecture_name : str\n
        Name of the model (for the plot title).\n
    output_path : Path, optional\n
        Directory to save the `.png` file if `is_demo=False`.\n
    is_demo : bool\n
        If True, returns a PIL Image instead of saving to disk.

    Returns
    -------
    PIL.Image.Image or numpy.ndarray\n
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
    y_true : array-like\n
        Ground truth labels.\n
    y_probs : array-like\n
        Probability estimates of the positive class.\n
    num_classes : int\n
        Total number of classes.\n
    architecture_name : str\n
        Name of the model.\n
    output_path : Path, optional\n
        Path to save the plot.\n
    is_demo : bool\n
        If True, returns the plot as an image and the AUC score.

    Returns
    -------
    tuple\n
        A tuple containing (PIL.Image.Image, float_auc_score) if `is_demo=True`.
        A tuple containing (None, float_auc_score) if `is_demo=False`.
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
    y_true : array-like\n
        True labels.\n
    y_probs : array-like\n
        Predicted probabilities.\n
    num_classes : int\n
        Number of classes.\n
    architecture_name : str\n
        Name of the model.\n
    output_path : Path, optional\n
        Save location on the filesystem.\n
    is_demo : bool\n
        If True, returns the generated plot as a PIL image object.
        
    Returns
    -------
    PIL.Image.Image or None\n
        The reliability diagram image object if is_demo is True, otherwise None.
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