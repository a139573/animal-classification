"""
Gradio Interface for the Animal Classification Project.

This script creates an interactive web dashboard using Gradio to run
the project's three main workflows:
1.  Data Reduction (creating the 'mini_animals' dataset).
2.  Model Training (VGG16, VGG11).
3.  Inference and metric visualization (Confusion Matrix, ROC, etc.).
"""

import gradio as gr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from ..modeling.train import main as train_main
from ..modeling.inference import run_inference
from ..reduce_data import reduce_dataset
import pandas as pd
import glob
import os

# === PATHS ===
# Fallback defaults relative to where user runs dashboard
DEFAULT_DATA_DIR = Path("data/mini_animals/animals")
DEFAULT_MODELS_DIR = Path("models")
# REPORTS_DIR = (PROJECT_ROOT / "reports" / "figures").absolute()

architecture = None

VGG11 = None
"""Placeholder for the VGG11 model."""
VGG16 = None
"""Placeholder for the VGG16 model."""

# === DATA REDUCTION ===
# def reduce_and_visualize(images_per_class, image_size, progress=gr.Progress(track_tqdm=True)):
#     """
#     Runs the dataset reduction and provides feedback to the Gradio UI.

#     Parameters
#     ----------
#     images_per_class : int
#         Number of images to save for each class.
#     image_size : int
#         The size (width and height) to resize images to.
#     progress : gradio.Progress, optional
#         Gradio object to track and display progress in the UI.
#         Automatically injected if `track_tqdm=True` is enabled
#         in the Gradio component.

#     Returns
#     -------
#     str
#         A status message indicating the operation's result.
#     """
#     data_dir = DATA_DIR / "animals" / "animals"
#     output_dir = DATA_DIR / "mini_animals" / "animals"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     progress(0, desc="Starting reduction...")
#     reduce_dataset(data_dir, output_dir, images_per_class, image_size, progress)
#     progress(1, desc="Finished!")
#     return f"✅ Reduced dataset created with {images_per_class} images per class, size {image_size}×{image_size}."

# === TRAINING ===
def run_training(model_choice, seed, epochs, num_workers, progress=gr.Progress()):
    """
    Runs the model training script and returns metrics.

    Gradio wrapper for the `train_main` function.

    Parameters
    ----------
    model_choice : str
        Model architecture to train (e.g., "VGG16").
    seed : int or float
        Random seed for reproducibility. Will be cast to int.
    epochs : int or float
        Maximum number of epochs. Will be cast to int.
    num_workers : int
        Number of subprocesses to use for data loading.
    progress : gradio.Progress, optional
        Gradio object to track training progress.

    Returns
    -------
    None
        A placeholder for the plot output (currently unused).
    str
        A formatted string with performance metrics (Test Accuracy).
    str
        A simple status message indicating completion.
    """

    results = train_main(
        architecture=model_choice.lower(),
        dataset_choice="mini",
        seed=int(seed),
        max_epochs=int(epochs),
        num_workers=num_workers,
        progress=progress,
        is_demo=True
    )
    val_acc = results.get("val_acc", None)
    val_acc_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"

    

    train_losses = results.get("train_loss_list", None)
    val_losses = results.get("val_loss_list", None)
    val_accs = results.get("val_acc_list", None)

    # --- FIGURE 1: LOSSES ---
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))

    if train_losses:
        ax_loss.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Train Loss",
            marker="o"
        )

    if val_losses:
        ax_loss.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label="Validation Loss",
            marker="x"
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_choice} — Loss Convergence")
    ax_loss.grid(True)
    ax_loss.legend()

   # loss_img = fig_to_image(fig_loss)  # or save if not demo



    # --- FIGURE 2: VALIDATION ACCURACY ---
    if val_accs:
        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))

        ax_acc.plot(
            range(1, len(val_accs) + 1),
            val_accs,
            label="Validation Accuracy",
            marker="s"
        )

        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{model_choice} — Validation Accuracy")
        ax_acc.grid(True)
        ax_acc.legend()

       # acc_img = fig_to_image(fig_acc)
   # else:
   #     acc_img = None


    metrics_text = f"✅ **Training finished for {model_choice} on reduced dataset.**\n**Final val Accuracy:** {val_acc_str}"
    return results.get("model", None), metrics_text, fig_loss, fig_acc

# === INFERENCE ===
def run_inference_gr(architecture, trained_model, batch_size, num_workers):
    """
    Runs inference on the validation set and generates visualizations.

    If a trained_model is provided (from demo memory), it is used directly.
    Otherwise, the function loads a model checkpoint from disk.

    Metrics are displayed in the dashboard but not stored on disk.
    """
    if trained_model is None:
        print("NO TRAINED MODEL! Attempting to find the latest saved checkpoint...")
        
        # Strictly search for the standard PyTorch Lightning checkpoint extension (.ckpt).
        model_pattern = str(DEFAULT_MODELS_DIR / '*.ckpt')
        print(f"Searching for standard checkpoint files: {model_pattern}")
        
        # Use glob.glob to find all matching files
        model_files = glob.glob(model_pattern)
        
        if len(model_files) == 0:
            # If no CKPT files are found, raise an error immediately. No fallback to .pth.
            raise FileNotFoundError(
                f"No standard model checkpoint files (.ckpt) found in the directory: {DEFAULT_MODELS_DIR}. "
                f"Checked pattern: '{model_pattern}'. "
                f"The dashboard requires a PyTorch Lightning checkpoint (.ckpt) to ensure all necessary metadata is loaded."
            )
            
        # Sort files by modification time, newest first (reverse=True)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        model_path = Path(model_files[0])
        print(f"Using latest saved checkpoint from {model_path}")
    else:
        model_path = None


    val_probs, val_labels, acc, f1, cm_img, roc_img, cal_img = run_inference(
        model_path=model_path,
        trained_model=trained_model,
        data_dir=DEFAULT_DATA_DIR,
        architecture=architecture,
        output_path=None,
        batch_size=int(batch_size),
        num_workers=num_workers,
        is_demo=True
    )

    

    metrics_text = f"✅ **Validation Accuracy:** {acc:.3f} | **F1-Score:** {f1:.3f}"


    return metrics_text, cm_img, roc_img, cal_img


# ------ DASHBOARD ------ #
dark_theme = gr.themes.Monochrome()

with gr.Blocks(title="Animal Classification Dashboard") as demo:
    # --- Reduce Dataset Tab ---
    # with gr.Tab("Reduce Dataset"):
    #     images_per_class = gr.Dropdown([15,30,60], value=30, label="Images per Class")
    #     image_size = gr.Dropdown([56,112,224], value=224, label="Image Size")
    #     run_button = gr.Button("Run Reduction")
    #     reduce_status = gr.Markdown()
    #     run_button.click(fn=reduce_and_visualize, inputs=[images_per_class, image_size], outputs=reduce_status)

    # --- Train Tab ---
    with gr.Tab("Train"):
        seed = gr.Number(value=123, label="Random Seed")
        model_choice = gr.Dropdown(["VGG16","VGG11"], value="VGG16", label="Model")
        epochs = gr.Number(value=2, label="Epochs")
        num_workers = gr.Slider(
            minimum=1,
            maximum=16,
            step=1,
            value=2,
            label="Num workers"
        )
        run_train_button = gr.Button("Run Training")
        metrics_output = gr.Markdown()
        train_losses = gr.Plot(label="Loss convergence")
        train_accs = gr.Plot(label="Accuracy convergence")
        trained_model = gr.State()
        run_train_button.click(
            fn=run_training,
            inputs=[model_choice, seed, epochs, num_workers],
            outputs=[trained_model, metrics_output, train_losses, train_accs]
        )

    # --- Inference Tab ---
    with gr.Tab("Inference"):
        infer_model_choice = model_choice
        infer_num_workers = gr.Slider(
            minimum=1,
            maximum=16,
            step=1,
            value=2,
            label="Num workers"
        )
        infer_batch_size = gr.Number(value=16, label="Batch size")
        run_inference_btn = gr.Button("Run Inference")
        infer_metrics = gr.Markdown()
        infer_cm = gr.Image(label="Confusion Matrix")
        infer_roc = gr.Image(label="ROC Curve")
        infer_cal = gr.Image(label="Calibration Plot")
       # infer_table = gr.Dataframe(headers=["True Label", "Predicted Prob for True Label"], interactive=False, label="Predictions Table")
        run_inference_btn.click(
            fn=run_inference_gr,
            inputs=[infer_model_choice, trained_model, infer_batch_size, infer_num_workers],
            outputs=[infer_metrics, infer_cm, infer_roc, infer_cal]
        )

def main():
    """
    Launches the Gradio web application.
    """
    demo.launch()

if __name__ == "__main__":
    main()
