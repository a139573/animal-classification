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

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mini_animals"
# REPORTS_DIR = (PROJECT_ROOT / "reports" / "figures").absolute()

VGG11 = None
VGG16 = None

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
def run_training(dataset_choice, model_choice, seed, epochs, progress=gr.Progress(track_tqdm=True)):
    """
    Runs the model training script and returns metrics.

    Gradio wrapper for the `train_main` function.

    Parameters
    ----------
    dataset_choice : str
        Dataset selection ("Full dataset" or "Reduced dataset").
    model_choice : str
        Model architecture to train (e.g., "VGG16").
    seed : int or float
        Random seed for reproducibility. Will be cast to int.
    epochs : int or float
        Maximum number of epochs. Will be cast to int.
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
    #dataset_str = "full" if dataset_choice == "Full dataset" else "mini"
    progress(0, desc="Starting training...")
    results = train_main(
        architecture=model_choice.lower(),
        dataset_choice="mini",
        seed=int(seed),
        max_epochs=int(epochs),
        progress=progress,
        is_demo=True
    )
    val_acc = results.get("val_acc", None)
    val_acc_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"

    

    train_losses = results.get("train_loss_list", None)
    val_losses = results.get("val_loss_list", None)
    val_accs = results.get("val_acc_list", None)

    fig, ax = plt.subplots(figsize=(6, 4))
    if train_losses:
        ax.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker="o")
    if val_losses:
        ax.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="x")
    if val_accs:
        ax.plot(range(1, len(val_accs)+1), val_accs, label="Validation Acc", marker="s")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(f"{model_choice} Convergence")
    ax.grid(True)
    ax.legend()

    metrics_text = f"✅ **Training finished for {model_choice} on {dataset_choice} dataset.**\n**Final val Accuracy:** {val_acc_str}"
    return results.get("model", None), metrics_text, "✅ Training completed", fig

# === INFERENCE ===
def run_inference_gr(trained_model, batch_size):
    """
    Runs inference on the validation set and generates visualizations.

    Gradio wrapper for the `run_inference` function. Loads the
    generated metric images (confusion matrix, ROC, calibration).

    Parameters
    ----------
    dataset_choice : str
        Dataset selection ("Full dataset" or "Reduced dataset").
    model_choice : str
        Model architecture to use (e.g., "VGG16").
    batch_size : int or float
        Batch size for inference. Will be cast to int.

    Returns
    -------
    metrics_text : str
        Formatted string with metrics (Accuracy, F1-Score).
    cm_img : np.ndarray or None
        Confusion matrix image loaded via mpimg.
    roc_img : np.ndarray or None
        ROC curve image loaded via mpimg.
    cal_img : np.ndarray or None
        Calibration plot image loaded via mpimg.
    """

    val_probs, val_labels, acc, f1, cm_img, roc_img, cal_img = run_inference(
        model_path=None,
        trained_model=trained_model,
        data_dir=DATA_DIR,
        architecture=None,
        output_path=None,
        batch_size=int(batch_size),
        is_demo=True
    )

    

    metrics_text = f"✅ **Validation Accuracy:** {acc:.3f} | **F1-Score:** {f1:.3f}"


    n_display = 100  # show first 100 rows
    table_df = pd.DataFrame({
        "True Label": val_labels[:n_display],
        "Predicted Prob": val_probs[:n_display].max(axis=1)  # top-class probability
    })

    return metrics_text, cm_img, roc_img, cal_img, table_df


# ------ DASHBOARD ------ #
dark_theme = gr.themes.Monochrome()

with gr.Blocks(title="Animal Classification Dashboard", theme=dark_theme) as demo:
    # --- Reduce Dataset Tab ---
    # with gr.Tab("Reduce Dataset"):
    #     images_per_class = gr.Dropdown([15,30,60], value=30, label="Images per Class")
    #     image_size = gr.Dropdown([56,112,224], value=224, label="Image Size")
    #     run_button = gr.Button("Run Reduction")
    #     reduce_status = gr.Markdown()
    #     run_button.click(fn=reduce_and_visualize, inputs=[images_per_class, image_size], outputs=reduce_status)

    # --- Train Tab ---
    with gr.Tab("Train"):
        dataset_choice = gr.Dropdown(["Full dataset","Reduced dataset"], value="Full dataset", label="Dataset")
        seed = gr.Number(value=123, label="Random Seed")
        model_choice = gr.Dropdown(["VGG16","VGG11"], value="VGG16", label="Model")
        epochs = gr.Number(value=2, label="Epochs")
        run_train_button = gr.Button("Run Training")
        plot_output = gr.Plot()
        metrics_output = gr.Markdown()
        status_output = gr.Markdown()
        train_convergence = gr.Plot(label="Loss and accuracy convergence")
        trained_model = gr.State()
        run_train_button.click(
            fn=run_training,
            inputs=[dataset_choice, model_choice, seed, epochs],
            outputs=[trained_model, metrics_output, status_output, train_convergence]
        )

    # --- Inference Tab ---
    with gr.Tab("Inference"):
        infer_dataset_choice = gr.Dropdown(["Full dataset","Reduced dataset"], value="Full dataset", label="Dataset")
        # infer_model_choice = gr.Dropdown(["VGG16","VGG11"], value="VGG16", label="Model")
        infer_batch_size = gr.Number(value=16, label="Batch size")
        run_inference_btn = gr.Button("Run Inference")
        infer_table = gr.Dataframe(headers=["True Label", "Predicted Prob"], interactive=False, label="Predictions Table")
        infer_metrics = gr.Markdown()
        infer_cm = gr.Image(label="Confusion Matrix")
        infer_roc = gr.Image(label="ROC Curve")
        infer_cal = gr.Image(label="Calibration Plot")
        run_inference_btn.click(
            fn=run_inference_gr,
            inputs=[trained_model, infer_batch_size],
            outputs=[infer_metrics, infer_cm, infer_roc, infer_cal, infer_table]
        )

def main():
    """
    Launches the Gradio web application.
    """
    demo.launch()

if __name__ == "__main__":
    main()
