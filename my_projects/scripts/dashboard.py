"""
Gradio Interface for the Animal Classification Project.

This script creates an interactive web dashboard using Gradio to run
the project's three main workflows:
1.  Data Reduction (creating the 'mini_animals' dataset).
2.  Model Training (VGG16, VGG11).
3.  Inference and metric visualization (Confusion Matrix, ROC, etc.).
"""

import gradio as gr
import matplotlib.image as mpimg
from pathlib import Path
from ..modeling.train import main as train_main
from ..modeling.inference import run_inference
from ..reduce_data import reduce_dataset

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = (PROJECT_ROOT / "reports" / "figures").absolute()

# === DATA REDUCTION ===
def reduce_and_visualize(images_per_class, image_size, progress=gr.Progress(track_tqdm=True)):
    """
    Runs the dataset reduction and provides feedback to the Gradio UI.

    Parameters
    ----------
    images_per_class : int
        Number of images to save for each class.
    image_size : int
        The size (width and height) to resize images to.
    progress : gradio.Progress, optional
        Gradio object to track and display progress in the UI.
        Automatically injected if `track_tqdm=True` is enabled
        in the Gradio component.

    Returns
    -------
    str
        A status message indicating the operation's result.
    """
    data_dir = DATA_DIR / "animals" / "animals"
    output_dir = DATA_DIR / "mini_animals" / "animals"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(0, desc="Starting reduction...")
    reduce_dataset(data_dir, output_dir, images_per_class, image_size, progress)
    progress(1, desc="Finished!")
    return f"✅ Reduced dataset created with {images_per_class} images per class, size {image_size}×{image_size}."

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
    dataset_str = "full" if dataset_choice == "Full dataset" else "mini"
    progress(0, desc="Starting training...")
    results = train_main(
        architecture=model_choice.lower(),
        dataset_choice=dataset_str,
        seed=int(seed),
        max_epochs=int(epochs),
        progress=progress
    )
    progress(1, desc="Training finished!")
    test_acc = results.get("test_acc", None)
    test_acc_str = f"{test_acc:.3f}" if test_acc is not None else "N/A"
    metrics_text = f"✅ **Training finished for {model_choice} on {dataset_choice} dataset.**\n**Test Accuracy:** {test_acc_str}"
    return None, metrics_text, "✅ Training completed"

# === INFERENCE ===
def run_inference_gr(dataset_choice, model_choice, batch_size):
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
    dataset_str = "full" if dataset_choice=="Full dataset" else "mini"
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = PROJECT_ROOT / "models" / f"{model_choice.lower()}_state_dict.pth"
    data_dir = DATA_DIR / ("animals/animals" if dataset_str=="full" else "mini_animals/animals")

    results = run_inference(
        model_path=model_path,
        data_dir=data_dir,
        architecture=model_choice.lower(),
        output_path=output_dir,
        batch_size=int(batch_size)
    )

    metrics_text = f"✅ **Validation Accuracy:** {results['val_acc']:.3f} | **F1-Score:** {results['f1_score']:.3f}"

    def load_img(path):
        return mpimg.imread(path) if path.exists() else None

    cm_img = load_img(output_dir / f"{model_choice.lower()}_confusion_matrix.png")
    roc_img = load_img(output_dir / f"{model_choice.lower()}_roc_curve.png")
    cal_img = load_img(output_dir / f"{model_choice.lower()}_calibration.png")

    return metrics_text, cm_img, roc_img, cal_img

# === DASHBOARD ===
with gr.Blocks(title="Animal Classification Dashboard") as demo:
    # --- Reduce Dataset Tab ---
    with gr.Tab("Reduce Dataset"):
        images_per_class = gr.Dropdown([15,30,60], value=30, label="Images per Class")
        image_size = gr.Dropdown([56,112,224], value=224, label="Image Size")
        run_button = gr.Button("Run Reduction")
        reduce_status = gr.Markdown()
        run_button.click(fn=reduce_and_visualize, inputs=[images_per_class, image_size], outputs=reduce_status)

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
        run_train_button.click(
            fn=run_training,
            inputs=[dataset_choice, model_choice, seed, epochs],
            outputs=[plot_output, metrics_output, status_output]
        )

    # --- Inference Tab ---
    with gr.Tab("Inference"):
        infer_dataset_choice = gr.Dropdown(["Full dataset","Reduced dataset"], value="Full dataset", label="Dataset")
        infer_model_choice = gr.Dropdown(["VGG16","VGG11"], value="VGG16", label="Model")
        infer_batch_size = gr.Number(value=16, label="Batch size")
        run_inference_btn = gr.Button("Run Inference")
        infer_metrics = gr.Markdown()
        infer_cm = gr.Image(label="Confusion Matrix")
        infer_roc = gr.Image(label="ROC Curve")
        infer_cal = gr.Image(label="Calibration Plot")
        run_inference_btn.click(
            fn=run_inference_gr,
            inputs=[infer_dataset_choice, infer_model_choice, infer_batch_size],
            outputs=[infer_metrics, infer_cm, infer_roc, infer_cal]
        )

def main():
    """
    Launches the Gradio web application.
    """
    demo.launch()

if __name__ == "__main__":
    main()
