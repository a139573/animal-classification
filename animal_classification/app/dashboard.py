"""
# 🦁 Animal Classification Dashboard

This module launches an interactive web-based Graphical User Interface (GUI) using **Gradio**.

It serves as the central control hub for the project, allowing users to perform end-to-end Machine Learning workflows without writing code.

## 🌟 Features
1.  **Interactive Training:**
    - Select architectures (VGG16 vs VGG11).
    - Configure hyperparameters (Epochs, Seed, Workers).
    - **Real-time Visualization:** Watch Loss and Accuracy curves update live as the model trains.
2.  **Model Evaluation:**
    - Seamlessly pass the *just-trained* model from memory to the inference engine.
    - Fallback to disk loading: If no model is in memory, it automatically finds the latest checkpoint in `models/`.
    - visualizes **Confusion Matrices**, **ROC Curves**, and **Calibration Plots**.

## 🚀 How to Run
Execute the following command from your project root:
```bash
python -m animal_classification.app.dashboard
```

## 🏗 Architecture
This script acts as an orchestrator. It does not contain ML logic itself but imports and executes:

animal_classification.modeling.train.main for the training loop.

animal_classification.modeling.inference.run_inference for evaluation. """

import glob
import os
from pathlib import Path

import gradio as gr
from matplotlib import pyplot as plt

# --- Project Imports ---# 
# These imports link the GUI to the core logic libraries
from ..modeling.train import main as train_main
from ..modeling.inference import run_inference

# (Optional) Preprocessing module - currently disabled in UI
from ..preprocessing.reduce_data import reduce_dataset
# === CONSTANTS & PATHS ===
DEFAULT_DATA_DIR = Path("data/mini_animals/animals")
DEFAULT_MODELS_DIR = Path("models")

# === WORKFLOW 1: DATA REDUCTION (Disabled) ===#
# def reduce_and_visualize(images_per_class, image_size, progress=gr.Progress(track_tqdm=True)):
# === WORKFLOW 2: TRAINING ===
def run_training(model_choice, seed, epochs, num_workers, progress=gr.Progress()): 
    """ Orchestrates the training workflow triggered by the UI.

    It calls the main training script in 'demo mode' (is_demo=True), which prevents
    saving heavy files to disk and instead returns the model object and metrics 
    directly to memory for immediate visualization.

    Parameters
    ----------
    model_choice : str
        Dropdown selection (e.g., "VGG16").
    seed : float
        Random seed (cast to int internally).
    epochs : float
        Number of epochs (cast to int internally).
    num_workers : float
        Number of data loading workers.
    progress : gradio.Progress
        Automatically injected by Gradio to track the loop.

    Returns
    -------
    tuple
        (Trained Model Object, Metrics Markdown, Loss Figure, Accuracy Figure)
    """
    # Run the training pipeline
    results = train_main(
        architecture=model_choice.lower(),
        dataset_choice="mini",
        seed=int(seed),
        max_epochs=int(epochs),
        num_workers=int(num_workers),
        progress=progress,
        is_demo=True
    )

    # Extract Metrics
    val_acc = results.get("val_acc", None)
    val_acc_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"

    train_losses = results.get("train_loss_list", [])
    val_losses = results.get("val_loss_list", [])
    val_accs = results.get("val_acc_list", [])

    # --- Plot 1: Loss ---
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    if train_losses:
        ax_loss.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    if val_losses:
        ax_loss.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="x")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_choice} — Loss Convergence")
    ax_loss.grid(True)
    ax_loss.legend()

    # --- Plot 2: Accuracy ---
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    if val_accs:
        ax_acc.plot(range(1, len(val_accs) + 1), val_accs, label="Validation Accuracy", marker="s", color="green")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{model_choice} — Validation Accuracy")
        ax_acc.grid(True)
        ax_acc.legend()

    metrics_text = f"✅ **Training finished for {model_choice}**\n**Final Val Accuracy:** {val_acc_str}"

    # Return the trained model object (to be stored in Gradio State) plus the plots
    return results.get("model", None), metrics_text, fig_loss, fig_acc

# === WORKFLOW 3: INFERENCE ===
def run_inference_gr(architecture, trained_model, batch_size, num_workers):
    """ Orchestrates the inference/evaluation workflow.

    This function is smart about model loading:
    1. **Memory First:** Checks if `trained_model` (passed from the Train tab via Gradio State) exists.
    2. **Disk Fallback:** If memory is empty, searches `models/` for the most recent `.ckpt` file.

    Returns
    -------
    tuple
        (Metrics Markdown, Confusion Matrix Image, ROC Curve Image, Calibration Plot Image)
    """
    model_path = None

    if trained_model is None:
        print("NO TRAINED MODEL IN MEMORY! Attempting to find the latest saved checkpoint...")
        
        # Search for .ckpt files (PyTorch Lightning default)
        model_pattern = str(DEFAULT_MODELS_DIR / '*.ckpt')
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            # Try finding state_dict .pth files as fallback
            fallback_pattern = str(DEFAULT_MODELS_DIR / '*.pth')
            model_files = glob.glob(fallback_pattern)

        if not model_files:
            raise FileNotFoundError(
                f"No model checkpoints found in {DEFAULT_MODELS_DIR}. "
                "Please train a model first."
            )
            
        # Sort by newest first
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = Path(model_files[0])
        print(f"Using latest saved checkpoint: {model_path}")

    # Execute Inference Pipeline
    val_probs, val_labels, acc, f1, cm_img, roc_img, cal_img = run_inference(
        model_path=model_path,
        trained_model=trained_model,
        data_dir=DEFAULT_DATA_DIR,
        architecture=architecture.lower(),
        output_path=None,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        is_demo=True
    )

    metrics_text = f"✅ **Validation Accuracy:** {acc:.3f} | **F1-Score:** {f1:.3f}"
    return metrics_text, cm_img, roc_img, cal_img

# === UI LAYOUT ===
def create_demo():
    """ Constructs the Gradio Blocks layout.

    Defines the Tabs, Inputs, Outputs, and event listeners (button clicks).
    It also initializes the `gr.State` component used to share the trained model
    between the Training and Inference tabs.
    """
    with gr.Blocks(title="Animal Classification Dashboard") as demo: 
        gr.Markdown("# 🦁 Animal Classification Dashboard")
        
        # --- Tab 1: Reduce Data (Disabled) ---
        """
        with gr.Tab("1. Prepare Data"):
            ... (Hidden) ...
        """

        # --- Tab 2: Training ---
        with gr.Tab("1. Train Model"):
            gr.Markdown("Train a VGG model from scratch (or transfer learning) on the mini dataset.")
            
            with gr.Row():
                model_choice = gr.Dropdown(["VGG16", "VGG11"], value="VGG16", label="Architecture")
                epochs = gr.Number(value=2, label="Epochs", precision=0)
                seed = gr.Number(value=42, label="Random Seed")
                num_workers = gr.Slider(minimum=0, maximum=8, step=1, value=2, label="Num Workers")
            
            train_btn = gr.Button("Start Training", variant="primary")
            
            with gr.Row():
                train_losses = gr.Plot(label="Loss History")
                train_accs = gr.Plot(label="Accuracy History")
            
            metrics_output = gr.Markdown()
            
            # State variable to hold the model in memory between Train and Inference tabs
            trained_model_state = gr.State()
            
            train_btn.click(
                fn=run_training,
                inputs=[model_choice, seed, epochs, num_workers],
                outputs=[trained_model_state, metrics_output, train_losses, train_accs]
            )

        # --- Tab 3: Inference ---
        with gr.Tab("2. Evaluation"):
            gr.Markdown("Evaluate the model (either the one just trained, or the latest saved checkpoint).")
            
            with gr.Row():
                infer_batch_size = gr.Number(value=16, label="Batch Size", precision=0)
                infer_workers = gr.Slider(minimum=0, maximum=8, step=1, value=2, label="Num Workers")
            
            infer_btn = gr.Button("Run Evaluation", variant="primary")
            infer_metrics = gr.Markdown()
            
            with gr.Row():
                infer_cm = gr.Image(label="Confusion Matrix", type="pil")
                infer_roc = gr.Image(label="ROC Curve", type="pil")
                infer_cal = gr.Image(label="Calibration Plot", type="pil")

            infer_btn.click(
                fn=run_inference_gr,
                inputs=[model_choice, trained_model_state, infer_batch_size, infer_workers],
                outputs=[infer_metrics, infer_cm, infer_roc, infer_cal]
            )

    return demo
        

def main(): 
    """Launches the Gradio web server."""
    demo = create_demo()
    demo.launch()

if __name__ == "main":
    main()