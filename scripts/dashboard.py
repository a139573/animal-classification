import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from my_projects.modeling.train import main as train_main
from my_projects.modeling.inference import run_inference
from my_projects.reduce_data import reduce_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level up from /scripts/
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_DIR = PROJECT_ROOT / "data"

from tqdm import tqdm

# === DATA REDUCTION FUNCTION ===
def reduce_and_visualize(images_per_class, image_size, progress=gr.Progress(track_tqdm=True)):
    data_dir = DATA_DIR / "animals" / "animals"
    output_dir = DATA_DIR / "mini_animals" / "animals"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(0, desc="Starting reduction...")
    reduce_dataset(data_dir, output_dir, images_per_class, image_size, progress)
    progress(1, desc="Finished!")


    status = f"✅ Reduced dataset created with {images_per_class} images per class, size {image_size}×{image_size}."
    return status


def run_training(dataset_choice, model_choice, seed, epochs, progress=gr.Progress(track_tqdm=True)):
    # Map Gradio dropdown to dataset choice string expected by train.py
    dataset_str = "full" if dataset_choice == "Full dataset" else "mini"

    progress(0, desc="Starting training...")
    train_main(
        architecture=model_choice.lower(),
        dataset_choice=dataset_str,
        seed=int(seed),
        max_epochs=int(epochs),
        progress = progress
    )
    progress(1, desc="Training finished!")

    # You can add dummy outputs for now (plots will be updated later)
    metrics_text = f"✅ Training finished for {model_choice} on {dataset_choice} dataset."
    return None, metrics_text, "✅ Training completed"


# === TRAINING VISUALIZATION FUNCTION ===
calibration_path = REPORTS_DIR / "calibration.png"
confusion_path = REPORTS_DIR / "confusion_matrix.png"

def train_and_visualize(dataset_choice, model_choice, seed, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    calib_img = mpimg.imread(calibration_path)
    axes[0].imshow(calib_img)
    axes[0].set_title("Calibration Plot")
    axes[0].axis("off")

    cm_img = mpimg.imread(confusion_path)
    axes[1].imshow(cm_img)
    axes[1].set_title("Confusion Matrix")
    axes[1].axis("off")

    metrics_text = (
        f"**Dataset:** {dataset_choice}\n"
        f"**Model:** {model_choice}\n"
        f"**Seed:** {seed}\n"
        f"**Epochs:** {epochs}\n"
        f"**Test Accuracy:** 0.85"
    )

    status = "✅ Training simulated"
    return fig, metrics_text, status


# === GRADIO INTERFACE ===
with gr.Blocks(title="Animal Classification Dashboard") as demo:
    with gr.Tab("Reduce Dataset"):
        gr.Markdown("### Configure dataset reduction, then click **Run Reduction**.")
        images_per_class = gr.Dropdown([15, 30, 60], value=30, label="Images per Class")
        image_size = gr.Dropdown([56, 112, 224], value=224, label="Image Size")
        run_button = gr.Button("Run Reduction")
        reduce_status = gr.Markdown()

        # Only run when the button is clicked
        run_button.click(
            fn=reduce_and_visualize,
            inputs=[images_per_class, image_size],
            outputs=reduce_status
        )

    with gr.Tab("Train"):
        gr.Markdown("### Configure training parameters and click **Run Training**.")
        dataset_choice = gr.Dropdown(["Full dataset", "Reduced dataset"], value="Full dataset", label="Dataset")
        seed = gr.Number(value=123, label="Random Seed")
        model_choice = gr.Dropdown(["VGG16", "VGG11"], value="VGG16", label="Model")
        epochs = gr.Number(value=5, label="Epochs")
        run_button = gr.Button("Run Training")
        plot_output = gr.Plot()
        metrics_output = gr.Markdown()
        status_output = gr.Markdown()


    with gr.Tab("Results"):
        plot_output = gr.Plot()
        metrics_output = gr.Markdown()
        status_output = gr.Markdown()

    # Run dataset reduction
    inputs = [dataset_choice, model_choice, seed, epochs]
    outputs = [plot_output, metrics_output, status_output]
    for input_component in inputs:
        input_component.change(fn=train_and_visualize, inputs=inputs, outputs=outputs)

    # Run training
    inputs = [dataset_choice, model_choice, seed, epochs]
    outputs = [plot_output, metrics_output, status_output]
    run_button.click(
        fn=run_training,
        inputs=inputs,
        outputs=outputs
    )


    demo.load(fn=train_and_visualize, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()