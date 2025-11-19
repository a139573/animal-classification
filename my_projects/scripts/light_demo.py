"""
Gradio Demo for Animal Classification Project

Lightweight demo: uses a mini dataset and pre-trained mini models.
No folders are created in the user's current directory.
"""

import gradio as gr
import matplotlib.image as mpimg
from pathlib import Path
import tempfile
from ..modeling.inference import run_inference
import importlib.resources as pkg_resources
from importlib.resources import files

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes up from scripts/
DATA_DIR = PROJECT_ROOT / "data" / "mini_animals"

# === TEMPORARY PATHS ===

# === DEMO MODE ===
# MINI_MODEL_PATHS = {
#     "VGG16": Path(__file__).resolve().parents[2] / "models" / "vgg16_mini.pth",
#     "VGG11": Path(__file__).resolve().parents[2] / "models" / "vgg11_mini.pth"
# }

# === INFERENCE ONLY ===
def run_demo_inference(model_choice, batch_size):
    model_path = MINI_MODEL_PATHS[model_choice]
    results = run_inference(
        model_path=model_path,
        data_dir=DATA_DIR,
        architecture=model_choice.lower(),
        output_path=None,
        batch_size=int(batch_size)
    )
    metrics_text = (
        f"✅ **Validation Accuracy:** {results['val_acc']:.3f} | "
        f"**F1-Score:** {results['f1_score']:.3f}"
    )

    def load_img(path):
        return mpimg.imread(path) if path.exists() else None

    cm_img = load_img(REPORTS_DIR / f"{model_choice.lower()}_confusion_matrix.png")
    roc_img = load_img(REPORTS_DIR / f"{model_choice.lower()}_roc_curve.png")
    cal_img = load_img(REPORTS_DIR / f"{model_choice.lower()}_calibration.png")

    return metrics_text, cm_img, roc_img, cal_img

# === DASHBOARD ===
with gr.Blocks(title="Animal Classification Demo") as demo:
    with gr.Tab("Inference (Demo)"):
        model_choice = gr.Dropdown(["VGG16", "VGG11"], value="VGG16", label="Model")
        batch_size = gr.Number(value=16, label="Batch Size")
        run_btn = gr.Button("Run Inference")
        metrics_output = gr.Markdown()
        cm_output = gr.Image(label="Confusion Matrix")
        roc_output = gr.Image(label="ROC Curve")
        cal_output = gr.Image(label="Calibration Plot")

        run_btn.click(
            fn=run_demo_inference,
            inputs=[model_choice, batch_size],
            outputs=[metrics_output, cm_output, roc_output, cal_output]
        )

def main():
    demo.launch()

if __name__ == "__main__":
    main()
