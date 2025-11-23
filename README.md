# 🐾 Animal Classification Project

[![Dataset](https://img.shields.io/badge/🤗_dataset-kaggle-red.svg)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
[![PyPI](https://img.shields.io/badge/PyPI-TestPyPI-blue.svg)](https://test.pypi.org/project/animal-classification/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/) ---

## 📘 Overview
This project trains VGG-based deep learning models to classify images of **90 animal species**.
It uses **PyTorch** and **PyTorch Lightning** for modeling and provides command-line tools for:

* Data downloading and preprocessing
* Model training and inference
* Visualization and interactive dashboards via Gradio

The project follows a modular cookiecutter-style structure, with code organized inside `animal_classification/my_projects/`.

---

## 📈 Key Results & Demo

A VGG16 model trained for 10 epochs achieved the following performance:

| Metric | Score |
| :--- | :--- |
| **Top-1 Accuracy** | 92.5% |
| **Loss (Validation)** | 0.21 |

### Interactive Dashboard Preview
Launch the demo with `animal-dashboard` to interact with the trained model and view metrics.


---

## 📂 Dataset
Dataset: **[Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)** Contains **5,400 images** across **90 classes**.

---

## ⚙️ Installation Options

### 1. 🚀 Run the Demo (No installation needed)
Use `uvx` (assuming `uv` is installed) to download and run the Gradio demo instantly. This includes a mini dataset for quick experiments.

```bash
uvx animal-dashboard --index [https://test.pypi.org/simple/animal-classification](https://test.pypi.org/simple/animal-classification)
```

### 2. 📦 Install as a Dependency
To use this project as a library within your own code:

```bash
uv add --index [https://test.pypi.org/simple/animal-classification](https://test.pypi.org/simple/animal-classification)
```

### 3. 🔧 Local Development (Your own repo / experiments)
For running training/inference scripts locally, clone the repository and set up the development environment:
```bash
git clone [https://github.com/a139573/animal-classification](https://github.com/a139573/animal-classification)
cd animal-classification
```
Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

Install the project and its dependencies in editable mode
```bash
uv pip install -e .
```

### 4. 🧪 Install Built Wheel (QA/Internal Testing)

If you have a pre-built wheel file (.whl) from the uv build command and want to test the non-editable final artifact (e.g., for QA or distributing a specific version), use the file path.

To install the CPU-only version:

```bash
uv pip install animal_classification-1.0.0.whl
```

To install the GPU version (Requires NVIDIA/CUDA environment):

```bash
uv pip install 'animal_classification-1.0.0.whl[gpu]'
```

## 🧩 Project Structure
animal_classification/
│
├── animal_classification/
│   ├── dataset.py
│   ├── download_data.py
│   ├── reduce_data.py
│   ├── plots.py
│   ├── modeling/
│   └── scripts/
│
├── notebooks/           # Exploratory data analysis and model prototyping
├── reports/             # Generated charts, metrics, and figures
├── pyproject.toml
└── README.md

## 🚀 Usage
After installation, the following CLI commands are available:

### 1️⃣ Download Full Dataset
Downloads the full 90-class dataset into data/animals/.
```bash
animal-download-data
```
Required arguments: None
Optional arguments: None

### 2️⃣ Reduce Dataset (Optional)
Creates a smaller, preprocessed dataset for faster testing or development. There is already one included. If you run this command, the given reduced dataset will be overwritten with the one you create.
```bash
animal-reduce-data
```

Required arguments: None

Optional arguments:

--num-images: The number of images to keep from each class. Default: 30

--img-size: The width and height to which the images will be reshaped. Default: 224

### 3️⃣ Train Model

Trains a classification model on either the full or reduced dataset. Use arguments for non-interactive scripting. Example for non-interactive training:
```bash
animal-train
```
Required arguments: None
Optional arguments:
--architecture: The architecture of the model to train. Default: VGG16
--dataset-choice: Whether to train on the whole dataset ("full") or on the reduced version ("mini"), either the one included in the wheel or a different one generated (overwritten) by running the reduce-dataset script.
--seed: The seed used for the whole training process. Default: 42
--test-frac: The proportion of data to be held out for test. Default: 0.2
--max-epochs: The max number of epochs for the model to train for. Default: 5
--batch-size: The batch size used by the dataloaders for training. Default: 8

### 4️⃣ Run Inference
Predicts classes for test images using a trained model.

```bash
animal-infer
```

Required arguments: None
Optional arguments:
--model-path: The path to the VGG11 or VGG16 model to use in case it's stored somewhere else and not where the training script saves them by default
--architecture: The architecture of the trained model (VGG11 or VGG16) you want to use for inference. Default: VGG16
--batch-size: The batch size to use in the test dataloader. Default: 16
--num-workers: The number of processes to load data simulatenously. Default: 2
--output-path: The path in which to store the results and metrics from the inference. Default: "inference_outputs/"

### 5️⃣ Launch Dashboard
Opens the interactive visualization and demo dashboard in your browser.

```bash
animal-dashboard
```

## 🧱 Development & Release
If you want to build and publish your own version (requires uv and PyPI credentials):
1. Build the package distribution files
```bash
uv build
```

# 2. Publish to TestPyPI
```bash
uv publish --repository testpypi
```
(Alternatively, for manual upload using twine)
```bash
python -m build
twine upload --repository testpypi dist/*
```

## 👥 Authors
Andrés Malón – Public University of Navarre, Spain
Roberto Aldanondo – Public University of Navarre, Spain

## 📧 Contact:
andresmalon@gmail.com
robertoaj02@gmail.com

## 📝 License
MIT License – see LICENSE file for details.