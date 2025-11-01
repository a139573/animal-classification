# 🐾 Animal Classification Project

[![Dataset](https://img.shields.io/badge/🤗_dataset-kaggle-red.svg)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
[![PyPI](https://img.shields.io/badge/PyPI-TestPyPI-blue.svg)](https://test.pypi.org/project/animal-classification/)

---

## 📘 Overview
This project trains VGG-based deep learning models to classify images of 90 animal species.
It uses PyTorch and PyTorch Lightning for modeling and provides command-line tools for:

- Data downloading and preprocessing

- Model training and inference

- Visualization and dashboards via Gradio

The project follows a modular cookiecutter-style structure, with code organized inside animal_classification/my_projects/.

---

## 📂 Dataset
Dataset: **[Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)**  
Contains **5,400 images** across **90 classes**.

---

## ⚙️ Installation Options

### Option 1 – Install from TestPyPI
```bash
uv pip install -i https://test.pypi.org/simple/ animal-classification
```

### Option 2 – Install from local wheel
If you downloaded the .whl file (for example animal_classification-0.2.2-py3-none-any.whl):
```bash
uv pip install dist/animal_classification-0.2.2-py3-none-any.whl
```

### Option 3 - Local environment setup
Create a virtual environment and install dependencies:
```bash
uv venv .venv
source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
uv pip install -r requirements.txt
```

## 🚀 Usage

After installation, several CLI commands are available:

### 1️⃣ Download data
```bash
animal-download-data
```

Downloads the dataset into data/animals/.

### 2️⃣ (Optional) Reduce data
```bash
animal-reduce-data
```

Arguments:

--num-images [15|30|60] (default: 30)

--img-size [56|112|224] (default: 224)

--input-dir (default: "data/animals")

--output-dir (default: "data/mini_animals/animals")

### 3️⃣ Train model
```bash
animal-train
```

Interactive prompts:

Choose dataset (full or reduced)

Select model (VGG11 or VGG16)

Set seed and epochs

### 4️⃣ Run inference
```bash
animal-infer
```

Predict classes for test images using trained models.

### 5️⃣ Launch dashboard
```bash
animal-dashboard
```

Opens an interactive visualization dashboard (Gradio-based).

## 🧱 Development

If you want to build and publish your own version:

```bash
uv build
uv publish --repository testpypi
```


If needed, you can still upload manually:

```bash
python -m build
twine upload --repository testpypi dist/*
```

## 🧩 Project Structure
```
animal_classification/
│
├── my_projects/
│   ├── dataset.py
│   ├── download_data.py
│   ├── reduce_data.py
│   ├── plots.py
│   ├── modeling/
│   └── scripts/
│
├── notebooks/
├── reports/
├── pyproject.toml
└── README.md
```

## 👥 Authors

Andrés Malón – Public University of Navarre, Spain

Roberto Aldanondo – Public University of Navarre, Spain

## 📧 Contact:

andresmalon@gmail.com

robertoaj02@gmail.com

## 📝 License

MIT License – see LICENSE file for details.
