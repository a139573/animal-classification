# Animal-Classification-Project

In this README, we will explain where the data was obtained from, as well as list the libraries necessary to execute the files and how to run this project.

[![Dataset](https://img.shields.io/badge/🤗_dataset-kaggle-red.svg)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
[![Pypi packages](https://img.shields.io/badge/packages-TestPyPI-blue.svg)](https://test.pypi.org/project/animal-classification/)


# Data Source
The dataset was obtained from the Kaggle website. The dataset is titled **Animal Image Dataset (90 Different Animals)**. This dataset contains 5400 Animal Images Across 90 Diverse Classes


# Libraries to import
To install the necessary dependencies for the environment you can use:

```bash
pip install -r requirements.txt
```

# What is this project
The goal of this project is to develop a model capable of identifying which of the 90 animals is depicted in a given image. The model will be trained on a portion of the dataset, with the objective of learning features that enable it to distinguish between the different animals. Once trained, the model should be able to accurately classify animals in images it has not previously seen.

# How to run the project 
To execute the project, it is necessary to download the compressed folder and then open the Jupyter Notebook named _Project 1 Animals Image Dataset_. Afterward, run all the cells.

# Steps to follow
## 1. Download data
From the root folder, run "animal-download-data
". This will download the full dataset into the subdirectory "data/animals".

## 2. (Optinal) Reduce data
From the root folder, run "animal-reduce-data
". You can adjust the following system arguments:
- Number of images per class (15, 30, or 60). Default: 30
- Target image size (56, 112, or 224). Default: 224
- Path to the original dataset directory. Default: "data/mini_animals/animals"
- Path to save the reduced dataset. Default="data/mini_animals/animals"

# 3. Train the model
From the root folder, run "animal-train". You will be queried to configure the following parameters:
- Use complete dataset or subsample
- Choose seed
- Choose model: VGG11 or VGG16
- Choose number of epochs to train for

# 4. Inference
From the root folder, run "animal-infer". You will be queried to configure the following parameters:
- Use complete dataset or subsample
- Choose model: VGG11 or VGG16


# 👥 Team
## Authors
* Andrés Malón - Public University of Navarre, Spain
* Roberto Aldanondo - Public University of Navarre, Spain 
# 📧 Contact 
For any question or issues, please:
1. Open an issue in this repository
2. Contact one of the corresponding authors: andresmalon@gmail.com, robertoaj02@gmail.com 

