# Readme Practice 1

In this README, we will explain where the data was obtained from, as well as list the libraries necessary to execute the files and how to run this project.


# Data Source
The dataset was obtained from the Kaggle website. The dataset is titled **Animal Image Dataset (90 Different Animals)**. The dataset is available at the following URL: [https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals). This dataset contains 5400 Animal Images Across 90 Diverse Classes




# Libraries to import

In this project you need to import this libraries:

 - Torch
 - Torchvision
 - PIL
 - pathlib

# What the project do
The goal of this project is to develop a model capable of identifying which of the 90 animals is depicted in a given image. The model will be trained on a portion of the dataset, with the objective of learning features that enable it to distinguish between the different animals. Once trained, the model should be able to accurately classify animals in images it has not previously seen.

# How to run the project 
To execute the project, it is necessary to download the compressed folder and then open the Jupyter Notebook named _Project 1 Animals Image Dataset_. Afterward, run all the cells.

# How to train the model
To train the model, you should run the following command in the terminal from the project root folder "python -m my_projects.modeling.train". It trains with a subsample that contains 6% of the data.

# How to generate predictions
To train the model, you should run the following command in the terminal from the project root folder "python -m my_projects.modeling.inference".

# Enviroment used
The enviroment that we use in this project is anaconda. The environment py311ml has been used.

**Project developed by Andrés Malón and Roberto Aldanondo**
