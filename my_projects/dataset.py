# Load the dataset

import os
from pathlib import Path
import pdoc
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(123)

project_dir = Path(os.getcwd()).parent.absolute()
data_dir = os.path.join(os.getcwd(), "data")
subsample_dir = os.path.join(data_dir, "mini_animals")


class AnimalsDataModule(pl.LightningDataModule):
    """
    Data module para cargar y preparar un dataset de imágenes de animales.

    Utiliza `torchvision.datasets.ImageFolder` para cargar imágenes organizadas
    en carpetas por clase. Divide el dataset en conjuntos de entrenamiento,
    validación y prueba. Aplica transformaciones estándar para modelos pretrained.

    Parámetros
    ----------
    data_dir : str, optional
        Directorio raíz donde se encuentran las imágenes organizadas en subcarpetas por clase.
        Por defecto es el directorio actual (".").
    batch_size : int, optional
        Tamaño del batch para los dataloaders. Por defecto es 32.
    num_workers : int, optional
        Número de procesos paralelos para cargar datos. Por defecto es 8.

    Atributos
    ---------
    data_dir : str
        Directorio raíz del dataset.
    batch_size : int
        Tamaño de batch para cargar datos.
    num_workers : int
        Número de procesos para carga paralela.
    transform : torchvision.transforms.Compose
        Transformaciones que se aplican a las imágenes.
    class_names : list
        Lista con nombres de las clases detectadas en el dataset.
    train_data : torch.utils.data.Dataset
        Subconjunto del dataset para entrenamiento.
    val_data : torch.utils.data.Dataset
        Subconjunto del dataset para validación.
    test_data : torch.utils.data.Dataset
        Subconjunto del dataset para prueba.

    Métodos
    -------
    setup(stage=None)
        Prepara y divide el dataset en train, val y test según la etapa.
    train_dataloader()
        Retorna un DataLoader para los datos de entrenamiento.
    val_dataloader()
        Retorna un DataLoader para los datos de validación.
    test_dataloader()
        Retorna un DataLoader para los datos de prueba.
    """
    def __init__(self, data_dir=".", batch_size=32, num_workers=8):
        """
        Inicializa el DataModule con las configuraciones básicas.

        Parámetros
        ----------
        data_dir : str, optional
            Ruta al directorio con los datos de imágenes. Por defecto es ".".
        batch_size : int, optional
            Tamaño del batch para los dataloaders. Por defecto es 32.
        num_workers : int, optional
            Número de workers para cargar datos en paralelo. Por defecto es 8.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        """
        Carga y divide el dataset en conjuntos de entrenamiento, validación y prueba.

        Separa el dataset original en 60% entrenamiento, 20% validación y 20% prueba.

        Parámetros
        ----------
        stage : str, optional
            Etapa actual ('fit', 'validate', 'test') para preparar los datos correspondientes.
            Si es None, prepara todos los splits.
        """
        full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.class_names = full_dataset.classes

        train_size = int(len(full_dataset) * 0.6)
        val_size = int(len(full_dataset) * 0.2)
        test_size = len(full_dataset) - train_size - val_size

        # self.train_data, self.val_data, self.test_data = random_split(
        #         full_dataset, [train_size, val_size, test_size]
        #     )
        if stage == "fit" or stage is None:
            self.train_data, self.val_data, _ = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        if stage == "validate" or stage is None:
            _, self.val_data, _ = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        if stage == "test" or stage is None:
            _, _, self.test_data = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        """
        DataLoader para el conjunto de entrenamiento.

        Retorna
        -------
        torch.utils.data.DataLoader
            DataLoader configurado para el conjunto de entrenamiento.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        DataLoader para el conjunto de validación.

        Retorna
        -------
        torch.utils.data.DataLoader
            DataLoader configurado para el conjunto de validación.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        DataLoader para el conjunto de prueba.

        Retorna
        -------
        torch.utils.data.DataLoader
            DataLoader configurado para el conjunto de prueba.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )


data_module = AnimalsDataModule(data_dir=subsample_dir)