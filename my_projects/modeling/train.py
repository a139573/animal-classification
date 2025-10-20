import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn, optim
from torchvision import models
from my_projects.dataset import AnimalsDataModule
from pathlib import Path
import torch
import os


class VGGNet(pl.LightningModule):
    def __init__(self, architecture="vgg16", num_classes=90, pretrained=True, lr=1e-3):
        super().__init__()
        # --- Choose model architecture dynamically ---
        if architecture.lower() == "vgg16":
            self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif architecture.lower() == "vgg11":
            self.vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # --- Freeze feature extractor ---
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # --- Replace final classifier layer ---
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

        # --- Loss and learning rate ---
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.vgg(x)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def main():
    # --- Ask to choose dataset ---
    dataset_choice = ""
    data_dir = os.path.join(os.getcwd(), "data")
    while dataset_choice not in ["1", "2"]:
        print("Select the dataset to use:")
        print("1: Full dataset (animals)")
        print("2: Reduced dataset (mini_animals)")
        dataset_choice = input("Enter 1 or 2: ")
    subsample_dir = os.path.join(data_dir, "animals" if dataset_choice == "1" else "mini_animals")

    # --- Ask for seed ---
    while True:
        try:
            seed = int(input("Enter random seed (integer): "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    # --- Ask for test fraction ---
    while True:
        try:
            test_frac = float(input("Enter test fraction (0.05 to 0.5): "))
            if 0.05 <= test_frac <= 0.5:
                break
            else:
                print("Test fraction must be between 0.05 and 0.5")
        except ValueError:
            print("Please enter a valid number.")

    # --- Ask user for model architecture ---
    architecture = ""
    while architecture.lower() not in ["vgg16", "vgg11"]:
        architecture = input("Choose model architecture (vgg16 or vgg11): ")

    # --- Ask user for number of epochs ---
    while True:
        try:
            max_epochs = int(input("Enter the number of epochs (positive integer): "))
            if max_epochs > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

    # --- Prepare data ---
    data_module = AnimalsDataModule(
        data_dir=subsample_dir,
        batch_size=32,
        seed=seed,
        test_frac=test_frac
    )
    data_module.setup()
    num_classes = len(data_module.class_names)

    # --- Instantiate model ---
    net = VGGNet(architecture=architecture, num_classes=num_classes)

    # --- Make models dir ---
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Callbacks (keep only best model to save space) ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(models_dir),
        filename=f"{architecture}-best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,       # ✅ only keep best model
        save_last=False,    # ❌ don't save 'last.ckpt'
        verbose=False,
    )

    logger = CSVLogger(save_dir="logs", name=f"{architecture}_runs")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        val_check_interval=10.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=100,
    )

    # --- Train ---
    trainer.fit(net, datamodule=data_module)

    # --- Save final artifacts ---
    torch.save(net.state_dict(), str(models_dir / f"{architecture}_state_dict.pth"))

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
