import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn, optim
from torchvision import models
from ..dataset import AnimalsDataModule
from pathlib import Path
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

torch.set_float32_matmul_precision("medium")


class VGGNet(pl.LightningModule):
    def __init__(self, architecture="vgg16", num_classes=90, pretrained=True, lr=1e-3):
        super().__init__()
        if architecture.lower() == "vgg16":
            self.vgg = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif architecture.lower() == "vgg11":
            self.vgg = models.vgg11(
                weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        for param in self.vgg.features.parameters():
            param.requires_grad = False

        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

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


import tempfile
import contextlib

def main(
    architecture: str = "vgg16",
    dataset_choice: str = "mini",
    seed: int = 42,
    test_frac: float = 0.2,
    max_epochs: int = 5,
    batch_size: int = 16,
    progress: gr.Progress = None,
    is_demo: bool = False,  # 👈 add this flag
):
    data_dir = Path("data")
    subsample_dir = data_dir / ("animals/animals" if dataset_choice == "full" else "mini_animals/animals")

    data_module = AnimalsDataModule(
        data_dir=subsample_dir,
        batch_size=batch_size,
        seed=seed,
        test_frac=test_frac,
    )
    data_module.setup()
    num_classes = len(data_module.class_names)

    net = VGGNet(architecture=architecture, num_classes=num_classes)

    # 👇 handle directories conditionally
    if is_demo:
        temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(temp_dir.name)
        models_dir = base_path / "models"
        logs_dir = base_path / "logs"
        reports_dir = base_path / "reports/figures"
    else:
        models_dir = Path("models")
        logs_dir = Path("logs")
        reports_dir = Path("reports/figures")

    # ensure dirs exist
    for d in [models_dir, logs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        filename=f"{architecture}-best",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    logger = CSVLogger(save_dir=logs_dir, name=f"{architecture}_runs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    if progress:
        progress(0, desc="Starting training...")

    trainer.fit(net, datamodule=data_module)
    test_results = trainer.test(net, datamodule=data_module)

    if progress:
        progress(1, desc="Training complete!")

    # Save model (only permanent if not demo)
    model_path = models_dir / f"{architecture}_state_dict.pth"
    torch.save(net.state_dict(), model_path)
    print(f"✅ Best checkpoint: {checkpoint_callback.best_model_path}")

    # Load CSV logs for visualization
    csv_path = Path(logger.log_dir) / "metrics.csv"
    df = pd.read_csv(csv_path)

    # Plot validation accuracy
    if "val_acc" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{architecture.upper()} Validation Accuracy")
        plt.legend()
        plt.grid(True)
        acc_plot_path = reports_dir / f"{architecture}_val_acc.png"
        plt.savefig(acc_plot_path)
        plt.close()
    else:
        acc_plot_path = None

    # Extract final metrics
    final_val_acc = df["val_acc"].dropna().iloc[-1] if "val_acc" in df.columns else None
    test_acc = test_results[0]["test_acc"] if test_results else None

    if is_demo:
        result = {
            "model": net,         # trained PyTorch Lightning model
            "data_module": data_module,  # optional, so inference can reuse val loader
            "metrics": {'val_acc': final_val_acc, 'test_acc': test_acc}    # dictionary with validation/test metrics and paths
        }
    else:
        result = {
            "val_acc": float(final_val_acc) if final_val_acc is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "log_path": str(csv_path),
            "plot_path": str(acc_plot_path) if acc_plot_path else None,
        }

    # 👇 Clean up temporary directories automatically
    if is_demo:
        temp_dir.cleanup()

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VGG model on the Animals dataset.")
    parser.add_argument("--architecture", default="vgg16", choices=["vgg16", "vgg11"])
    parser.add_argument("--dataset-choice", default="mini", choices=["full", "mini"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    main(
        architecture=args.architecture,
        dataset_choice=args.dataset_choice,
        seed=args.seed,
        test_frac=args.test_frac,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        is_demo=False
    )
