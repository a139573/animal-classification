import os
import sys

# REMOVED LINES that were forcing CPU mode:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["PYTORCH_NO_CUDA"] = "1"

import torch
# Now safe to import PyTorch Lightning
import lightning.pytorch as pl # MARK: 1. Changed import to lightning.pytorch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# The rest of your imports
from torch import nn, optim
from torchvision import models
from ..dataset import AnimalsDataModule
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import tempfile

import importlib.resources as pkg_resources


#os.environ.setdefault("TORCH_LOGS", "OFF")
#os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") # MARK: 2. Removed redundant commented env var
#os.environ.setdefault("PYTORCH_NO_CUDA", "1") # MARK: 3. Removed redundant commented env var


torch.set_float32_matmul_precision("medium")

# --- Safe device selection, avoid CUDA issues ---
def get_accelerator():
    # CUDA available?
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            torch.cuda.get_device_properties(0)
            return "gpu"
        except Exception as e:
            print(f"⚠️ CUDA available but broken: {e}")
            return "cpu"
    # no CUDA at all
    return "cpu"




def get_packaged_mini_data_path():
    """Locates the 'mini_animals' dataset inside the installed package."""
    # Use files() to correctly reference the path inside site-packages
    try:
        return pkg_resources.files('animal_classification').joinpath('data/mini_animals/animals')
    except Exception:
        # Fallback for local dev setup where package is not installed as a wheel
        return Path("data/mini_animals/animals")





class VGGNet(pl.LightningModule):
    """
    A VGG-based classifier implemented using PyTorch Lightning.

    This model uses a pre-trained VGG backbone (frozen features) and a custom
    classifier head for the specific number of animal classes in the dataset.
    """
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
        """Standard CrossEntropyLoss for classification tasks."""
        self.lr = lr
        """Learning rate used by the optimizer."""

    def forward(self, x):
        return self.vgg(x)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class GradioProgressCallback(pl.Callback):
    """
    Custom PyTorch Lightning callback to report training progress to the Gradio UI.

    This callback hooks into the training loop and updates a Gradio progress bar
    after every batch, calculating the overall percentage based on the total
    number of epochs and batches.
    """
    def __init__(self, progress_fn, max_epochs):
        """
        Initializes the callback.

        Parameters
        ----------
        progress_fn : callable
            The Gradio progress update function (usually `gradio.Progress()`).
        max_epochs : int
            The total number of epochs to train for. Used to calculate
            the completion percentage.
        """
        super().__init__()
        self.progress_fn = progress_fn
        """Gradio function to update the UI progress bar."""
        self.max_epochs = max_epochs
        """Total number of epochs scheduled for training."""
        self.total_batches = None
        """Cached number of batches per epoch (lazy-loaded)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Updates the Gradio progress bar at the end of each training batch.

        Calculates the global step count and updates the progress percentage
        and description string in the UI.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The model being trained.
        outputs : dict
            The outputs of the training step.
        batch : Any
            The current batch of data.
        batch_idx : int
            The index of the current batch.
        """
        # Get total batches once (Lightning has dataloader ready now)
        if self.total_batches is None:
            self.total_batches = len(trainer.train_dataloader)

        total_batches = self.total_batches

        current_epoch = trainer.current_epoch
        global_step = current_epoch * total_batches + batch_idx + 1
        total_steps = self.max_epochs * total_batches

        pct = global_step / total_steps

        self.progress_fn(
            pct,
            desc=f"Epoch {current_epoch+1}/{self.max_epochs} — batch {batch_idx+1}/{total_batches}"
        )



def main(
    architecture: str = "vgg16",
    dataset_choice: str = "mini",
    seed: int = 42,
    test_frac: float = 0.2,
    max_epochs: int = 5,
    batch_size: int = 16,
    num_workers: int = 2,
    is_demo: bool = False,
    progress: gr.Progress = None,
):
    """
    Main training pipeline for the Animal Classification project.

    This function handles the setup of the DataModule, the initialization of the
    model (VGGNet), and the training loop using PyTorch Lightning. It creates
    necessary directories for models, logs, and reports.

    Parameters
    ----------
    architecture : str
        The model architecture to use ('vgg16' or 'vgg11').
    dataset_choice : str
        Which dataset to use ('full' or 'mini').
    seed : int
        Random seed for reproducibility.
    test_frac : float
        Fraction of data to use for testing.
    max_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for data loaders.
    num_workers : int
        Number of CPU workers for data loading.
    is_demo : bool
        If True, runs in 'demo mode' (no saving to disk, returns objects).
    progress : gradio.Progress, optional
        Gradio progress tracker object for the dashboard.

    Returns
    -------
    dict
        A dictionary containing the trained model, metrics, and paths/objects
        for visualization.
    """
    accelerator = get_accelerator()
    use_gpu = accelerator == "gpu"

    print("🟢 Using GPU" if use_gpu else "🟡 Falling back to CPU")


    # If is demo, always use packaged data
    if is_demo:
        subsample_dir = get_packaged_mini_data_path()
        print(f"📦 DEMO MODE: Using packaged mini dataset from: {subsample_dir}")

    # If not demo (CLI execution), check for local availability
    else:
        # 1. Determine local paths based on user choice
        base_data_dir = Path("data")
        
        if dataset_choice == "full":
            local_target_path = base_data_dir / "animals/animals"
            dataset_name = "FULL"
        else: # "mini"
            local_target_path = base_data_dir / "mini_animals/animals"
            dataset_name = "MINI"

        # 2. Check if the local data exists
        if local_target_path.exists():
            subsample_dir = local_target_path
            print(f"📁 CLI MODE: Using local {dataset_name} dataset from: {subsample_dir}")
        else:
            # 3. Fallback: Local data not found, use the packaged mini dataset instead
            subsample_dir = get_packaged_mini_data_path()
            print(f"⚠️ Local {dataset_name} dataset not found at {local_target_path}. Falling back to packaged MINI dataset from: {subsample_dir}")

    # IMPORTANT: Assumes AnimalsDataModule is defined elsewhere
    from ..dataset import AnimalsDataModule 

    data_module = AnimalsDataModule(
        data_dir=subsample_dir,
        batch_size=batch_size,
        seed=seed,
        test_frac=test_frac,
    )
    data_module.setup()
    num_classes = len(data_module.class_names)

    net = VGGNet(architecture=architecture, num_classes=num_classes)

    callbacks = []

    
    if is_demo:
        tmp_dir = tempfile.TemporaryDirectory()
        logs_dir = Path(tmp_dir.name)
        callbacks.append(GradioProgressCallback(progress, max_epochs))
    else:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=models_dir,
            filename=f"{architecture}-best",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)

    logger = CSVLogger(save_dir=logs_dir, name=f"{architecture}_runs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if use_gpu else 32,   # FP16 only on GPU
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(net, datamodule=data_module)

    # Optional: run test set if available. NO, this on inference
    # test_results = trainer.test(net, datamodule=data_module)

    # Save model
    if not is_demo:
        torch.save(net.state_dict(), models_dir / f"{architecture}_state_dict.pth")
        print(f"✅ Best checkpoint: {checkpoint_callback.best_model_path}")

    # Load CSV logs for visualization
    csv_path = Path(logger.log_dir) / "metrics.csv"
    df = pd.read_csv(csv_path)
    # Aggregate per epoch to fix multiple steps per epoch issue
    train_loss_per_epoch = df.groupby('epoch')['train_loss'].mean().tolist()
    val_loss_per_epoch = df.groupby('epoch')['val_loss'].mean().tolist()
    val_acc_per_epoch = df.groupby('epoch')['val_acc'].mean().tolist()


    if not is_demo:
        # --- Permanent mode: save plots ---
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, marker="o", label="Training Loss")
        plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, marker="x", label="Validation Loss")
        plt.plot(range(len(val_acc_per_epoch)), val_acc_per_epoch, marker="s", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{architecture.upper()} Convergence")
        plt.legend()
        plt.grid(True)
        plt.savefig(reports_dir / f"{architecture}_convergence.png")
        plt.close()

    # Extract final metrics
    final_val_acc = df["val_acc"].dropna().iloc[-1]
    # test_acc = test_results[0]["test_acc"] if test_results else None

    return {
        "model": net,
        "val_acc": float(final_val_acc) if final_val_acc is not None else None,
        # "test_acc": float(test_acc) if test_acc is not None else None,
        "log_path": str(csv_path),
       #  "plot_path": str(conv_plot_path) if conv_plot_path else None,
        "val_acc_list": val_acc_per_epoch if is_demo else None,
        "val_loss_list": val_loss_per_epoch if is_demo else None,
        "train_loss_list": train_loss_per_epoch if is_demo else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VGG model on the Animals dataset."
    )
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
        is_demo=False,
    )