"""
# 🚂 Model Training Orchestrator

This module handles the end-to-end training pipeline using **PyTorch Lightning**.

It is designed to be run in two modes:
1.  **CLI Mode:** From the terminal, saving results to `models/` and `logs/`.
2.  **Demo Mode:** From the Dashboard, keeping results in memory.

## 💻 CLI Usage
You can train a model directly from your terminal:

```bash
# Train VGG16 for 10 epochs on the mini dataset
python -m animal_classification.modeling.train --architecture vgg16 --max-epochs 10 --lr 1e-4
```
"""

import argparse 
import tempfile 
import os
from pathlib import Path

import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import lightning.pytorch as pl 
import gradio as gr 
from lightning.pytorch.callbacks import ModelCheckpoint 
from lightning.pytorch.loggers import CSVLogger

# --- Local Project Imports ---
from animal_classification.modeling.architecture import VGGNet
from animal_classification.modeling.metrics import plot_training_curves 
from animal_classification.preprocessing import AnimalsDataModule 
from animal_classification.utils import get_accelerator, get_packaged_mini_data_path

# Set precision for Tensor Cores (Performance Optimization for Blackwell/Ada GPUs)
torch.set_float32_matmul_precision("medium")

class GradioProgressCallback(pl.Callback): 
    """ 
    Lightning callback to synchronize training progress with the Gradio UI.

    This class hooks into the training loop's batch-end events to calculate 
    the overall completion percentage and update the progress bar provided 
    by the Gradio web interface.

    Attributes
    ----------
    progress_fn : gradio.Progress\n
        The progress tracker instance injected by the Gradio interface.\n
    max_epochs : int\n
        The total number of epochs configured for the training run.\n
    total_batches : int\n
        Cached count of training batches per epoch.
    """
    def __init__(self, progress_fn, max_epochs):
        super().__init__()
        self.progress_fn = progress_fn
        self.max_epochs = max_epochs
        self.total_batches = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.total_batches is None:
            self.total_batches = len(trainer.train_dataloader)

        current_epoch = trainer.current_epoch
        # Calculate global progress
        total_steps = self.max_epochs * self.total_batches
        global_step = current_epoch * self.total_batches + batch_idx + 1
        pct = global_step / total_steps

        self.progress_fn(
            pct,
            desc=f"Epoch {current_epoch+1}/{self.max_epochs} — batch {batch_idx+1}/{self.total_batches}"
        )

def train_model( architecture: str = "vgg16", dataset_choice: str = "mini", seed: int = 42, test_frac: float = 0.2, max_epochs: int = 5, batch_size: int = 16, lr: float = 1e-3, num_workers: int = 2, is_demo: bool = False, progress: gr.Progress = None, ): 
    """ Executes the training pipeline.

    Parameters
    ----------
    architecture : str.\n
        The model backbone to use. Options: `'vgg16'`, `'vgg11'`.\n
    dataset_choice : str.\n
        Which dataset folder to use. Options: `'mini'` (default), `'full'`.\n
    seed : int.\n
        Random seed for reproducibility.\n
    test_frac : float.\n
        Fraction of data to hold out for testing.\n
    max_epochs : int.\n
        Total number of training epochs.\n
    batch_size : int.\n
        Number of images per training batch.\n
    num_workers : int.\n
        Number of CPU subprocesses for data loading.\n
    is_demo : bool.\n
        If `True`, runs in Dashboard mode (in-memory).\n
    progress : gradio.Progress, optional.
        Gradio progress tracker.

    Returns
    -------
    dict
        A dictionary containing training metrics and the model object.
    """
    # Normalize architecture string to lowercase
    architecture = architecture.lower()

    # 1. Setup Hardware
    accelerator = get_accelerator()
    use_gpu = accelerator == "gpu"
    print(f"{'🟢 Using GPU' if use_gpu else '🟡 Falling back to CPU'}")

    # 2. Resolve Data Path
    if is_demo:
        subsample_dir = get_packaged_mini_data_path()
        print(f"📦 DEMO MODE: Using packaged mini dataset from: {subsample_dir}")
    else:
        base_data_dir = Path("data")
        local_target = base_data_dir / ("animals/animals" if dataset_choice == "full" else "mini_animals/animals")
        
        if local_target.exists():
            subsample_dir = local_target
            print(f"📁 CLI MODE: Using local dataset from: {subsample_dir}")
        else:
            subsample_dir = get_packaged_mini_data_path()
            print(f"⚠️ Local dataset not found. Falling back to: {subsample_dir}")

    # 3. Initialize Data Module
    data_module = AnimalsDataModule(
        data_dir=subsample_dir,
        batch_size=batch_size,
        seed=seed,
        test_frac=test_frac,
        num_workers=num_workers
    )
    data_module.setup()

    # 4. Initialize Model with custom Learning Rate
    net = VGGNet(
        architecture=architecture, 
        num_classes=len(data_module.class_names),
        lr=lr
    )

    # 5. Setup Callbacks & Loggers
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

    # 6. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if use_gpu else 32,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    # 7. Start Training
    trainer.fit(net, datamodule=data_module)

    # 8. Post-Training: Save & Plot
    if not is_demo:
        torch.save(net.state_dict(), models_dir / f"{architecture}_state_dict.pth")
        print(f"✅ Best checkpoint saved: {checkpoint_callback.best_model_path}")

    # Process logs for return values
    csv_path = Path(logger.log_dir) / "metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        metrics = df.groupby('epoch').mean()
        
        train_loss = metrics['train_loss'].dropna().tolist()
        val_loss = metrics['val_loss'].dropna().tolist()
        val_acc = metrics['val_acc'].dropna().tolist()
        final_acc = val_acc[-1] if val_acc else 0.0

        plot_training_curves(train_loss, val_loss, val_acc, architecture_name=architecture, output_path=None, is_demo=is_demo)
    else:
        final_acc = 0.0
        train_loss, val_loss, val_acc = [], [], []

    return {
        "model": net,
        "val_acc": float(final_acc),
        "log_path": str(csv_path),
        "val_acc_list": val_acc if is_demo else None,
        "val_loss_list": val_loss if is_demo else None,
        "train_loss_list": train_loss if is_demo else None,
    }

def main():
    """ 
    Command-line interface entry point. 
    This parses arguments and then calls the logic in train_model().
    """
    parser = argparse.ArgumentParser(description="Train Animal Classification Model") 
    parser.add_argument("--architecture", default="vgg16", choices=["vgg16", "vgg11"], help="Backbone (vgg16 or vgg11)") 
    parser.add_argument("--dataset-choice", default="mini", choices=["full", "mini"], help="Data subset") 
    parser.add_argument("--seed", type=int, default=42) 
    parser.add_argument("--test-frac", type=float, default=0.2) 
    parser.add_argument("--max-epochs", type=int, default=5) 
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    print(f"🚀 Training Triggered via CLI. Architecture: {args.architecture}")

    # Call logic with all parsed arguments
    train_model(
        architecture=args.architecture,
        dataset_choice=args.dataset_choice,
        seed=args.seed,
        test_frac=args.test_frac,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()