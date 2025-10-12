# train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn, optim
from torchvision import models
from my_projects.dataset import AnimalsDataModule, subsample_dir
from pathlib import Path
import torch


class VGGNet(pl.LightningModule):
    def __init__(self, num_clases=90, pretrained=True, lr=1e-3):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_clases)
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
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    # --- Prepare data ---
    data_module = AnimalsDataModule(data_dir=subsample_dir, batch_size=32)
    data_module.setup()
    num_classes = len(data_module.class_names)

    # --- Model ---
    net = VGGNet(num_clases=num_classes)

    # --- Make models dir ---
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(models_dir),
        filename="vgg-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    logger = CSVLogger(save_dir="logs", name="vgg_runs")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=5,
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
    trainer.save_checkpoint(str(models_dir / "vgg_last.ckpt"))
    torch.save(net.state_dict(), str(models_dir / "vgg_state_dict.pth"))

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
