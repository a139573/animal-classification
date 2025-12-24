import lightning.pytorch as pl
from torchvision import models
from torch import nn
from torch import optim
import torch

class VGGNet(pl.LightningModule):
    """
    A VGG-based classifier implemented using PyTorch Lightning.

    This model uses a pre-trained VGG backbone (frozen features) and a custom
    classifier head for the specific number of animal classes in the dataset.
    """
    def __init__(self, architecture="vgg16", num_classes=90, pretrained=True, lr=1e-3):
        """
        Args:
            architecture (str): Backbone name ('vgg16' or 'vgg11').
            num_classes (int): Number of output categories.
            pretrained (bool): Whether to use ImageNet weights.
            lr (float): Learning rate for the Adam optimizer.
        """
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

        # Freeze the backbone
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # Replace the classifier head
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        """Standard CrossEntropyLoss for classification tasks."""
        self.lr = lr
        """Learning rate used by the optimizer."""

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing a batch of images. 
            Shape: (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The raw output logits from the classifier head.
            Shape: (batch_size, num_classes).
        """
        return self.vgg(x)

    def training_step(self, batch, batch_idx):
        """
        Computes the training loss.
        
        Logs 'train_loss' to the progress bar and logger.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes validation metrics (Accuracy and Loss).
        
        Logs 'val_acc' and 'val_loss'.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Computes test accuracy on unseen data.
        
        Logs 'test_acc'.
        """
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Sets up the Adam optimizer with the specified learning rate.
        """
        return optim.Adam(self.parameters(), lr=self.lr)