import pytorch_lightning as pl
from torch import nn, optim
from torchvision import models
import pdoc
from my_projects.dataset import AnimalsDataModule, subsample_dir


class VGGNet(pl.LightningModule):
    """
    VGG16-based neural network model for image classification.

    This class implements a VGG16 model pretrained on ImageNet using the torchvision library.
    It supports freezing convolutional layers for transfer learning and replacing the final
    classifier layer to adapt to a custom number of output classes.

    Parameters
    ----------
    num_clases : int, optional
        Number of classes for classification. Default is 90.
    pretrained : bool, optional
        If True, loads pretrained weights on ImageNet. Default is True.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.

    Attributes
    ----------
    vgg : torchvision.models.VGG
        The underlying VGG16 model.
    loss_fn : torch.nn.Module
        Cross-entropy loss function.
    lr : float
        Learning rate for optimizer.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    training_step(batch, batch_idx)
        Training step that computes the loss.
    validation_step(batch, batch_idx)
        Validation step that computes accuracy.
    test_step(batch, batch_idx)
        Test step that computes accuracy.
    configure_optimizers()
        Configures the Adam optimizer with the specified learning rate.
    """
    def __init__(self,num_clases = 90, pretrained=True, lr=1e-3):
        """
        Initialize the VGGNet model.

        Parameters
        ----------
        num_clases : int, optional
            Number of classes for classification. Default is 90.
        pretrained : bool, optional
            If True, use pretrained ImageNet weights. Default is True.
        lr : float, optional
            Learning rate for Adam optimizer. Default is 1e-3.
        """
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)

         # Congelar capas si quieres transfer learning 
        for param in self.vgg.features.parameters(): #congelar --> No actualiza los pesos de las primeras capas convulocionales (Conveniente porque nuestro dataset es pequeño (<10000 imagenes)
                                                    #Así evitamos riesgo de overfitting
            param.requires_grad = False
        
        # Reemplazar la última capa del clasificador
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_clases)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        """
        Forward pass through the VGG16 model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (batch_size, num_clases).
        """
        return self.vgg(x)

    def training_step(self, batch, batch_idx):
        """
        Training step executed during training.

        Parameters
        ----------
        batch : tuple
            Tuple containing input data and labels.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Computed training loss.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        #Guardamos la perdida del entrenamiento
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step executed during validation.

        Parameters
        ----------
        batch : tuple
            Tuple containing input data and labels.
        batch_idx : int
            Index of the batch.
        """
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step executed during testing.

        Parameters
        ----------
        batch : tuple
            Tuple containing input data and labels.
        batch_idx : int
            Index of the batch.
        """
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(1)
        acc = (preds == yb).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer with the specified learning rate.
        """
        return optim.Adam(self.parameters(), lr = self.lr)


if __name__ == '__main__':
    data_module = AnimalsDataModule(data_dir=subsample_dir, batch_size=32)
    data_module.setup()
    num_classes = len(data_module.class_names)
    net = VGGNet(num_clases = num_classes)
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu", devices=1, val_check_interval=3.0, num_sanity_val_steps=0)
    trainer.fit(net, datamodule=data_module)