from pytorch_lightning import LightningModule
from torch import nn, optim

from mlops_cookiecutter_internal.data.mnist_dataloader import mnist


class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(nn.Linear(64 * 7 * 7, 1000), nn.ReLU(), nn.Linear(1000, 10))

        self.criterium = nn.CrossEntropyLoss()

        self.train_set, self.test_set = mnist(batch_size=32, num_work=23)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input to a 3D tensor")
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def train_dataloader(self):
        return self.train_set

    def test_dataloader(self):
        return self.test_set
