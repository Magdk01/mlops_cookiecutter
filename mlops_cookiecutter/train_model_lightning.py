import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from mlops_cookiecutter.data.mnist_dataloader import mnist
from models.model_lightning import MyAwesomeModel
from pytorch_lightning import Trainer
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model = MyAwesomeModel()
    trainer = Trainer(
        accelerator="gpu",
        # limit_train_batches=0.2,
        max_epochs=10,
        logger=pl.loggers.WandbLogger(project="dtu_mlops"),
    )
    trainer.fit(model)

    trainer.test(model)
