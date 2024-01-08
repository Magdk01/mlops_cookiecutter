from torch.utils.data import DataLoader, Dataset
from mlops_cookiecutter_internal.data.mnist_dataloader import mnist
import torch
import pytest


@pytest.mark.parametrize(
    "test_batch,expected_batch", [(8, 8), (6, 6), (42, 42), (64, 64)]
)
def test_data(test_batch, expected_batch):
    train, test = mnist(batch_size=test_batch)
    assert isinstance(
        train, DataLoader
    ), "Train dataloader not an instance of dataloader"
    assert isinstance(test, DataLoader)

    assert (
        train.dataset.__len__() == 30000
    ), "Train split did not have the correct number of samples"
    assert test.dataset.__len__() == 5000
    assert torch.unique(test.dataset.targets).tolist() == list(range(10))

    inputs, classes = next(iter(train))

    assert list(inputs.shape) == [expected_batch, 28, 28]

    assert classes.dtype == torch.int64


if __name__ == "__main__":
    test_data()
