import torch
from mlops_cookiecutter_internal.models.model_lightning import MyAwesomeModel
import pytest


def test_model_pass():
    model = MyAwesomeModel()
    test_data = torch.rand([64, 28, 28])
    test_pred = model(test_data)
    assert list(test_pred.shape) == [64, 10]


def test_model_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyAwesomeModel().to(device)
    assert model.device.type == device.type


def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 3D tensor"):
        model(torch.randn(2, 3))


if __name__ == "__main__":
    test_model_pass()
    test_model_device()
    test_error_on_wrong_shape
