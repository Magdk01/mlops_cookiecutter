from mlops_cookiecutter import MyAwesomeModel
from pytorch_lightning import Trainer


def test_training_runs():
    model = MyAwesomeModel()
    assert model.global_step == 0
    trainer = Trainer(accelerator="gpu", fast_dev_run=10)
    trainer.fit(model)
    assert model.global_step == 10


if __name__ == "__main__":
    test_training_runs()
