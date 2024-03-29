from pytorch_lightning import Trainer

from mlops_cookiecutter_internal.models.model_lightning import MyAwesomeModel


def test_training_runs():
    model = MyAwesomeModel()
    assert model.global_step == 0
    trainer = Trainer(fast_dev_run=10)
    trainer.fit(model)
    assert model.global_step == 10


if __name__ == "__main__":
    test_training_runs()
