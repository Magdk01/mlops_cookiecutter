def test_import():
    import mlops_cookiecutter_internal

    print(dir(mlops_cookiecutter_internal))

    from mlops_cookiecutter_internal import models

    print(dir(models))

    from mlops_cookiecutter_internal.models import model_lightning


if __name__ == "__main__":
    test_import()
