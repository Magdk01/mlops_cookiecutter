from glob import glob

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


def predict(model: torch.nn.Module, dataloader: torch.utils.data.dataloader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    return torch.cat([model(batch) for batch in dataloader], 0)


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        return data_item


if __name__ == "__main__":
    data_folder = ""
    model_path = ""
    png_in_folder = glob(f"{data_folder}/*.png")
    loader = DataLoader(
        CustomDataset(torch.cat([plt.imread(img) for img in png_in_folder], dim=0)),
        batch_size=16,
    )

    model = torch.load(model_path)

    predict(model, loader)
