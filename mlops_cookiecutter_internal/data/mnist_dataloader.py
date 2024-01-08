import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        target_item = self.targets[idx]
        return data_item, target_item


def mnist(batch_size=64, num_work=1):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    train_dataset = CustomDataset(
        torch.load("data/processed/train_images.pt"),
        torch.load("data/processed/train_target.pt"),
    )
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_work)

    test_dataset = CustomDataset(
        torch.load("data/processed/test_images.pt"),
        torch.load("data/processed/test_target.pt"),
    )
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_work)
    return train, test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train, test = mnist()
    for batch in train:
        print(batch)
        for i in range(5):
            plt.imshow(batch[0][i])
            plt.show()
        break
