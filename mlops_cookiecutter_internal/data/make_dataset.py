from glob import glob

import torch


def collocate_and__save_files(split, type):
    data = [torch.load(file) for file in sorted(glob(f"data/raw/corruptmnist/{split}_{type}*"))]
    concatenated_data = torch.cat(data, dim=0)

    if type == "images":  # Normalize input only
        concatenated_data = (concatenated_data - torch.mean(concatenated_data)) / torch.std(concatenated_data)
    torch.save(concatenated_data, f"data/processed/{split}_{type}.pt")


if __name__ == "__main__":
    # Get the data and process it

    collocate_and__save_files("train", "images")
    collocate_and__save_files("train", "target")

    collocate_and__save_files("test", "images")
    collocate_and__save_files("test", "target")
