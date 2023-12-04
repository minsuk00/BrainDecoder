import torch
from torch.utils.data import Dataset
import os
import random

root_path = ""
dataset_path = os.path.join(root_path, "dataset")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")
eeg_dataset_path = os.path.join(dataset_path, "eeg")


class EEGDataset(Dataset):
    def __init__(self, eeg_dataset_file_name="eeg_5_95_std.pth") -> None:
        super().__init__()
        loaded = torch.load(os.path.join(eeg_dataset_path, eeg_dataset_file_name))
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size = len(self.data)

    def __getitem__(self, idx):
        # t() -> transpose
        eeg = self.data[idx]["eeg"].t()
        eeg = eeg[20:460, :]

        label = self.data[idx]["label"]
        return eeg, label

    def __len__(self):
        return self.size


class SplitDataset(Dataset):
    def __init__(self, dataset, split_name="train") -> None:
        super().__init__()
        self.dataset = dataset

        loaded = torch.load(
            os.path.join(eeg_dataset_path, "block_splits_by_image_all.pth")
        )
        self.target_data_indices = loaded["splits"][0][split_name]
        # filter data that is too short
        self.target_data_indices = [
            i
            for i in self.target_data_indices
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]

        self.size = len(self.target_data_indices)

    def __getitem__(self, idx):
        eeg, label = self.dataset[self.target_data_indices[idx]]
        return eeg, label

    def __len__(self):
        return self.size

    def get_data(self, anchor_label, positive: bool = True):
        cnt = 0
        while True:
            idx = random.choice(self.target_data_indices)
            if positive and self.dataset[idx][1] == anchor_label:
                return self.dataset[idx]
            if not positive and self.dataset[idx][1] != anchor_label:
                return self.dataset[idx]

            if cnt >= 2000:
                raise Exception(f"get_data failed after {cnt} tries")
            cnt += 1
