import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from PIL import Image
import cv2

root_path = "/Users/ms/cs/ML/NeuroImagen/"
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
        img_name = self.images[self.data[idx]["image"]]
        return eeg, label, img_name

    def __len__(self):
        return self.size


class Splitter(Dataset):
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
        self.all_labels = np.array(self.get_all_labels())
        self.all_eegs = self.get_all_eegs()

    def __getitem__(self, idx):
        eeg, label, img_name = self.dataset[self.target_data_indices[idx]]
        return eeg, label, img_name

    def __len__(self):
        return self.size

    def get_all_labels(self):
        data = [self.dataset[idx] for idx in self.target_data_indices]
        return [item[1] for item in data]

    def get_all_eegs(self):
        data = [self.dataset[idx] for idx in self.target_data_indices]
        return [item[0] for item in data]

    def generate_data_points(self, anchor_labels, positive=True):
        eeg_shape = self.__getitem__(0)[0].size()
        eegs = torch.empty(0, eeg_shape[0], eeg_shape[1])
        labels = torch.empty(0)
        for anchor_label in anchor_labels:
            indices = (
                np.argwhere(self.all_labels == anchor_label.item())[:, 0]
                if positive
                else np.argwhere(self.all_labels != anchor_label.item())[:, 0]
            )
            data_idx = np.random.choice(indices)
            eeg = self.all_eegs[data_idx]
            eegs = torch.cat((eegs, eeg.unsqueeze(dim=0)))
            labels = torch.cat(
                (labels, torch.tensor(self.all_labels[data_idx]).unsqueeze(dim=0))
            )

        return eegs, labels

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


class EEGImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        eeg, _, img_name = self.dataset[idx]

        # read img
        img_path = os.path.join(
            images_dataset_path, img_name.split("_")[0], img_name + ".jpeg"
        )
        img = Image.open(img_path).convert("RGB")
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return eeg, img
