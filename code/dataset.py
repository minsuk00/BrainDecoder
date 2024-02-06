import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
from PIL import Image
import pickle

# import cv2

root_path = "/home/choi/BrainDecoder/"
dataset_path = os.path.join(root_path, "dataset")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")
eeg_dataset_path = os.path.join(dataset_path, "eeg")


class EEGDataset(Dataset):
    def __init__(
        self, eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth"
    ) -> None:
        super().__init__()
        loaded = torch.load(os.path.join(eeg_dataset_path, eeg_dataset_file_name))
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size = len(self.data)

    def __getitem__(self, idx):
        # t() -> transpose
        eeg = self.data[idx]["eeg"].t().to(torch.float)
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
    def __init__(
        self, dataset, transform=None, init_imgdict=False, stochastic_transform=False
    ):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.img_dict = {}
        self.stochastic_transform = stochastic_transform

        if init_imgdict:
            for idx in range(self.__len__()):
                _, _, img_name = self.dataset[idx]
                img_path = os.path.join(
                    images_dataset_path, img_name.split("_")[0], img_name + ".JPEG"
                )
                img = Image.open(img_path).convert("RGB")
                # img = Image.open(img_path)
                # img.draft("RGB", (128, 128))
                # img = img.load()
                if not stochastic_transform and self.transform:
                    img = self.transform(img)
                self.img_dict[img_name] = img

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        eeg, label, img_name = self.dataset[idx]

        if img_name in self.img_dict:
            img = self.img_dict[img_name]
        else:
            # read img
            img_path = os.path.join(
                images_dataset_path, img_name.split("_")[0], img_name + ".JPEG"
            )
            img = Image.open(img_path).convert("RGB")
            # img = Image.open(img_path)
            # img.draft("RGB", (128, 128))
            # img = img.load()
            if not self.stochastic_transform and self.transform:
                img = self.transform(img)
            self.img_dict[img_name] = img

        if self.stochastic_transform and self.transform:
            img = self.transform(img)
        return eeg, label, img


from typing import Literal

Options = Literal[
    "eeg_dataset",
    "split_dataset",
    "eeg_image_dataset_64_diffaug_none",
    "eeg_image_dataset_64_diffaug_all",
    "eeg_image_dataset_64_diffaug_none_norm",
]


def loadDatasetPickle(pickleFileName: Options):
    file = os.path.join(dataset_path, "pickle", pickleFileName + ".pickle")
    with open(file, "rb") as fr:
        print(f"loading {pickleFileName} pickle...")
        data = pickle.load(fr)
        # print(data)
        print(f"finished loading {pickleFileName}!")
    return data


def dumpDatasetPickle():
    # eegDataset = EEGDataset()
    # splitDataset = {
    #     split: Splitter(eegDataset, split_name=split)
    #     for split in ["train", "val", "test"]
    # }
    # print(splitDataset["train"])
    # print(splitDataset["val"])
    # print(splitDataset["test"])

    from torchvision.transforms import transforms
    from diff_augment import DiffAugment

    print("dumping pickle...")

    # pickleFileName = "eeg_image_dataset_64_diffaug_all.pickle"
    # pickleFileName = "eeg_image_dataset_64_diffaug_none_norm.pickle"
    pickleFileName = "eeg_image_dataset_128_diffaug_none_norm.pickle"
    config = {
        # "img-size": (3, 64, 64),
        "img-size": (3, 128, 128),
        "diffaug-policy": "color,translation,cutout",
    }
    transform = transforms.Compose(
        [
            transforms.Resize((config["img-size"][1:3])),
            transforms.ToTensor(),
            # DiffAugment(policy="color,translation,cutout"),
            # DiffAugment(policy=config["diffaug-policy"]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    splitDataset = loadDatasetPickle("split_dataset")
    eegImageDataset = {
        split: EEGImageDataset(
            splitDataset[split],
            transform,
            init_imgdict=True,
            stochastic_transform=False,
            # stochastic_transform=True,
        )
        for split in ["train", "val", "test"]
    }

    file = os.path.join(dataset_path, "pickle", pickleFileName)
    with open(file, "wb") as fw:
        pickle.dump(eegImageDataset, fw)
        print("finished dumping!")


if __name__ == "__main__":
    # data = loadDatasetPickle("eeg_image_dataset_64_diffaug_none")
    # print(data["train"].__getitem__(10))
    # print(data["test"].__len__())
    dumpDatasetPickle()
    print("hi :)")
