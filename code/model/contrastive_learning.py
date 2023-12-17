import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-L", "--lr", type=float, default=1e-4)
parser.add_argument(
    "-O", "--optimizer", choices=("Adam", "AdamW", "SGD"), default="Adam"
)
parser.add_argument("-S", "--scheduler", choices=("LambdaLR"), default="LambdaLR")
parser.add_argument("-LF", "--lambdafactor", type=float, default=0.95)
parser.add_argument("-W", "--weightdecay", type=float, default=0)
parser.add_argument("-M", "--margin", type=float, default="1.5")

config = parser.parse_args()
print(config)

# import
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import os
import numpy as np
import random
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from dataset import EEGDataset, Splitter


def setup_dataloaders():
    dataset = EEGDataset(eeg_dataset_file_name="eeg_5_95_std.pth")
    loaders = {
        split: DataLoader(
            Splitter(dataset, split_name=split),
            batch_size=16,
            shuffle=True,
            drop_last=True,
        )
        for split in ["train", "val", "test"]
    }
    return loaders


class FeatureExtractor_ContrastiveLearning_NN(L.LightningModule):
    def __init__(self, loaders):
        super().__init__()
        self.save_hyperparameters()

        self.loaders = loaders

        # Triplet loss
        def dist_fn(x1, x2):
            return torch.sum(torch.pow(torch.subtract(x1, x2), 2), dim=0)

        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=dist_fn, margin=config.margin
        )

        # model
        self.input_size = 128
        self.hidden_size = 128
        self.lstm_layers = 1
        self.out_size = 128

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
            nn.ReLU(),
        )

    def forward(self, input):
        input = input.to(self.device)

        lstm_out, _ = self.lstm(input)
        res = self.output(lstm_out[:, -1, :])
        return res

    def training_step(self, batch, batch_idx):
        anchor_eeg, anchor_label = batch

        positive_eeg, positive_label = self.loaders[
            "train"
        ].dataset.generate_data_points(anchor_label, positive=True)
        negative_eeg, negative_label = self.loaders[
            "train"
        ].dataset.generate_data_points(anchor_label, positive=False)

        anchor_feature = self(anchor_eeg)
        positive_feature = self(positive_eeg)
        negative_eeg = self(negative_eeg)

        loss = self.loss_fn(anchor_feature, positive_feature, negative_eeg)

        self.log_dict(
            {"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]},
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        anchor_eeg, anchor_label = batch

        positive_eeg, positive_label = self.loaders["val"].dataset.generate_data_points(
            anchor_label, positive=True
        )
        negative_eeg, negative_label = self.loaders["val"].dataset.generate_data_points(
            anchor_label, positive=False
        )

        anchor_feature = self(anchor_eeg)
        positive_feature = self(positive_eeg)
        negative_eeg = self(negative_eeg)

        loss = self.loss_fn(anchor_feature, positive_feature, negative_eeg)

        self.log_dict(
            {"val_loss": loss},
            on_epoch=True,
            prog_bar=True,
        )

    def create_optimizer(self):
        if config.optimizer == "Adam":
            return optim.Adam(
                self.parameters(), lr=config.lr, weight_decay=config.weightdecay
            )
        elif config.optimizer == "AdamW":
            return optim.AdamW(
                self.parameters(), lr=config.lr, weight_decay=config.weightdecay
            )
        elif config.optimizer == "SGD":
            return optim.SGD(
                self.parameters(), lr=config.lr, weight_decay=config.weightdecay
            )
        else:
            raise Exception("optimizer config error")

    def create_scheduler(self, optimizer):
        if config.scheduler == "LambdaLR":
            return optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: config.lambdafactor**epoch
            )
        else:
            raise Exception("scheduler config error")

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-4 * 5)
        optimizer = self.create_optimizer()
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95**epoch)
        scheduler = self.create_scheduler(optimizer)
        self.scheduler = scheduler
        return [optimizer], [scheduler]


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    loaders = setup_dataloaders()

    # model training
    model = FeatureExtractor_ContrastiveLearning_NN(loaders)
    model.to(device)

    logger = TensorBoardLogger(
        save_dir="/Users/ms/cs/ML/NeuroImagen/lightning_logs/ContrastiveLossFeatureLearning",
        name=f"{config.optimizer}_{config.lr}_{config.scheduler}_margin_{config.margin}",
        version=f"weight-decay_{config.weightdecay}_lambda-factor_{config.lambdafactor}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer = L.Trainer(max_epochs=200, logger=logger, callbacks=[lr_monitor])
    trainer.fit(
        model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"]
    )
