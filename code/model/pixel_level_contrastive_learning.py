import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
from datetime import datetime
from pprint import pprint
import os
import sys
import argparse
import json

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset import EEGDataset, Splitter

root_path = "/home/choi/BrainDecoder/"
dataset_path = os.path.join(root_path, "dataset")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")
eeg_dataset_path = os.path.join(dataset_path, "eeg")

loaders = None
device = None

config = {
    "optimizer": "Adam",  # ("Adam", "AdamW", "SGD")
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "scheduler": "LambdaLR",
    "lambda-factor": 0.99,
    "weight-decay": 0,
    "margin": 1.0,
    "lstm-layer": 2,
    "lstm_hidden_size": 128,
    "batch-size": 16,
    "tsne": False,
    "tsne-interval": 10,
    "ckpt": "None",
    "gpu_id": 1,
    "use_online_hard_triplet": True,
}


class PixelLevelFeatureExtractorNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # Triplet loss
        def dist_fn(x1, x2):
            return torch.sum(torch.pow(torch.subtract(x1, x2), 2), dim=0)

        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=dist_fn, margin=config["margin"]
        )

        # model
        self.input_size = 128
        self.hidden_size = config["lstm_hidden_size"]
        self.lstm_layers = config["lstm-layer"]
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

    def shared_step(self, batch, split_name="train"):
        anchor_eeg, anchor_label, _ = batch

        positive_eeg, positive_label = loaders[split_name].dataset.generate_data_points(
            anchor_label, positive=True
        )
        negative_eeg, negative_label = loaders[split_name].dataset.generate_data_points(
            anchor_label, positive=False
        )

        anchor_feature = self(anchor_eeg)
        positive_feature = self(positive_eeg)
        negative_eeg = self(negative_eeg)

        loss = self.loss_fn(anchor_feature, positive_feature, negative_eeg)

        return loss

    def shared_step_online_hard_triplet_mining(self, batch):
        eegs, labels, _ = batch

        batch_size = labels.size(0)
        # calculate embeddings(feature vector) for each eeg
        embeddings = self(eegs)

        # create pairwise distance matrix
        # https://github.com/eroj333/learning-cv-ml/blob/master/SNN/Online%20Triplet%20Mining.ipynb
        dot_product = torch.matmul(embeddings, embeddings.t())
        diag = torch.diag(dot_product)
        dist_matrix = diag.unsqueeze(dim=1) - 2.0 * dot_product + diag.unsqueeze(dim=0)
        dist_matrix = torch.clamp(dist_matrix, min=0)

        # create pairwise binary adjacency matrix
        eq_mask = torch.eq(labels, labels.unsqueeze(dim=1))

        identity = torch.eye(batch_size)
        distinct_identiy = torch.logical_not(identity).to(device)

        # get positive mask
        anchor_positive_mask = torch.logical_and(eq_mask, distinct_identiy).to(
            dtype=float
        )

        # calculate positive distance matrix
        anchor_positive_dist_matrix = torch.mul(dist_matrix, anchor_positive_mask)
        hardest_anchor_positive_dist = anchor_positive_dist_matrix.max(dim=1)

        # get negative mask
        anchor_negative_mask = torch.logical_not(eq_mask).to(dtype=float)

        # calculate negative distance matrix
        anchor_negative_dist_matrix = torch.mul(dist_matrix, anchor_negative_mask)
        hardest_anchor_negative_dist = anchor_negative_dist_matrix.max(dim=1)

        triplet_loss = torch.clamp(
            hardest_anchor_positive_dist.values
            - hardest_anchor_negative_dist.values
            + config["margin"],
            min=0,
        ).mean()

        # self.log_dict(
        #     {"train_loss": triplet_loss, "lr": self.scheduler.get_last_lr()[0]},
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=config["batch-size"],
        # )
        return triplet_loss

    def training_step(self, batch, _):
        if config["use_online_hard_triplet"]:
            loss = self.shared_step_online_hard_triplet_mining(batch)
        else:
            loss = self.shared_step(batch, "train")

        self.log_dict(
            {
                "train_loss": loss,
                "lr": self.scheduler.get_last_lr()[0],
            },
            on_epoch=True,
            prog_bar=True,
            batch_size=config["batch-size"],
        )
        return loss

    def validation_step(self, batch, _):
        if config["use_online_hard_triplet"]:
            loss = self.shared_step_online_hard_triplet_mining(batch)
        else:
            loss = self.shared_step(batch, "val")

        self.log_dict(
            {"val_loss": loss},
            on_epoch=True,
            prog_bar=True,
            batch_size=config["batch-size"],
        )

    def create_optimizer(self):
        if config["optimizer"] == "Adam":
            return optim.Adam(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight-decay"],
                betas=config["betas"],
            )
        elif config["optimizer"] == "AdamW":
            return optim.AdamW(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight-decay"],
            )
        elif config["optimizer"] == "SGD":
            return optim.SGD(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight-decay"],
            )
        else:
            raise Exception("optimizer config error")

    def create_scheduler(self, optimizer):
        if config["scheduler"] == "LambdaLR":
            return optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: config["lambda-factor"] ** epoch
            )
        else:
            raise Exception("scheduler config error")

    def configure_optimizers(self):
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        self.scheduler = scheduler
        return [optimizer], [scheduler]


def train():
    if config["ckpt"] != "None":
        model = PixelLevelFeatureExtractorNN.load_from_checkpoint(config["ckpt"])
    else:
        model = PixelLevelFeatureExtractorNN()
    model.to(device)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("===========================")
    pprint(config)
    print(now)
    print("===========================")

    logger = TensorBoardLogger(
        save_dir="/home/choi/BrainDecoder/lightning_logs/PixelLevelFeatureExtraction",
        name=f"{now}",
        version="version",
    )

    # Writing to sample.json
    os.makedirs(
        f"/home/choi/BrainDecoder/lightning_logs/PixelLevelFeatureExtraction/{now}/version",
        exist_ok=True,
    )
    # Serializing json
    config_json = json.dumps(config, indent=4)
    with open(
        f"/home/choi/BrainDecoder/lightning_logs/PixelLevelFeatureExtraction/{now}/version/config.json",
        "w+",
    ) as outfile:
        outfile.write(config_json)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="{epoch}_{val_loss:.4f}",
    )
    trainer = L.Trainer(
        max_epochs=1000,
        logger=logger,
        callbacks=[lr_monitor, ckpt_callback],
        accelerator="gpu",
        devices=[config["gpu_id"]],
    )

    # if config["ckpt"] != "None":
    #     trainer.fit(
    #         model,
    #         train_dataloaders=loaders["train"],
    #         val_dataloaders=loaders["val"],
    #         ckpt_path=config["ckpt"],
    #     )
    # else:

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )


def preload():
    global device
    global loaders

    device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    dataset = EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")
    loaders = {
        split: DataLoader(
            dataset=Splitter(dataset, split_name=split),
            batch_size=config["batch-size"],
            drop_last=True,
            shuffle=True if split == "train" else False,
            num_workers=8,
            pin_memory=True,
        )
        for split in ["train", "val", "test"]
    }


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--lr", type=float, default=1e-3)
    # parser.add_argument(
    #     "-O", "--optimizer", choices=("Adam", "AdamW", "SGD"), default="Adam"
    # )
    # parser.add_argument("-S", "--scheduler", choices=("LambdaLR"), default="LambdaLR")
    parser.add_argument("-LF", "--lambdafactor", type=float, default=0.99)
    parser.add_argument("-W", "--weightdecay", type=float, default=0)
    parser.add_argument("-M", "--margin", type=float, default="1.0")
    parser.add_argument("--ckpt", type=str, default="None")
    parser.add_argument("--useofflinetriplet", action="store_true")

    args = parser.parse_args()

    if args.ckpt != "None":
        config["ckpt"] = args.ckpt
    if args.margin != 1.0:
        config["margin"] = args.margin
    if args.lambdafactor != 0.99:
        config["lambda-factor"] = args.lambdafactor
    if args.weightdecay != 0:
        config["weight-decay"] = args.weightdecay
    if args.lr != 1e-3:
        config["lr"] = args.lr
    if args.useofflinetriplet:
        config["use_online_hard_triplet"] = False


if __name__ == "__main__":
    parseArgs()
    preload()
    train()
