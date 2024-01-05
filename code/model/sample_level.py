import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import clip
from lavis.models import load_model_and_preprocess


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import io
from datetime import datetime

import sys
import os

# sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import lookup_dict as LD
# import dataset as D

config = {
    "batch_size": 16,
    "optimizer": "Adam",  # ("Adam", "AdamW", "SGD")
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "scheduler": "LambdaLR",
    "lambda_factor": 0.99,
    "weight_decay": 0,
    "lstm_layer": 3,
    "tsne": True,
    "tsne_interval": 20,
    "use_blip": False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# dataset = D.EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")

# loaders = {
#     split: DataLoader(
#         dataset=D.Splitter(dataset, split_name=split),
#         batch_size=config["batch_size"],
#         drop_last=True,
#         shuffle=True if split == "train" else False,
#         num_workers=23,
#     )
#     for split in ["train", "val", "test"]
# }


class SampleLevelFeatureExtractorNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        # seed_everything(seed,workers=True)

        self.input_size = 128
        self.hidden_size = 128
        self.lstm_layers = config["lstm_layer"]
        self.out_size = 768 * 77

        # self.lstm = nn.LSTM(input_size=128,hidden_size=128,num_layers=128)
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

        # self.loss_fn = nn.CrossEntropyLoss()
        def l2_squared(x1, x2):
            # return torch.sum(torch.pow(torch.subtract(x1, x2), 2), dim=1)
            return torch.mean(torch.pow(torch.subtract(x1, x2), 2))

        self.loss_fn = l2_squared

        self.blip_caption_cache = {}

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        tmp_out = lstm_out[:, -1, :]
        out = self.output(tmp_out)
        out = out.reshape(out.size(0), 77, 768)

        return out

    def training_step(self, batch, batch_idx):
        eegs, labels, img_names = batch

        eegs = eegs.to(device)
        eeg_embeddings = self(eegs)

        labels = self.get_img_caption(labels, img_names, use_BLIP=config["use_blip"])
        labels = clip.tokenize(labels).to(device)

        # with torch.no_grad():
        #     label_features = clip_model.encode_text(labels)

        # loss = self.loss_fn(eeg_embeddings, label_features)
        loss = 0

        self.log_dict(
            {
                "train_loss": loss,
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=config["batch_size"],
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(
            {
                "lr": self.scheduler.get_last_lr()[0],
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=config["batch_size"],
        )
        if config["tsne"]:
            if self.current_epoch % config["tsne_interval"] == 0:
                self.show_manifold()

    def validation_step(self, batch, batch_idx):
        eegs, labels, img_names = batch

        eegs = eegs.to(device)
        eeg_features = self(eegs)

        labels = self.get_img_caption(labels, img_names, use_BLIP=config["use_blip"])
        labels = clip.tokenize(labels).to(device)

        # with torch.no_grad():
        #     label_features = clip_model.encode_text(labels)

        # loss = self.loss_fn(eeg_features, label_features)
        loss = 0

        self.log_dict(
            {"val_loss": loss},
            prog_bar=True,
            on_epoch=True,
            batch_size=config["batch_size"],
        )

    def create_optimizer(self):
        if config["optimizer"] == "Adam":
            return optim.Adam(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                betas=config["betas"],
            )
        elif config["optimizer"] == "AdamW":
            return optim.AdamW(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
        elif config["optimizer"] == "SGD":
            return optim.SGD(
                self.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
        else:
            raise Exception("optimizer config error")

    def create_scheduler(self, optimizer):
        if config["scheduler"] == "LambdaLR":
            return optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: config["lambda_factor"] ** epoch
            )
        else:
            raise Exception("scheduler config error")

    def configure_optimizers(self):
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        self.scheduler = scheduler
        return [optimizer], [scheduler]
        # return [optimizer]


# ckpt = "/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/04:15_Adam_0.001_LambdaLR_weight-decay_0_lambda-factor_0.99/2023-12-21 04:15:42/checkpoints/epoch=405-step=201782.ckpt"
# model = SampleLevelFeatureExtractorNN.load_from_checkpoint(ckpt)
# model.to(device)
