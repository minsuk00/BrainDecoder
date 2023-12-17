# Parse Argument
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument("-O", "--optim", type=str, default="Adam")
parser.add_argument("-B", "--batch", type=int, default=16)
parser.add_argument("-O", "--optim", type=str, default="Adam")
parser.add_argument("-L", "--lr", type=float, default=1e-3)
parser.add_argument("-W", "--weight-decay", type=float, default=0.001)
parser.add_argument("-LF", "--lambda-factor", type=float, default=0.95)
parser.add_argument("-LL", "--lstm-layer", type=int, default=3)


config = {
    "batch-size": 16,
    "optimizer": "Adam",  # ("Adam", "AdamW", "SGD")
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "scheduler": "LambdaLR",
    "lambda-factor": 0.95,
    "weight-decay": 0.001,
    "lstm-layer": 3,
}

########################################################################

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import os
import numpy as np
from datetime import datetime
from pytz import timezone
import sys
from pprint import pprint

# = "/home/choi/BrainDecoder/code"
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
import dataset as D

########################################################################

# Set Dataloaders
dataset = D.EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")
loaders = {
    split: DataLoader(
        D.Splitter(dataset, split_name=split),
        batch_size=config["batch-size"],
        shuffle=True if split == "train" else False,
        num_workers=23,
        drop_last=True,
    )
    for split in ["train", "val", "test"]
}

# Set Device
gpu_id = 2
device = f"cuda:{gpu_id}" if torch.cuda.is_available else "cpu"

########################################################################


# Model
class FeatureExtractorNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # seed_everything(seed,workers=True)

        self.input_size = 128
        self.hidden_size = 128
        self.lstm_layers = config["lstm-layer"]
        self.out_size = 128

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
        self.classifer = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=40),
            # don't use softmax with cross entropy loss
            # nn.Softmax(dim=1)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.NLLLoss()
        self.training_step_outputs = {"correct_num": 0, "loss_sum": 0}
        self.validation_step_outputs = {"correct_num": 0, "loss_sum": 0}

    def forward(self, input):
        batch_size = input.size(0)
        lstm_init = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_size),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_size),
        )
        lstm_init = (lstm_init[0].to(device), lstm_init[0].to(device))

        # dont need to transpose because already transposed when creating dataset
        # input = input.transpose(1,2)

        lstm_out, _ = self.lstm(input, lstm_init)
        # tmp_out = lstm_out[:,-1,:] if input.dim()==3 else lstm_out[-1,:]
        tmp_out = lstm_out[:, -1, :]
        out = self.output(tmp_out)
        # print("out shape",out.shape)
        res = self.classifer(out)

        return res

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        out = self(x)
        loss = self.loss_fn(out, y)

        self.log_dict({"train_loss": loss}, prog_bar=True, on_epoch=True)
        preds = out.argmax(dim=1)
        self.training_step_outputs["correct_num"] += (preds == y).sum()
        self.training_step_outputs["loss_sum"] += loss
        return loss

    def on_train_epoch_end(self) -> None:
        num_correct = self.training_step_outputs["correct_num"]
        acc = num_correct / loaders["train"].dataset.__len__()
        loss = self.training_step_outputs["loss_sum"] / loaders["train"].__len__()
        print("\n")
        # print("EPOCH:",self.current_epoch)
        print(
            f"Training accuracy: {acc.item()} ({num_correct.item()}/{loaders['train'].dataset.__len__()} correct)"
        )
        print("Training loss (average):", loss.item())
        # print("\n")
        self.training_step_outputs["correct_num"] = 0
        self.training_step_outputs["loss_sum"] = 0

        print("Learning rate:", self.scheduler.get_last_lr(), "\n")

        self.log_dict({"train_acc_epoch": acc.item()})

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        out = self(x)
        loss = self.loss_fn(out, y)

        self.log_dict({"val_loss": loss}, prog_bar=True, on_epoch=True)
        preds = out.argmax(dim=1)
        self.validation_step_outputs["correct_num"] += (preds == y).sum()
        self.validation_step_outputs["loss_sum"] += loss
        # return loss

    def on_validation_epoch_end(self) -> None:
        num_correct = self.validation_step_outputs["correct_num"]
        acc = num_correct / loaders["val"].dataset.__len__()
        loss = self.validation_step_outputs["loss_sum"] / loaders["val"].__len__()
        print("\n")
        # print("EPOCH:",self.current_epoch)
        print(
            f"Validation accuracy: {acc.item()} ({num_correct.item()}/{loaders['val'].dataset.__len__()} correct)"
        )
        print("Validation loss (average):", loss.item())
        print("\n")
        self.validation_step_outputs["correct_num"] = 0
        self.validation_step_outputs["loss_sum"] = 0
        self.log_dict({"val_acc_epoch": acc.item()})

    def test_step(self, batch, batch_idx):
        x, y, _ = batch

        out = self(x)
        loss = self.loss_fn(out, y)

        y_hat = torch.argmax(out, dim=1)
        # print("OUT,YHAT:",out,y_hat)
        test_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)

        self.log_dict(
            {"test_loss": loss, "test_acc": test_acc}, prog_bar=True, on_epoch=True
        )
        # print("   ||   test loss:",loss.item(), "   ||   test accuracy:",test_acc )

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
        # return [optimizer]


########################################################################


# Train
def train():
    model = FeatureExtractorNN()
    # model = FeatureExtractorNN.load_from_checkpoint(PATH)
    model.to(device)

    now = datetime.now(tz=timezone("Asia/Tokyo"))
    now_time = now.strftime("%H:%M")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(
        "/home/choi/BrainDecoder/lightning_logs/FeatureExtractionClassification",
        name=f"{now_time}_{config['optimizer']}_{config['lr']}_{config['scheduler']}_weight-decay_{config['weight-decay']}_lambda-factor_{config['lambda-factor']}",
        version=now.strftime("%Y-%m-%d %H:%M:%S"),
    )

    trainer = L.Trainer(
        max_epochs=500,
        callbacks=[lr_monitor],
        logger=logger,
        accelerator="gpu",
        devices=[gpu_id],
    )
    trainer.fit(
        model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"]
    )


def update_config():
    args = parser.parse_args()

    for key, value in vars(args).items():
        config[key] = value
    # print(args)


if __name__ == "__main__":
    update_config()
    pprint(config)
    train()
