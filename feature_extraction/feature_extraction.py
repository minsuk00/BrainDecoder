import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-O", "--optim", type=str, default="Adam")

args = parser.parse_args()

print(args)


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
import random

from dataset import EEGDataset, SplitDataset

dataset = EEGDataset(eeg_dataset_file_name="eeg_5_95_std.pth")
loaders = {
    split: DataLoader(
        SplitDataset(dataset, split_name=split),
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )
    for split in ["train", "val", "test"]
}


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# with classifier attached
class FeatureExtractorNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # seed_everything(seed,workers=True)

        self.input_size = 128
        self.hidden_size = 128
        self.lstm_layers = 1
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

    def configure_optimizers(self):
        # return optim.SGD(self.parameters(),lr=1e-4,weight_decay=0.1)
        # return optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.1)
        optimizer = optim.Adam(self.parameters(), lr=(1e-4) * 8, weight_decay=0.005)
        # optimizer = optim.Adam(self.parameters(), lr=(1e-3))
        # optimizer = optim.SGD(self.parameters(), lr=(1e-3), momentum=0.9)
        # optimizer = optim.SGD(self.parameters(), lr=(1e-4) * 5)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.975**epoch
        )
        # self.scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.0001, max_lr=0.01,step_size_up=5,mode="triangular2")
        # self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # self.scheduler = optim.lr_scheduler.CyclicLR(
        #     optimizer, base_lr=1e-6, max_lr=0.01, step_size_up=15, mode="triangular2"
        # )

        return [optimizer], [self.scheduler]
        # return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        print("\n")
        self.training_step_outputs["correct_num"] = 0
        self.training_step_outputs["loss_sum"] = 0

        print("Learning rate:", self.scheduler.get_last_lr(), "\n")

    def validation_step(self, batch, batch_idx):
        x, y = batch
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

    def test_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)
        loss = self.loss_fn(out, y)

        y_hat = torch.argmax(out, dim=1)
        # print("OUT,YHAT:",out,y_hat)
        test_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)

        self.log_dict(
            {"test_loss": loss, "test_acc": test_acc}, prog_bar=True, on_epoch=True
        )
        # print("   ||   test loss:",loss.item(), "   ||   test accuracy:",test_acc )


version_num = 96
epoch = 9
step = 23920
root_path = ""
PATH = os.path.join(
    root_path,
    "lightning_logs",
    "version_" + str(version_num),
    "checkpoints",
    "epoch=" + str(epoch) + "-step=" + str(step) + ".ckpt",
)

model = FeatureExtractorNN()
model.to(device)
# model = FeatureExtractorNN.load_from_checkpoint(PATH)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
logger = TensorBoardLogger(
    "/Users/ms/cs/ML/NeuroImagen/lightning_logs",
    name="Adam_1-e4*8_Lambda_0.975",
    version="weight_decay_0.005",
)

trainer = L.Trainer(max_epochs=200, callbacks=[lr_monitor], logger=logger)
# trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=PATH)
trainer.validate(model, dataloaders=loaders["val"])
trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
# trainer.fit(model,train_dataloaders=train_loader)
# trainer.validate(model, dataloaders=val_loader)


if __name__ == "__main__":
    print("hehe")
