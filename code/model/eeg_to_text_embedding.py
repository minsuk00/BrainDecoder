from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# import clip
# from lavis.models import load_model_and_preprocess
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import io
from datetime import datetime
from pprint import pprint
import argparse
import sys
import os
import json
from time import time

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import dataset as D
from dataset import Splitter, EEGDataset, loadPickle
import lookup_dict as LD

root_path = "/scratch/choi"
dataset_path = os.path.join(root_path, "dataset", "Brain2Image")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")
eeg_dataset_path = os.path.join(dataset_path, "eeg")

blip_model, vis_processors = None, None
loaders = None
device = None

config = {
    "batch_size": 128,
    "optimizer": "Adam",  # ("Adam", "AdamW", "SGD")
    "lr": 1e-4,
    # "lr": 1e-4 * 75,
    "betas": (0.9, 0.999),
    # "scheduler": "LambdaLR",
    "scheduler": "None",
    "lambda_factor": 0.975,
    # "lambda_factor": 1,
    "weight_decay": 0,
    "lstm_layer": 2,
    "lstm_hidden_size": 128,
    "out_size": 768 * 77,
    "use_blip": False,
    "gpu_id": 1,
    "ckpt": "None",
    "loss_fn": "mse",  # "mse" or "cos_sim"
    "pooling_method": "None",  # start, end, mean, None
}


class TextEmbeddingDictionary:
    def __init__(
        self,
        labels,
        model,
        tokenizer,
        blip_processor=None,
        blip_model=None,
        blip_captions=None,
        image_names=None,
        prefix="An image of ",
        postfix="",
        pooling_method: Literal["start", "end", "mean", "None"] = "None",
    ) -> None:
        self.use_blip = False
        if blip_captions:
            self.use_blip = True
            self.labels = list(blip_captions.keys())
            self.labels_processed = list(blip_captions.values())
            self.image_to_caption_dict = blip_captions
        elif blip_model and blip_processor and image_names:
            self.use_blip = True
            self.labels = []
            self.labels_processed = []
            self.image_to_caption_dict = {}
            image_name_to_skip = ["n03452741_17620"]
            for img_name in image_names:
                if img_name in image_name_to_skip:
                    continue
                if img_name in self.image_to_caption_dict:
                    continue
                else:
                    img_path = os.path.join(
                        images_dataset_path, img_name.split("_")[0], img_name + ".JPEG"
                    )
                    print(f"{img_name}:", end=" ")
                    pil_img = Image.open(img_path).convert("RGB")
                    inputs = blip_processor(pil_img, prefix, return_tensors="pt").to(
                        device
                    )
                    out = blip_model.generate(**inputs)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    self.labels.append(img_name)
                    self.labels_processed.append(caption)
                    self.image_to_caption_dict[img_name] = caption
                    print(caption)
            blip_json_path = "./blip_captions.json"
            # Writing JSON data
            with open(blip_json_path, "w") as file:
                json.dump(self.image_to_caption_dict, file, indent=4)
            print(
                f"{len(self.image_to_caption_dict)} caption saved at {blip_json_path}"
            )
        else:
            self.labels = labels
            self.labels_processed = [
                prefix + i.lower().replace("_", " ") + postfix for i in self.labels
            ]
        self.tokens = tokenizer(
            self.labels_processed,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        self.model = model
        with torch.no_grad():
            text_embedding = []
            batch_size = 128
            for i in range(self.tokens["input_ids"].shape[0] // batch_size + 1):
                print(
                    f"Extracting text embedding for  {batch_size*i}~{batch_size*(i+1)}"
                )
                output = self.model(
                    input_ids=self.tokens["input_ids"][
                        i * batch_size : (i + 1) * batch_size
                    ].to(device)
                )
                text_embedding.append(output.last_hidden_state)
            text_embedding = torch.vstack(text_embedding)
            print("text_embedding.shape", text_embedding.shape)
        # text_embedding = model_outputs.last_hidden_state

        if pooling_method == "mean":
            text_embedding = self.mean_pooling(
                text_embedding, self.tokens["attention_mask"]
            )
        elif pooling_method == "start":
            text_embedding = text_embedding[torch.arange(text_embedding.shape[0]), 0]
        elif pooling_method == "end":
            text_embedding = text_embedding[
                torch.arange(text_embedding.shape[0]),
                self.tokens["input_ids"].argmax(dim=-1),
            ]
        self.text_embedding = text_embedding
        self.embedding_dict = dict(zip(self.labels, self.text_embedding))

    def get(self, labels, image_names):
        if image_names and self.use_blip:
            # labels = []
            # for image_name in image_names:
            #     if image_name in self.image_to_caption_dict:
            #         labels.append(self.image_to_caption_dict[image_name])
            #     else:
            #         print(f"{image_names} caption does not exist")
            #         raise NotImplementedError
            labels = image_names
        else:
            labels = np.array(labels.cpu())
            labels = LD.batch_idx_to_id(labels)
            labels = LD.batch_id_to_name(labels)
        returning_embedding = []
        for l in labels:
            if l in self.embedding_dict:
                returning_embedding.append(self.embedding_dict[l])
            else:
                print(f"{l} label does not exist")
                raise NotImplementedError  # @TODO process label and add to dict
        return torch.stack(returning_embedding).to(device)

    @staticmethod
    def mean_pooling(text_embedding, attention_mask):
        attention_mask = attention_mask.detach().clone()
        first_zero_indices = (attention_mask == 0).long().argmax(dim=1) - 1
        for i, index in enumerate(first_zero_indices):
            attention_mask[i, index] = 0
        attention_mask[:, 0] = 0
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(text_embedding.size())
            .float()
            .to(device)
        )
        sum_embeddings = torch.sum(text_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class EEGtoTextEmbedding(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

        # self.text_embeddings = text_embeddings

        self.input_size = 128
        self.hidden_size = config["lstm_hidden_size"]
        self.lstm_layers = config["lstm_layer"]
        self.use_pooling = False if config["pooling_method"] == "None" else True
        self.out_size = 768 if self.use_pooling else 768 * 77
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        if not self.use_pooling:
            self.output = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=self.out_size),
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
            )

        self.loss_fn = self.get_loss_function()
        self.cos = nn.CosineSimilarity()
        self.blip_caption_cache = {}

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        tmp_out = lstm_out[:, -1, :]
        out = self.output(tmp_out)
        if not self.use_pooling:
            out = out.reshape(out.size(0), 77, 768)
        return out

    def training_step(self, batch, _):
        eegs, labels, image_names = batch
        # Process EEG data
        eegs = eegs.to(device)
        eeg_embeddings = self(eegs)
        # Get Text embedding
        text_embeddings = text_embeddings_dict.get(labels, image_names)
        # Calculate loss and log
        loss = self.loss_fn(eeg_embeddings, text_embeddings)
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
                "lr": (
                    self.scheduler.get_last_lr()[0] if self.scheduler else config["lr"]
                ),
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=config["batch_size"],
        )
        # if config["tsne"]:
        #     if self.current_epoch % config["tsne_interval"] == 0:
        #         self.show_manifold(loaders["val"])

    def validation_step(self, batch, _):
        eegs, labels, image_names = batch

        # Process EEG data
        eegs = eegs.to(device)
        eeg_embeddings = self(eegs)
        # Get Text embedding
        text_embeddings = text_embeddings_dict.get(labels, image_names)
        # Calculate loss and log
        loss = self.loss_fn(eeg_embeddings, text_embeddings)
        cos_sim = self.cos(eeg_embeddings, text_embeddings).mean()
        self.log_dict(
            {"val_loss": loss, "cos_sim": cos_sim},
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
        elif config["scheduler"] == "None":
            return None
        else:
            raise Exception("scheduler config error")

    def configure_optimizers(self):
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        self.scheduler = scheduler
        return {"optimizer": optimizer, "lr_schedulers": scheduler}
        # return [optimizer], [scheduler]

    def get_loss_function(self):
        if config["loss_fn"] == "mse":
            return nn.MSELoss()
        elif config["loss_fn"] == "cos_sim":
            return nn.CosineEmbeddingLoss()
            # return nn.CosineSimilarity().mean()
        else:
            raise Exception("Invalid loss function")


def train():
    global device

    device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    clip_version = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    clip_transformer = CLIPTextModel.from_pretrained(clip_version).to(device)
    blip_version = "Salesforce/blip-image-captioning-large"
    blip_processor = BlipProcessor.from_pretrained(blip_version)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_version).to(device)
    labels = LD.id_to_name.values()

    global loaders
    begin = time()
    dataset = EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")
    print(f"EEG Dataset created: {time()-begin} sec")
    begin = time()
    # loaders = {
    #     split: DataLoader(
    #         dataset=D.Splitter(dataset, split_name=split),
    #         batch_size=config["batch_size"],
    #         drop_last=True,
    #         shuffle=True if split == "train" else False,
    #         num_workers=8,
    #         pin_memory=True,
    #     )
    #     for split in ["train", "val", "test"]
    # }

    loaders = loadPickle("split_dataloader_128")
    print(f"Dataloader created: {time()-begin} sec")

    blip_captions = None
    with open("../blip_captions.json", "r") as file:
        blip_captions = json.load(file)
    global text_embeddings_dict
    text_embeddings_dict = TextEmbeddingDictionary(
        labels=labels,
        tokenizer=clip_tokenizer,
        model=clip_transformer,
        blip_processor=blip_processor,
        blip_model=blip_model,
        blip_captions=blip_captions,
        image_names=dataset.images,
        prefix="An image of ",  # "An image of ",
        postfix="",
        pooling_method=config["pooling_method"],  # start, end, mean, None
    )

    model = EEGtoTextEmbedding()
    model.to(device)
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    print("===========================")
    pprint(config)
    print(now)
    print("===========================")

    logger = TensorBoardLogger(
        save_dir="../../lightning_logs/SampleLevelFeatureExtraction",
        name=f"{now}",
        version="version",
    )

    # Writing to sample.json
    os.makedirs(
        f"/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/{now}/version",
        exist_ok=True,
    )
    # Serializing json
    config_json = json.dumps(config, indent=4)
    with open(
        f"/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/{now}/version/config.json",
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
        max_epochs=2000,
        logger=logger,
        callbacks=[lr_monitor, ckpt_callback],
        accelerator="gpu",
        devices=[config["gpu_id"]],
        # devices=2,
        # strategy="ddp",
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


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-C",
        "--ckpt",
        type=str,
        default="None",
    )

    args = parser.parse_args()

    if args.ckpt != "None":
        config["ckpt"] = args.ckpt


if __name__ == "__main__":
    parseArgs()
    train()
