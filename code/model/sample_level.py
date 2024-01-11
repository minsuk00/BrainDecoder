import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# import clip
from lavis.models import load_model_and_preprocess
from transformers import CLIPTokenizer, CLIPTextModel

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

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import lookup_dict as LD
import dataset as D

root_path = "/home/choi/BrainDecoder/"
dataset_path = os.path.join(root_path, "dataset")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")
eeg_dataset_path = os.path.join(dataset_path, "eeg")

tokenizer, transformer = None, None
blip_model, vis_processors = None, None
loaders = None
device = None

config = {
    "batch_size": 16,
    "optimizer": "Adam",  # ("Adam", "AdamW", "SGD")
    "lr": 1e-3,
    # "lr": 1e-4 * 75,
    "betas": (0.9, 0.999),
    "scheduler": "LambdaLR",
    "lambda_factor": 0.975,
    # "lambda_factor": 1,
    "weight_decay": 0,
    "lstm_layer": 2,
    "lstm_hidden_size": 128,
    "tsne": False,
    "tsne_interval": 20,
    "use_blip": False,
    # "mlp": True,
    # "mlp_layers_1": 128,
    # "mlp_layers_2": 256,
    # "mlp_layers_3": 512,
    # "mlp_layers_4": 512,
    # "mlp_layers_5": 512,
    "gpu_id": 2,
    "ckpt": "None",
    "loss_fn": "mse",  # "mse" or "cos_sim"
}


class SampleLevelFeatureExtractorNN(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        # seed_everything(seed,workers=True)

        self.input_size = 128
        # self.hidden_size = 128
        self.hidden_size = config["lstm_hidden_size"]
        self.lstm_layers = config["lstm_layer"]
        # self.out_size = 768 * 77
        self.out_size = 768

        # self.lstm = nn.LSTM(input_size=128,hidden_size=128,num_layers=128)
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        self.output = nn.Sequential(
            # nn.Linear(
            #     in_features=self.hidden_size, out_features=config["mlp_layers_1"]
            # ),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features=config["mlp_layers_1"], out_features=config["mlp_layers_2"]
            # ),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features=config["mlp_layers_2"], out_features=config["mlp_layers_3"]
            # ),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features=config["mlp_layers_3"], out_features=config["mlp_layers_4"]
            # ),
            # nn.ReLU(),
            # nn.Linear(
            #     in_features=config["mlp_layers_4"], out_features=config["mlp_layers_5"]
            # ),
            # nn.ReLU(),
            # nn.Linear(in_features=config["mlp_layers_5"], out_features=self.out_size),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
            nn.ReLU(),
        )

        # self.loss_fn = nn.CrossEntropyLoss()
        def l2_squared(x1, x2):
            # return torch.sum(torch.pow(torch.subtract(x1, x2), 2), dim=1)
            return torch.mean(torch.pow(torch.subtract(x1, x2), 2))

        # self.loss_fn = l2_squared
        self.loss_fn = self.get_loss_function()
        # self.loss_fn = nn.MSELoss()
        self.cos = nn.CosineSimilarity()

        self.blip_caption_cache = {}

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        tmp_out = lstm_out[:, -1, :]
        out = self.output(tmp_out)
        # out = out.reshape(out.size(0), 77, 768)
        # out = out.reshape(out.size(0), 1, 768)

        return out

    def training_step(self, batch, _):
        eegs, labels, img_names = batch

        eegs = eegs.to(device)
        eeg_embeddings = self(eegs)

        # get label clip embeddings
        labels = self.get_img_caption(labels, img_names, use_BLIP=config["use_blip"])
        batch_encoding = tokenizer(
            labels,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(device)
        outputs = transformer(input_ids=tokens)

        label_features = outputs.last_hidden_state
        label_features = label_features[
            torch.arange(label_features.shape[0]), tokens.argmax(dim=-1)
        ]

        # with torch.no_grad():
        #     label_features = clip_model.encode_text(labels)

        loss = self.loss_fn(eeg_embeddings, label_features)

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
                self.show_manifold(loaders["val"])

    def validation_step(self, batch, batch_idx):
        eegs, labels, img_names = batch

        eegs = eegs.to(device)
        eeg_features = self(eegs)

        labels = self.get_img_caption(labels, img_names, use_BLIP=config["use_blip"])
        # labels = clip.tokenize(labels).to(device)
        batch_encoding = tokenizer(
            labels,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(device)
        outputs = transformer(input_ids=tokens)

        label_features = outputs.last_hidden_state
        label_features = label_features[
            torch.arange(label_features.shape[0]), tokens.argmax(dim=-1)
        ]

        # with torch.no_grad():
        #     label_features = clip_model.encode_text(labels)

        loss = self.loss_fn(eeg_features, label_features)
        cos_sim = self.cos(eeg_features, label_features).mean()

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
        else:
            raise Exception("scheduler config error")

    def configure_optimizers(self):
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        self.scheduler = scheduler
        return [optimizer], [scheduler]
        # return [optimizer]

    def show_manifold(self, dataloader):
        features = []
        actuals = []

        # calculate feature vectors
        with torch.no_grad():
            for data in dataloader:
                eegs, labels, _ = data
                batch_size = eegs.size(0)
                eegs = eegs.to(device)

                actuals += labels.cpu().numpy().tolist()
                features += self(eegs).reshape(batch_size, -1).cpu().numpy().tolist()

        # tsne
        tsne = TSNE(n_components=2, random_state=0)
        cluster = np.array(tsne.fit_transform(np.array(features)))
        actuals = np.array(actuals)

        # make matplotlib figure
        plt.figure(figsize=(16, 10))
        for i in range(40):
            idx = np.where(actuals == i)
            plt.scatter(
                cluster[idx, 0],
                cluster[idx, 1],
                marker=".",
                label=LD.id_to_name[LD.idx_to_id[i]],
            )
        # plt.legend(bbox_to_anchor=(1.25, 0.6), loc="center left")
        plt.legend()

        # convert fig to tensor in order to log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        img = ToTensor()(Image.open(buf))

        # log to tensorboard
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        tb_logger.add_image(
            "t-SNE manifold of sample level feature extraction",
            img,
            self.current_epoch,
        )

        return

    def get_img_caption(self, labels, img_names, use_BLIP=False):
        if use_BLIP:
            # raw algorithm. no caching
            # processed_imgs = []
            # for img_name in img_names:
            #     img_path = os.path.join(
            #         images_dataset_path, img_name.split("_")[0], img_name + ".JPEG"
            #     )
            #     pil_img = Image.open(img_path).convert("RGB")
            #     processed_img = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
            #     processed_imgs.append(processed_img)
            # processed_imgs = torch.cat(processed_imgs)
            # captions = blip_model.generate({"image": processed_imgs})

            # caching
            captions = []
            for img_name in img_names:
                if img_name in self.blip_caption_cache:
                    captions.append(self.blip_caption_cache[img_name])
                else:
                    img_path = os.path.join(
                        images_dataset_path, img_name.split("_")[0], img_name + ".JPEG"
                    )
                    pil_img = Image.open(img_path).convert("RGB")
                    processed_img = (
                        vis_processors["eval"](pil_img).unsqueeze(0).to(device)
                    )
                    caption = blip_model.generate({"image": processed_img})[0]
                    captions.append(caption)
                    self.blip_caption_cache[img_name] = caption
        else:
            prefix = "An image of "
            labels = np.array(labels.cpu())
            labels = LD.batch_idx_to_id(labels)
            labels = LD.batch_id_to_name(labels)
            captions = [prefix + label.replace("_", " ") for label in labels]

        return captions

    def get_loss_function(self):
        if config["loss_fn"] == "mse":
            return nn.MSELoss()
        elif config["loss_fn"] == "cos_sim":
            return nn.CosineEmbeddingLoss()
            # return nn.CosineSimilarity().mean()
        else:
            raise Exception("Invalid loss function")


def preload():
    global tokenizer
    global transformer
    global device

    device = f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    version = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version).to(device)

    if config["use_blip"]:
        global blip_model
        global vis_processors

        blip_model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=device,
        )

    global loaders
    dataset = D.EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")

    loaders = {
        split: DataLoader(
            dataset=D.Splitter(dataset, split_name=split),
            batch_size=config["batch_size"],
            drop_last=True,
            shuffle=True if split == "train" else False,
            num_workers=8,
            pin_memory=True,
        )
        for split in ["train", "val", "test"]
    }


def train():
    if config["ckpt"] != "None":
        model = SampleLevelFeatureExtractorNN.load_from_checkpoint(config["ckpt"])
    else:
        model = SampleLevelFeatureExtractorNN()
    model.to(device)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("===========================")
    pprint(config)
    print(now)
    print("===========================")

    logger = TensorBoardLogger(
        save_dir="/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction",
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
    preload()
    train()
