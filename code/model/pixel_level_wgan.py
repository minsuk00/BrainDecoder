import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from torchvision.utils import make_grid

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import sys
import os
import numpy as np
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import json
from pprint import pprint
import inspect

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import dataset as D
from dataset import loadDatasetPickle, EEGImageDataset, Splitter, EEGDataset
from diff_augment import DiffAugment
import lookup_dict as LD

device, loaders, now = None, None, None

root_path = "/home/choi/BrainDecoder/"
dataset_path = os.path.join(root_path, "dataset")
images_dataset_path = os.path.join(dataset_path, "imageNet_images")

config = {
    "batch-size": 512,
    # "lr": 2e-4,
    "lr": 2e-4,
    # "lambda-factor": 0.99,
    "lambda-factor": 1,
    "gpu-id": 0,
    "lstm-hidden-size": 128,
    "lstm-layer": 2,
    "img-size": (3, 64, 64),
    "use-diffaug": False,
    # "diffaug-policy": "color,translation,cutout",
    "generate-image-every-n-epoch": 20,
    "save-checkpoint-every-n-epoch": 50,
    "feature-extractor-ckpt": "/home/choi/BrainDecoder/lightning_logs/PixelLevelFeatureExtraction/2024-02-06 04:01:23/epoch=466_val_loss=0.0642.ckpt",
    # "feature-extractor-ckpt": "/home/choi/BrainDecoder/lightning_logs/PixelLevelFeatureExtraction/2024-02-06 04:01:23/version/checkpoints/last.ckpt",
    "ckpt": "None",
}


class PixelLevelFeatureExtractorNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # Triplet loss
        # def dist_fn(x1, x2):
        #     return torch.sum(torch.pow(torch.subtract(x1, x2), 2), dim=0)

        # self.loss_fn = nn.TripletMarginWithDistanceLoss(
        # distance_function=dist_fn, margin=config["margin"]
        # )

        # model
        self.input_size = 128
        self.hidden_size = config["lstm-hidden-size"]
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
        # with torch.no_grad():
        input = input.to(self.device)

        lstm_out, _ = self.lstm(input)
        res = self.output(lstm_out[:, -1, :])

        return res


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        # def block(input_features, output_features, normalize=True):
        #     layers = [nn.Linear(input_features, output_features)]
        #     if normalize:
        #         layers.append(nn.BatchNorm1d(output_features, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        # self.model = nn.Sequential(
        #     # *block(latent_dim + 128, 128, normalize=False),
        #     *block(latent_dim, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(config["img-size"]))),
        #     nn.Tanh(),
        # )

        self.linear = nn.Sequential(nn.Linear(228, 1 * 1 * 100), nn.LeakyReLU())

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

        # self.seq = nn.Sequential(
        #     # nn.Linear(100 + 128, 128),
        #     nn.Linear(100, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(1024, int(np.prod(config["img-size"]))),
        #     nn.Tanh(),
        # )

        # def block(in_feat, out_feat, normalize=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     if normalize:
        #         layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        # self.model = nn.Sequential(
        #     *block(100 + 128, 128, normalize=False),
        #     # *block(100, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(config["img-size"]))),
        #     nn.Tanh(),
        # )

    def forward(self, noise, condition):
        # 과연...?
        # print("condition", condition)
        # print("noise", noise)
        # gen_input = torch.cat((condition, noise), -1)
        # # img = self.model(gen_input)
        # img = self.model(noise)
        # img = img.view(img.size(0), *config["img-size"])

        # print("noise shape: ", noise.shape)
        # print("condition shape: ", condition.shape)
        # out = self.main(noise)

        # print("noise shape: ", noise.shape)
        # print("cond shape: ", condition.shape)
        gen_input = torch.cat((condition, noise), -1)
        gen_input = self.linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), 100, 1, 1)
        # print("gen_input shape: ", gen_input.shape)
        # result = self.model(gen_input)
        result = self.main(gen_input)
        # result = self.model(noise)
        # result = result.view(result.size(0), *config["img-size"])
        # print("result shape", result.shape)
        return result
        # return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = nn.Sequential(
        #     # nn.Linear(int(np.prod(config["img-size"])) + 128, 512),
        #     nn.Linear(int(np.prod(config["img-size"])), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid(),
        # )

        # https://github.com/sobhanshukueian/Conditional-DCGAN/blob/main/cDCGAN_(Conditional_DCGAN).ipynb
        self.linear = nn.Sequential(nn.Linear(128, 1 * 64 * 64), nn.LeakyReLU())

        self.main = nn.Sequential(
            # nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

        # self.seq = nn.Sequential(
        #     # nn.Linear(784 + 128, 512),
        #     # nn.Linear(int(np.prod(config["img-size"])), 512),
        #     nn.Linear(int(np.prod(config["img-size"])) + 128, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 1),
        #     # nn.Sigmoid(),
        # )

    def forward(self, img, condition):
        # print(img.shape)  # torch.Size([16, 3, 128, 128])
        # print(condition.shape)  # torch.Size([16, 128])
        # d_input = torch.cat((img.view(img.size(0), -1), condition), -1)

        # d_input = img.view(img.size(0), -1)
        # validity = self.model(d_input)
        # return validity
        condition = self.linear(condition).view(condition.size(0), 1, 64, 64)
        d_input = torch.concat([img, condition], dim=1)
        # d_input = torch.cat((img.view(img.size(0), -1), condition), -1)
        out = self.main(d_input)
        out = torch.flatten(out)
        return out

        # d_input = torch.cat((img.view(img.size(0), -1), condition), -1)
        # input = img.view(img.size(0), -1)
        # result = self.seq(d_input)
        # return result


class saliency_map_GAN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # self.data_shape = (3, 32, 32)

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.feature_extractor = PixelLevelFeatureExtractorNN.load_from_checkpoint(
            config["feature-extractor-ckpt"]
        )
        self.feature_extractor.requires_grad_(False)
        # self.loss_fn = self.adversarial_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
        # return F.mse_loss(y_hat, y)

    def forward(self, noise, condition):
        return self.generator(noise, condition)

    def calculate_gradient_penalty(
        self, model, condition, real_images, fake_images, device
    ):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
        # Get random interpolation between real and fake data
        interpolates = (
            alpha * real_images + ((1 - alpha) * fake_images)
        ).requires_grad_(True)

        model_interpolates = model(interpolates, condition)
        grad_outputs = torch.ones(
            model_interpolates.size(), device=device, requires_grad=False
        )

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def training_step(self, batch, _):
        eegs, _, real_imgs = batch
        g_optim, d_optim = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        batch_size = real_imgs.size(0)
        noise = torch.randn(batch_size, 100)
        # noise = torch.randn(batch_size, 100, 1, 1)
        noise = noise.to(device)
        real_imgs = real_imgs.to(device)

        eeg_features = self.feature_extractor(eegs)
        # print(eeg_features.shape)
        # print(eeg_features)

        y_real = torch.ones([batch_size, 1], device=device, requires_grad=False)
        y_fake = torch.zeros([batch_size, 1], device=device, requires_grad=False)

        #####################
        # generator training
        #####################
        gen_imgs = self.generator(noise, eeg_features)
        y_hat = self.discriminator(gen_imgs, eeg_features)
        # g_loss = self.loss_fn(y_hat, y_real)
        g_loss = -torch.mean(y_hat)

        # print("########")
        # print("real_imgs: ", real_imgs)
        # print("eeg features: ", eeg_features)
        # print("gen_imgs: ", gen_imgs)
        # generator hinge loss from Geometric GAN
        # https://github.com/ChristophReich1996/Mode_Collapse/blob/master/loss.py
        # g_loss = -y_hat.mean()

        # mode seeking loss
        # lz = torch.mean(torch.abs(self.fake_image2 - self.fake_image1)) / torch.mean(
        #     torch.abs(self.z_random2 - self.z_random)
        # )
        # eps = 1 * 1e-5
        # loss_lz = 1 / (lz + eps)

        g_optim.zero_grad()
        self.manual_backward(g_loss)
        g_optim.step()
        g_sch.step()

        #########################
        # discriminator training
        #########################
        y_hat = self.discriminator(real_imgs, eeg_features)
        # d_loss_real = self.loss_fn(y_hat, y_real)
        d_loss_real = torch.mean(y_hat)

        gen_imgs = self.generator(noise, eeg_features)
        y_hat_2 = self.discriminator(gen_imgs, eeg_features)
        # d_loss_fake = self.loss_fn(y_hat_2, y_fake)
        d_loss_fake = torch.mean(y_hat_2)

        # d_loss = (d_loss_real + d_loss_fake) / 2
        gradient_penalty = self.calculate_gradient_penalty(
            self.discriminator,
            eeg_features,
            real_imgs,
            gen_imgs,
            self.device,
        )
        d_loss = -d_loss_real + d_loss_fake + gradient_penalty * 10

        # hinge loss from Geometric GAN
        # https://github.com/ChristophReich1996/Mode_Collapse/blob/master/loss.py
        # d_loss = (
        #     -torch.minimum(
        #         torch.tensor(0.0, dtype=torch.float, device=y_hat.device),
        #         y_hat - 1.0,
        #     ).mean()
        #     - torch.minimum(
        #         torch.tensor(0.0, dtype=torch.float, device=y_hat_2.device),
        #         -y_hat_2 - 1.0,
        #     ).mean()
        # )

        d_optim.zero_grad()
        self.manual_backward(d_loss)
        d_optim.step()
        d_sch.step()

        self.log_dict(
            {
                "g_loss": g_loss,
                "d_loss": d_loss,
                "lr": self.scheduler.get_last_lr()[0],
            },
            prog_bar=True,
            on_epoch=True,
        )

    # def on_train_epoch_end(self) -> None:
    #     loaders["test"].dataset.__getitem__()
    #     return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        if batch_idx > 0:
            return
        # save images
        eegs, _, real_imgs = batch

        batch_size = real_imgs.size(0)
        noise = torch.randn(batch_size, 100)
        # noise = torch.randn(batch_size, 100, 1, 1)
        noise = noise.to(device)
        real_imgs = real_imgs.to(device)

        eeg_features = self.feature_extractor(eegs)
        gen_imgs = self.generator(noise, eeg_features)

        outputs = [real_imgs, gen_imgs]

        save_dir = f"/home/choi/BrainDecoder/lightning_logs/SaliencyMapGAN/{now}/version/output_imgs"
        os.makedirs(save_dir, exist_ok=True)
        grid_count = len(os.listdir(save_dir))

        grid = torch.stack(outputs, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=batch_size, normalize=True)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        img.save(
            os.path.join(
                save_dir, f"grid_{grid_count:04}-epoch_{self.current_epoch}.png"
            )
        )

    # def on_validation_epoch_end(self):
    #     print("HI")

    def test(self, eeg):
        noise = torch.randn(1, 100)
        eeg = eeg.unsqueeze(dim=0)
        condition = self.feature_extractor(eeg)

        noise = noise.to(device)
        condition = condition.to(device)

        return self.forward(noise, condition)

    def configure_optimizers(self):
        g_optim = optim.Adam(
            # self.generator.parameters(), lr=config["lr"], betas=(0.9, 0.999)
            self.generator.parameters(),
            lr=config["lr"],
            betas=(0.5, 0.999),
        )
        # d_optim = optim.Adam(
        #     self.discriminator.parameters(), lr=config["lr"], betas=(0.9, 0.999)
        # )
        d_optim = optim.SGD(
            self.discriminator.parameters(),
            lr=config["lr"],
        )
        g_scheduler = optim.lr_scheduler.LambdaLR(
            g_optim, lambda epoch: config["lambda-factor"] ** epoch
        )
        d_scheduler = optim.lr_scheduler.LambdaLR(
            d_optim, lambda epoch: config["lambda-factor"] ** epoch
        )
        self.scheduler = d_scheduler
        return [g_optim, d_optim], [g_scheduler, d_scheduler]
        # return [g_optim, d_optim], []


def train(now):
    if config["ckpt"] != "None":
        model = saliency_map_GAN.load_from_checkpoint(config["ckpt"])
    else:
        model = saliency_map_GAN()
    model.to(device)
    print("===========================")
    pprint(config)
    print(now)
    print("===========================")

    logger = TensorBoardLogger(
        save_dir="/home/choi/BrainDecoder/lightning_logs/SaliencyMapGAN/",
        name=f"{now}",
        version="version",
    )

    # Writing to sample.json
    os.makedirs(
        f"/home/choi/BrainDecoder/lightning_logs/SaliencyMapGAN/{now}/version",
        exist_ok=True,
    )
    config_json = json.dumps(config, indent=4)
    with open(
        f"/home/choi/BrainDecoder/lightning_logs/SaliencyMapGAN/{now}/version/config.json",
        "w+",
    ) as outfile:
        outfile.write(config_json)
    # Saving class source code
    classSourceCode = inspect.getsource(PixelLevelFeatureExtractorNN)
    classSourceCode2 = inspect.getsource(Generator)
    classSourceCode3 = inspect.getsource(Discriminator)
    classSourceCode4 = inspect.getsource(saliency_map_GAN)
    sourceCode = (
        classSourceCode
        + "\n\n"
        + classSourceCode2
        + "\n\n"
        + classSourceCode3
        + "\n\n"
        + classSourceCode4
    )
    with open(
        f"/home/choi/BrainDecoder/lightning_logs/SaliencyMapGAN/{now}/version/class.py",
        "w+",
    ) as outfile:
        outfile.write(sourceCode)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = ModelCheckpoint(
        save_top_k=-1,
        # monitor="g_loss",
        # mode="min",
        every_n_epochs=config["save-checkpoint-every-n-epoch"],
        save_last=True,
        filename="{epoch}_{g_loss:.4f}_{d_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=1000,
        logger=logger,
        callbacks=[lr_monitor, ckpt_callback],
        accelerator="gpu",
        devices=[config["gpu-id"]],
        check_val_every_n_epoch=config["generate-image-every-n-epoch"],
        num_sanity_val_steps=1,
    )
    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )


def preload():
    device = f"cuda:{config['gpu-id']}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    print("Making dataloader...")
    # dataset = D.EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((config["img-size"][1:3])),
    #         transforms.ToTensor(),
    #         # DiffAugment(policy="color,translation,cutout"),
    #         # DiffAugment(policy=config["diffaug-policy"]),
    #     ]
    # )
    # loaders = {
    #     split: DataLoader(
    #         dataset=D.EEGImageDataset(D.Splitter(dataset, split_name=split), transform),
    #         batch_size=config["batch-size"],
    #         shuffle=True if split == "train" else False,
    #         drop_last=True,
    #         num_workers=4,
    #     )
    #     for split in ["train", "val", "test"]
    # }

    if config["img-size"][1] != 64:
        raise Exception("make pickle dataset first")
    if config["use-diffaug"]:
        dataset = loadDatasetPickle("eeg_image_dataset_64_diffaug_all")
    else:
        # dataset = loadDatasetPickle("eeg_image_dataset_64_diffaug_none")
        dataset = loadDatasetPickle("eeg_image_dataset_64_diffaug_none_norm")
    loaders = {
        split: DataLoader(
            dataset=dataset[split],
            batch_size=config["batch-size"],
            shuffle=True if split == "train" else False,
            drop_last=True,
            num_workers=4,
        )
        for split in ["train", "val", "test"]
    }
    print("Making Dataloader complete")
    return device, loaders


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--lr", type=float, default=1e-3)
    # parser.add_argument(
    #     "-O", "--optimizer", choices=("Adam", "AdamW", "SGD"), default="Adam"
    # )
    # parser.add_argument("-S", "--scheduler", choices=("LambdaLR"), default="LambdaLR")
    parser.add_argument("-LF", "--lambdafactor", type=float, default=0.99)
    parser.add_argument("-W", "--weightdecay", type=float, default=0)
    parser.add_argument("--ckpt", type=str, default="None")
    parser.add_argument("--feature_extractor_ckpt", type=str, default="None")
    parser.add_argument("-B", "--batchsize", type=int, default=16)
    parser.add_argument("-I", "--imgsize", type=int, default=64)
    parser.add_argument("-G", "--gpuid", type=int, default=2)

    args = parser.parse_args()

    if args.ckpt != "None":
        config["ckpt"] = args.ckpt
    if args.feature_extractor_ckpt != "None":
        config["feature-extractor-ckpt"] = args.feature_extractor_ckpt
    if args.lambdafactor != 0.99:
        config["lambda-factor"] = args.lambdafactor
    if args.weightdecay != 0:
        config["weight-decay"] = args.weightdecay
    if args.lr != 1e-3:
        config["lr"] = args.lr
    if args.batchsize != 16:
        config["batch-size"] = args.batchsize
    if args.imgsize != 64:
        config["img-size"] = (3, args.imgsize, args.imgsize)
    if args.gpuid != 2:
        config["gpu-id"] = args.gpuid


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parseArgs()
    device, loaders = preload()
    train(now)
