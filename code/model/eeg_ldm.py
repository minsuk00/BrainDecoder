import torch
import numpy as np
import torch.nn as nn
import os
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm import tqdm, trange
from time import time
import datetime
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.transforms import transforms

import cv2
import sys

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.eeg_to_text_embedding import EEGtoTextEmbedding
import lookup_dict as LD

# from sample_level import SampleLevelFeatureExtractorNN

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname((os.path.abspath(__file__)))),
        "stable-diffusion",
    )
)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class eLDM:
    def __init__(self, cond_model_ckpt: str, use_pooling: bool = False):
        self.ckp_path = "/scratch/choi/model/stable-diffusion-v1-5/v1-5-pruned.ckpt"
        self.config_path = "/home/choi/BrainDecoder/pretrains/ldm/config.yaml"

        config = OmegaConf.load(self.config_path)
        model = load_model_from_config(config, self.ckp_path)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = device
        self.model = model
        self.ldm_config = config
        self.use_pooling = use_pooling

        # ckpt = "/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-02-02 10:34:21/version/checkpoints/epoch=992_val_loss=0.0229.ckpt"
        # if self.use_pooling:
        # cond_model_ckpt = "/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-05-04 20:08:40/version/checkpoints/epoch=238_val_loss=0.5985.ckpt"
        self.eeg_cond_model = EEGtoTextEmbedding.load_from_checkpoint(cond_model_ckpt)
        self.eeg_cond_model.use_pooling = self.use_pooling
        # else:
        #     cond_model_ckpt = "/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-02-02 10:34:21/version/checkpoints/epoch=992_val_loss=0.0229.ckpt"
        #     self.eeg_cond_model = SampleLevelFeatureExtractorNN.load_from_checkpoint(
        #         cond_model_ckpt
        #     )
        print(f"Loading conditional model... : {cond_model_ckpt}")

    @torch.no_grad()
    def generate(
        self,
        eeg_embedding,
        num_samples,
        ddim_steps: int = 50,
        limit: int = None,
        seed: int = -1,  # -1 for random seed
        uc_scale: float = 7.5,
    ):
        # fmri_embedding: n, seq_len, embed_dim

        if seed == -1:
            seed_everything(torch.randint(0, 100, ()))
        else:
            seed_everything(seed)

        w = 512  # image width, in pixel space
        h = 512  # image height, in pixel space
        f = 8  # downsampling factor
        c = 4  # latent channels
        shape = [c, h // f, w // f]

        model = self.model.to(self.device)
        cond_model = self.eeg_cond_model.to(self.device)

        # sampler = PLMSSampler(model)
        sampler = DDIMSampler(model)

        with model.ema_scope():
            all_samples = []
            model.eval()
            avg_time = 0
            begin = time()
            for count, item in enumerate(eeg_embedding):
                tmp_time = time()
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                if limit is not None:
                    if count >= limit:
                        break
                eeg, label, img_name = item
                eeg = eeg.unsqueeze(0)
                eeg = eeg.to(self.device)
                cond = cond_model(eeg.repeat(num_samples, 1, 1))
                print(
                    f"############# Label: {LD.id_to_name[LD.idx_to_id[label]]}, Image: {img_name} #############"
                )

                uc = None
                if uc_scale != 1.0:
                    uc = model.get_learned_conditioning(num_samples * [""])
                    # print("uc shape", uc.shape) # [2,77,768]
                    # print("cond shape", cond.shape) # [2,768]
                    # for pooling
                    if self.use_pooling:
                        # uc = uc[:, -1:, :]
                        uc = uc[:, :, :]
                        # cond = cond.unsqueeze(dim=1)
                        uc_cp = uc.clone().detach()
                        uc_cp[:, -1:, :] = cond.unsqueeze(dim=1)
                        cond = uc_cp
                        print("uc shape", uc.shape)  # [2,77,768]
                        print("cond shape", cond.shape)  # [2,768]

                    # cond = cond[:, -1:, :]

                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=cond,
                    batch_size=num_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=uc_scale,
                    unconditional_conditioning=uc,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                root_path = "/scratch/choi/dataset/Brain2Image/imageNet_images"
                img_path = os.path.join(
                    root_path, img_name.split("_")[0], img_name + ".JPEG"
                )
                gt_image = Image.open(img_path).convert("RGB")

                transform = transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.PILToTensor(),
                        transforms.ConvertImageDtype(float),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
                gt_image_torch = transform(gt_image)

                gt_image = rearrange(gt_image_torch, "c h w -> 1 c h w")  # w h c?
                gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)
                all_samples.append(
                    torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)
                )  # put groundtruth at first

                avg_time = (avg_time * count + (time() - tmp_time)) / (count + 1)
                print(
                    f"{count+1}/{len(eeg_embedding) if limit is None else limit}, eta: {datetime.timedelta(seconds=(avg_time*((len(eeg_embedding) if limit is None else limit)-count-1)))}, passed_time: {datetime.timedelta(seconds=(time()-begin))}"
                )

        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=num_samples + 1)

        model = model.to("cpu")

        return grid, (255.0 * torch.stack(all_samples, 0).cpu().numpy()).astype(
            np.uint8
        )
