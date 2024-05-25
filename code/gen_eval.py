import argparse
import os
import torch
import sys
from PIL import Image
import numpy as np
from time import time
from datetime import timedelta
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

# sys.path.append("home/choi/BrainDecoder/code/model")
# import feature_extraction as FE

sys.path.append("home/choi/BrainDecoder/code")
import dataset as D
from model.eeg_ldm import eLDM
from eval_metrics import get_similarity_metric


def get_eval_metric(samples, avg=True):
    # metric_list = ["mse", "pcc", "ssim", "psm"]
    # print(samples.shape)  # (rows, col+1, 3, 512, 512)
    # metric_list = ["ssim"]
    metric_list = []
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), "n c h w -> n h w c")
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    # if False:
    begin = time()
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), "n c h w -> n h w c")
            res = get_similarity_metric(
                pred_images, gt_images, method="pair-wise", metric_name=m
            )
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    print(f"SSIM evaluation finished: {time()-begin}")
    begin = time()

    # inception score
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), "n c h w -> n h w c")
        res = get_similarity_metric(pred_images, gt_images, method="is")
        res_part.append(res)
    res_list.append(np.mean(res_part))
    metric_list.append("inception_score")
    res_part = []
    print(f"IS evaluation finished: {time()-begin}")
    begin = time()

    # n-way top-k accuracy
    # if False:
    # for s in samples_to_run:
    #     pred_images = [img[s] for img in samples]
    #     pred_images = rearrange(np.stack(pred_images), "n c h w -> n h w c")
    #     res = get_similarity_metric(
    #         pred_images,
    #         gt_images,
    #         "class",
    #         None,
    #         n_way=50,
    #         num_trials=1000,
    #         top_k=1,
    #         device="cuda",
    #     )
    #     res_part.append(np.mean(res))
    # res_list.append(np.mean(res_part))
    # res_list.append(np.max(res_part))
    # metric_list.append("top-1-class")
    # metric_list.append("top-1-class (max)")
    # print(f"ACC evaluation finished: {time()-begin}")

    return res_list, metric_list


def parse_args():
    parser = argparse.ArgumentParser(
        "EEG conditioned Latent Diffusion Model", add_help=False
    )

    # Conditional model
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-05-05 06:09:14/version/checkpoints/epoch=220_val_loss=0.5405.ckpt",
        help="ckpt path for eeg conditional model",
    )
    parser.add_argument(
        "--use_pooling",
        "-P",
        action="store_true",
        help="Whether to use pooling for EEG encoding",
    )

    # Evaluation / Generation
    parser.add_argument(
        "--eval",
        "-E",
        action="store_true",
        help="whether to generate and evaluate results",
    )
    parser.add_argument(
        "--row",
        "-R",
        type=int,
        default=2,
        help="number of EEG to generate from",
    )
    parser.add_argument(
        "--col",
        "-C",
        type=int,
        default=3,
        help="number of samples to generate for each EEG",
    )
    parser.add_argument(
        "--seed",
        "-S",
        type=int,
        default=-1,
        help="random seed for generation. -1 will be random seed between 0~100",
    )
    parser.add_argument(
        "--ddim_steps",
        "-D",
        type=int,
        default=50,
        help="steps for DDIM sampling",
    )
    parser.add_argument(
        "--uc_scale",
        "-U",
        type=float,
        default=7.5,
        help="scale for Classifier Free Guidance",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # args.ckpt = "/home/choi/BrainDecoder/lightning_logs/SampleLevelFeatureExtraction/2024-05-05 06:09:14/version/checkpoints/epoch=220_val_loss=0.5405.ckpt"

    print("Loading dataset...")
    test_dataset = D.Splitter(D.EEGDataset(), "test")
    print(f"Test dataset loaded. dataset length: {len(test_dataset)}")

    s_time = time()
    model = eLDM(
        cond_model_ckpt=args.ckpt,
        use_pooling=args.use_pooling,
    )
    grid, samples = model.generate(
        test_dataset,
        num_samples=args.col,
        limit=None if args.row == -1 else args.row,
        seed=args.seed,
        ddim_steps=args.ddim_steps,
        uc_scale=args.uc_scale,
    )
    gen_time = time() - s_time
    print(f"Image generation complete: {timedelta(seconds=gen_time)} elapsed")
    s_time = time()

    del model, test_dataset
    torch.cuda.empty_cache()

    # ssim, is, acc, acc(max)
    metric, metric_list = get_eval_metric(samples, avg=True)

    metric_dict = {metric_list[i]: metric[i] for i in range(len(metric_list))}
    print(metric_dict)

    outpath = "/home/choi/BrainDecoder/outputs/samplelevel2img-samples"
    grid_count = len(os.listdir(outpath)) - 1
    grid_imgs = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    grid_imgs = Image.fromarray(grid_imgs.astype(np.uint8))
    grid_imgs.save(os.path.join(outpath, f"grid-{grid_count:04}.png"))

    log_dir = (
        f"/home/choi/BrainDecoder/lightning_logs/gen_eval/{args.row}rows-{args.col}cols"
    )
    writer = SummaryWriter(log_dir)
    writer.add_image("eval/samples_test", grid)
    for metric, val in metric_dict.items():
        writer.add_scalar("eval/" + metric, val)
    # writer.add_scalar("eval/ssim", metric[0])
    # writer.add_scalar("eval/inception_score", metric[1])
    # # writer.add_scalar("eval/inception_score", metric[0])
    # writer.add_scalar("eval/top-1-class", metric[2])
    # writer.add_scalar("eval/top-1-class (max)", metric[3])
    writer.add_scalar("eval/eval-time", time() - s_time)
    writer.add_scalar("eval/gen-time", gen_time)
    writer.close()
    print(f"Evaluation complete: {timedelta(seconds=(time()-s_time))} elapsed")
