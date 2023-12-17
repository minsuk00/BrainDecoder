import lightning as L
from torch.utils.data import DataLoader
import argparse

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "code", "model"))
import feature_extraction as FE

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "code"))
import dataset as D


parser = argparse.ArgumentParser()

# parser.add_argument("-O", "--optim", type=str, default="Adam")
parser.add_argument("-M", "--model_class", type=str)
parser.add_argument("-C", "--ckpt_file", type=str)


def get_model(model_class, ckpt_path):
    model = None
    if model_class == "fe":
        model = FE.FeatureExtractorNN.load_from_checkpoint(ckpt_path)

    return model


def get_dataloader(model):
    dataset = D.EEGDataset(eeg_dataset_file_name="eeg_signals_raw_with_mean_std.pth")
    dataloader = DataLoader(
        D.Splitter(dataset, split_name="test"),
        batch_size=model.hparams.batch_size,
        shuffle=False,
        num_workers=23,
        drop_last=True,
    )

    return


gpu_id = 2
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model = get_model(args.model_class, args.ckpt_path)
    test_dataloader = get_dataloader(model)

    trainer = L.Trainer(accelerator="gpu", devices=gpu_id)
    trainer.test(model, dataloaders=test_dataloader)
