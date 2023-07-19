import os
from datetime import datetime, timedelta, timezone
from typing import List, Union

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import wandb
from image_datatset import CableCrackImageDataset
from model import CrackDetectionModel
from utils import fix_seed


def get_dataloader(
    dataset_path: Union[str, List[str]], batch_size: int = 8, shuffle: bool = True, use_sampler: bool = False
) -> DataLoader:
    """データローダーを返す。

    Args:
        dataset_path (str): データセットのあるパス
        batch_size (int, optional): バッチサイズ. Defaults to 8.

    Returns:
        DataLoader: データローダー
    """
    if type(dataset_path) is str:
        img_dir = os.path.abspath(dataset_path)
    elif type(dataset_path) is list:
        img_dir = list(map(os.path.abspath, dataset_path))
    else:
        assert False

    dataset_ = CableCrackImageDataset(images_directory=img_dir)
    if use_sampler:
        num_anomalous = np.array(dataset_.labels).sum()
        num_normal = len(dataset_.labels) - num_anomalous
        class_counts = [num_normal, num_anomalous]
        class_weights = 1 / torch.Tensor(class_counts)
        weights = [class_weights[int(i)] for i in dataset_.labels]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset_.labels), replacement=True)
        return DataLoader(dataset=dataset_, batch_size=batch_size, sampler=sampler)

    dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle)
    return dataloader_


def train(config):
    fix_seed(seed=config["seed"])
    dataloader_train = get_dataloader(
        config["train_data_dir"], config["train_batch_size"], shuffle=True, use_sampler=True
    )
    dataloader_val = get_dataloader(config["val_data_dir"], config["val_batch_size"], shuffle=False, use_sampler=False)
    dataloader_test = get_dataloader(
        config["test_data_dir"], config["test_batch_size"], shuffle=False, use_sampler=False
    )

    ####################
    # prepare callback
    ####################
    if wandb.run is not None:
        JST = timezone(timedelta(hours=+9), "JST")
        dt = datetime.fromtimestamp(wandb.run.start_time).replace(tzinfo=timezone.utc).astimezone(tz=JST)
        dt_iso = dt.isoformat()
        checkpoint = ModelCheckpoint(
            filename=f"{dt_iso}_{wandb.run.name}_" + "{epoch}_{step}_{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            dirpath="./saved_models/model_ckpt/",
        )
    else:
        checkpoint = ModelCheckpoint(
            filename="{epoch}_{step}_{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            dirpath="./saved_models/model_ckpt/",
        )

    wandb_logger = WandbLogger()

    ####################
    # train
    ####################
    model = CrackDetectionModel(model_name=config["model_name"], lr=config["lr"])
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    ####################
    # test
    ####################
    trainer.test(model=model, dataloaders=dataloader_test)


if __name__ == "__main__":
    model_name = [
        # "swin_v2_s",
        # "swin_v2_b",
        # "efficientnet_v2_s",
        "efficientnet_v2_m",
        # "convnext_small",
        # "convnext_base",
        # "convnext_large",
        # "vit_b_16",
    ]
    config = {
        "seed": 42,
        "model_name": model_name[0],
        "train_data_dir": [
            "dataset/re_annotation_crop_20230613_splited_preprocessing_cont_lap_lbp/train",
            "dataset/dev3_testing_correction_v2_splited_preprocessing_cont_lap_lbp/train",
        ],
        "val_data_dir": [
            "dataset/re_annotation_crop_20230613_splited_preprocessing_cont_lap_lbp/val",
            "dataset/dev3_testing_correction_v2_splited_preprocessing_cont_lap_lbp/val",
        ],
        "test_data_dir": [
            "dataset/re_annotation_crop_20230613_splited_preprocessing_cont_lap_lbp/test",
            "dataset/dev3_testing_correction_v2_splited_preprocessing_cont_lap_lbp/test",
        ],
        "train_batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "max_epochs": 10,
        "wandb_group": "first",
        "lr": 1e-6,
    }
    wandb.init(
        project="cable-crack-detection-v3",
        name="",
        config=config,
        group=config["wandb_group"],
        settings=wandb.Settings(start_method="fork"),
        job_type="train",
    )
    train(config)
    wandb.finish()
