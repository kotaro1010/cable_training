import os
from datetime import datetime, timedelta, timezone

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from image_datatset import CableCrackImageDataset
from model import CrackDetectionModel
from utils import fix_seed


def get_dataloader(dataset_path: str, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """データローダーを返す。

    Args:
        dataset_path (str): データセットのあるパス
        batch_size (int, optional): バッチサイズ. Defaults to 8.

    Returns:
        DataLoader: データローダー
    """
    img_dir = os.path.abspath(dataset_path)
    assert os.path.isdir(img_dir)

    dataset_train = CableCrackImageDataset(images_directory=img_dir)
    dataloader_ = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle)
    return dataloader_


def train(config):
    fix_seed(seed=config["seed"])
    dataloader_train = get_dataloader(config["train_data_dir"], config["train_batch_size"])
    dataloader_val = get_dataloader(config["val_data_dir"], config["val_batch_size"])
    dataloader_test = get_dataloader(config["test_data_dir"], config["test_batch_size"], shuffle=False)

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
        "efficientnet_v2_s",
        # "convnext_small",
        # "vit_b_16",
    ]
    config = {
        "seed": 42,
        "model_name": model_name[0],
        # "train_data_dir": "./dataset/dev3_testing_correction_splited",
        "train_data_dir": "dataset/re_annotation_crop_20230613_splited/train",
        "val_data_dir": "dataset/re_annotation_crop_20230613_splited/val",
        "test_data_dir": "dataset/dev3_testing_correction",
        "train_batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "max_epochs": 10,
        "wandb_group": "first",
        "lr": 1e-5,
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
