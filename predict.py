import argparse
import datetime
from glob import glob
import os
import sys
import cv2

import lightning as pl
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
import torchvision

from image_datatset import CableCrackImageDataset
from model import CrackDetectionModel
from preprocessing import equialize_hist


def get_orientation_by_path(path):
    filename = os.path.basename(os.path.abspath(path))
    if "左" in filename:
        return "L"
    elif "上" in filename:
        return "T"
    elif "右" in filename:
        return "R"
    elif "下" in filename:
        return "B"
    else:
        return


def write_progress(filepath: str, text: str):
    JST = datetime.timezone(datetime.timedelta(hours=+9), "JST")
    time_now = datetime.datetime.now(tz=JST).isoformat()
    try:
        with open(f"{filepath}/prediction.log", "a") as f:
            f.write(f"[{time_now}]: {text}\n")
    except FileNotFoundError() as e:
        print(e)
    print(text)


def predict(img_dir: str = ""):
    # 顧客PCのGPUメモリ容量に合わせる。(GTX1050)
    # TODO 本番では解除するように。
    torch.cuda.set_per_process_memory_fraction(0.2, 0)

    write_progress(img_dir, "=====Start Prediction=====")

    # 前処理
    write_progress(img_dir, "=====Session 1 Start Preprocessing=====")
    # 前処理実行
    img_dir_cor = os.path.join(img_dir, "Correction")
    img_dir_dist = os.path.join(img_dir, "Enhanced")
    os.makedirs(img_dir_dist, exist_ok=True)
    png_imgs_cor = sorted(glob(os.path.join(img_dir_cor, "**", "*.png"), recursive=True))
    png_imgs_cor = [img for img in png_imgs_cor if not "/all_bind_Correction/" in img]
    for png_path in png_imgs_cor:
        img_ = cv2.imread(png_path)
        img_ = equialize_hist(img_)

        filename = ".".join(os.path.basename(png_path).split(".")[:-1])
        # TODO フォルダ構成が違うので要変更。
        # 右左上下と分かれてる。
        cv2.imwrite(os.path.join(img_dir_dist, f"{filename}.jpg"), img_, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # オリジナル画像のサイズが大きいので縮小する。
    img_dir_cor = os.path.join(img_dir, "Original")
    img_dir_dist = os.path.join(img_dir, "Converted")
    os.makedirs(img_dir_dist, exist_ok=True)
    png_imgs_cor = sorted(glob(os.path.join(img_dir_cor, "**", "*.png"), recursive=True))
    png_imgs_cor = [img for img in png_imgs_cor if not "/all_bind_Correction/" in img]
    for png_path in png_imgs_cor:
        img_ = cv2.imread(png_path)
        # H, W, C = img_.shape
        img_ = cv2.resize(img_, dsize=None, fx=0.5, fy=0.5)

        filename = ".".join(os.path.basename(png_path).split(".")[:-1])
        cv2.imwrite(os.path.join(img_dir_dist, f"{filename}.jpg"), img_, [cv2.IMWRITE_JPEG_QUALITY, 95])

    write_progress(img_dir, "=====Session 1 Preprocessing Complete=====")

    write_progress(img_dir, "=====Session 2 Start Analysis=====")
    # データローダーの用意
    img_dir_cor = os.path.join(img_dir, "Enhanced")
    dataset_test = CableCrackImageDataset(images_directory=img_dir_cor)

    ##################
    # DUMMY
    ##################
    # dataset_test = CableCrackImageDataset(images_directory=img_dir)

    dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False)

    # モデルの用意
    model_path = "saved_models/model_ckpt/2023-08-22T22:17:01.500486+09:00_splendid-deluge-127_epoch=9_step=6980_val_loss=0.11.ckpt"
    model_name = os.path.basename(model_path)
    wandb_run_name = model_name.split("_")[1]
    model = CrackDetectionModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )
    predictions_raw = trainer.predict(model, dataloaders=dataloader_test, return_predictions=True)
    predictions = torch.where(torch.concatenate(predictions_raw) > 0.5, 1, 0).squeeze()
    write_progress(img_dir, "=====Session 2 Analysis Complete=====")

    show_analysis = True
    if show_analysis is True:
        print("accuracy", accuracy_score(dataset_test.labels, predictions))
        print("recall", recall_score(dataset_test.labels, predictions))
        print("precision", precision_score(dataset_test.labels, predictions))
        print(classification_report(dataset_test.labels, predictions))
        print(confusion_matrix(dataset_test.labels, predictions))

        # 見逃したサンプルの保存
        true_pos = torch.where(torch.tensor(dataset_test.labels) == 1)
        pred_neg = torch.where(predictions == 0)
        miss_sample_idx = set(true_pos[0].numpy()) & set(pred_neg[0].numpy())
        for idx in miss_sample_idx:
            img_path = dataset_test.all_imgs[idx]
            img_name = os.path.basename(img_path)
            img_type = img_name.split(".")[-1]
            img = torchvision.io.read_image(img_path)
            os.makedirs(f"./miss_sample/{wandb_run_name}", exist_ok=True)
            if img_type == "png":
                torchvision.io.write_png(img, f"./miss_sample/{wandb_run_name}/{img_name}")
            elif img_type == "jpg" or img_type == "jpeg":
                torchvision.io.write_jpeg(img, f"./miss_sample/{wandb_run_name}/{img_name}")
            else:
                assert False

    #########
    # 後処理 #
    #########
    write_progress(img_dir, "=====Session 3 Start Postprocessing=====")
    df = pd.DataFrame.from_dict({"imgs": dataset_test.all_imgs, "prediction": predictions})
    # 各方向（左上右下）毎に、それぞれ列に配置する。
    df["orientation"] = df["imgs"].apply(get_orientation_by_path)
    df_left = df[df["orientation"] == "L"].reset_index()
    df_top = df[df["orientation"] == "T"].reset_index()
    df_right = df[df["orientation"] == "R"].reset_index()
    df_bottom = df[df["orientation"] == "B"].reset_index()
    predictions_of_all_orientation = pd.concat(
        [
            df_left["prediction"],
            df_top["prediction"],
            df_right["prediction"],
            df_bottom["prediction"],
        ],
        axis=1,
    )

    # 推論結果を合算
    predictions_of_all_orientation["result"] = predictions_of_all_orientation.apply(lambda row: row.sum() > 0, axis=1)

    # 結果を保存
    df_for_save = pd.concat(
        [
            df_left["imgs"],
            df_top["imgs"],
            df_right["imgs"],
            df_bottom["imgs"],
            predictions_of_all_orientation["result"],
        ],
        axis=1,
        keys=["left", "top", "right", "bottom", "result"],
    )
    df_for_save.to_csv(f"{img_dir}/result_pred.csv", index=False)
    write_progress(img_dir, "=====Session 3 Postprocessing Complete=====")

    write_progress(img_dir, "=====ALL Complete=====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--result_dir_name", default=None)
    args = parser.parse_args()

    if args.result_dir_name is None:
        print("args.result_dir_name is None")
        sys.exit(1)
    img_dir = os.path.abspath(os.path.join(os.environ["HOME"], "videos", args.result_dir_name))

    #########
    # DUMMY #
    #########
    # img_dir = "dataset/re_annotation_crop_20230613_splited_preprocessing_eq_hist2/test"

    if img_dir is not None and os.path.isdir(img_dir):
        predict(img_dir=img_dir)
    else:
        print("img_dir = ", img_dir)
        raise OSError
