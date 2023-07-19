import argparse
import os

import lightning as pl
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
import torchvision

from image_datatset import CableCrackImageDataset
from model import CrackDetectionModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="解析をしたいディレクトリを指定")

    # 顧客PCのGPUメモリ容量に合わせる。(GTX1050)
    # TODO 本番では解除するように。
    torch.cuda.set_per_process_memory_fraction(0.2, 0)

    img_dir_for_test = "./dataset/dev3_testing_correction_v2_splited/test"
    # img_dir_for_test = "./dataset/Correction"
    # img_dir_for_test = "dataset/splited_normal_anomalous_v5/test"
    dataset_test = CableCrackImageDataset(images_directory=img_dir_for_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False)
    # model = CrackDetectionModel.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=9-step=710.ckpt")
    model_path = "saved_models/model_ckpt/2023-07-05T22:21:47.341853+09:00_dainty-universe-95_epoch=8_step=2520_val_loss=0.14.ckpt"
    model_name = os.path.basename(model_path)
    wandb_run_name = model_name.split("_")[1]
    model = CrackDetectionModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )
    predictions_raw = trainer.predict(model, dataloaders=dataloader_test, return_predictions=True)
    predictions = torch.where(torch.concatenate(predictions_raw) > 0.5, 1, 0).squeeze()

    show_analysis = True
    if show_analysis is True:
        print(accuracy_score(dataset_test.labels, predictions))
        print(recall_score(dataset_test.labels, predictions))
        print(precision_score(dataset_test.labels, predictions))
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
    df_for_save.to_csv(f"{img_dir_for_test}/result_pred.csv", index=False)
    print()


if __name__ == "__main__":
    main()
