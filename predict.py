import argparse
import os

import lightning as pl
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader

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

    # img_dir_for_test = "./dataset/dev3_testing_correction_splited/test"
    # img_dir_for_test = "./dataset/Correction"
    img_dir_for_test = "dataset/splited_normal_anomalous_v5/test"
    dataset_test = CableCrackImageDataset(images_directory=img_dir_for_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False)
    # model = CrackDetectionModel.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=9-step=710.ckpt")
    model = CrackDetectionModel.load_from_checkpoint(
        "saved_models/model_ckpt/2023-06-09T20:17:22.500886+09:00_iconic-monkey-13_epoch=0_step=411_val_loss=0.17.ckpt"
    )
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
