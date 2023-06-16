import os
import shutil
from glob import glob

from sklearn.model_selection import train_test_split


def os_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def main():
    img_dir = "./dataset/re_annotation_crop_20230613"
    labels = ["anomalous", "normal"]

    # split
    imgs_dict = dict()
    for label in labels:
        img_set = {"train": None, "val": None, "test": None}
        imgs = glob(os.path.abspath(os.path.join(img_dir, label, "*.png")))
        # train, val, test = 6 : 2 : 2
        img_train, img_test = train_test_split(imgs, test_size=0.4)
        img_val, img_test = train_test_split(img_test, test_size=0.5)
        img_set["train"] = img_train
        img_set["val"] = img_val
        img_set["test"] = img_test
        imgs_dict[label] = img_set

    # 格納用のフォルダを作る
    for mode in ["train", "val", "test"]:
        os_mkdir(f"{img_dir}_splited")
        os_mkdir(f"{img_dir}_splited/{mode}")
        for label in labels:
            os_mkdir(f"{img_dir}_splited/{mode}/{label}")

    # copy
    for mode in ["train", "val", "test"]:
        for label in labels:
            for img_path in imgs_dict[label][mode]:
                shutil.copy2(img_path, f"{img_dir}_splited/{mode}/{label}")


if __name__ == "__main__":
    main()
