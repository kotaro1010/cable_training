from glob import glob
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class CableCrackImageDataset(Dataset):
    def __init__(self, images_directory, input_shape: Tuple[int, int] = (224, 224)):
        pngs = glob(f"{images_directory}/**/*.png", recursive=True)
        jpgs = glob(f"{images_directory}/**/*.jpg", recursive=True)
        all_imgs = pngs + jpgs
        all_imgs = [img_path for img_path in all_imgs if "all_bind_Correction" not in img_path]  # 展開図画像を除く
        self.all_imgs = sorted(all_imgs)
        self.labels = [1 if "/anomalous/" in path else 0 for path in all_imgs]

        self.autoaug = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        # self.autoaug = transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)
        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),  # swin-t, efficientNetV2, convnext
                # transforms.Resize((224, 224)),  # ViT
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize to ImageNet stats
            ]
        )

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.all_imgs[idx])
        # TODO 実装。
        # img = preprocessing(img)
        img = self.autoaug(img)
        img = self.transform(img.float())
        img = img.float()
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.all_imgs)
