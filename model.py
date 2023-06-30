from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from torch import nn
from torch.nn import functional as F
from torchvision.models.efficientnet import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_m,
)
from torchvision.models.swin_transformer import Swin_V2_S_Weights, swin_v2_s, Swin_V2_B_Weights, swin_v2_b
from torchvision.models.convnext import (
    ConvNeXt_Small_Weights,
    convnext_small,
    ConvNeXt_Base_Weights,
    convnext_base,
    ConvNeXt_Large_Weights,
    convnext_large,
)
from torchvision.models.vision_transformer import ViT_B_16_Weights, vit_b_16


class CrackDetectionModel(pl.LightningModule):
    def __init__(self, model_name: str, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_name == "efficientnet_v2_s":
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        elif model_name == "efficientnet_v2_m":
            self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        elif model_name == "swin_v2_s":
            self.model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
            self.model.head = nn.Linear(self.model.head.in_features, 1)
        elif model_name == "swin_v2_b":
            self.model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            self.model.head = nn.Linear(self.model.head.in_features, 1)
        elif model_name == "convnext_small":
            self.model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        elif model_name == "convnext_base":
            self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        elif model_name == "convnext_large":
            self.model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        elif model_name == "vit_b_16":
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 1)
        else:
            assert False
        self.model.eval()

        self.test_step_y_pred = []
        self.test_step_y_true = []

    def forward(self, batch) -> Any:
        img, label = batch
        output = torch.sigmoid(self.model(img))
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, label = batch
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output.view(-1), label.type_as(output))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img, label = batch
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output.view(-1), label.type_as(output))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Calculate validation metrics
        preds = (output > 0.5).float().view(-1).cpu()
        label = label.cpu()
        self.log("val_acc", accuracy_score(label, preds), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img, label = batch
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output.view(-1), label.type_as(output))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Calculate test metrics
        preds = (output > 0.5).float().view(-1).cpu()
        label = label.cpu()
        # self.log("test_acc", accuracy_score(label, preds), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_recall", recall_score(label, preds), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(
        #     "test_precision", precision_score(label, preds), on_step=True, on_epoch=True, prog_bar=True, logger=True
        # )
        self.test_step_y_pred.append(preds)
        self.test_step_y_true.append(label)
        return preds

    def on_test_epoch_end(self) -> None:
        all_preds = torch.cat(self.test_step_y_pred)
        all_label = torch.cat(self.test_step_y_true)
        self.log("test_acc", accuracy_score(all_label, all_preds))
        self.log("test_recall", recall_score(all_label, all_preds))
        self.log("test_precision", precision_score(all_label, all_preds))

    def on_test_end(self) -> None:
        all_preds = torch.cat(self.test_step_y_pred)
        all_label = torch.cat(self.test_step_y_true)
        print(confusion_matrix(all_label, all_preds))
        print(classification_report(all_label, all_preds))

        self.test_step_y_pred.clear()
        self.test_step_y_true.clear()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
