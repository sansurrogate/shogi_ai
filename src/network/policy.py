import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import common


class PolicyNet(pl.LightningModule):
    def __init__(
        self,
        in_ch=104,
        middle_ch=192,
        out_ch=common.MOVE_DIRECTION_LABEL_NUM,
        ksize=3,
        hidden_layer_num=11,
        batch_norm=True,
        lr=1e-3,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.middle_ch = middle_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.hidden_layer_num = hidden_layer_num
        self.lr = lr

        if batch_norm:
            self.input_layer = nn.Sequential(
                nn.Conv2d(in_ch, middle_ch, ksize, padding="same"),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(),
            )
            self.hidden_layer = nn.Sequential(
                *sum(  # flatten 2d list
                    [
                        [
                            nn.Conv2d(
                                middle_ch, middle_ch, ksize, padding="same"
                            ),
                            nn.BatchNorm2d(middle_ch),
                            nn.ReLU(),
                        ]
                        for _ in range(hidden_layer_num)
                    ],
                    [],
                )
            )
        else:
            self.input_layer = nn.Sequential(
                nn.Conv2d(in_ch, middle_ch, ksize, padding="same"), nn.ReLU()
            )
            self.hidden_layer = nn.Sequential(
                *sum(  # flatten 2d list
                    [
                        [
                            nn.Conv2d(
                                middle_ch, middle_ch, ksize, padding="same"
                            ),
                            nn.ReLU(),
                        ]
                        for _ in range(hidden_layer_num)
                    ],
                    [],
                )
            )

        self.conv_output = nn.Conv2d(middle_ch, out_ch, 1)

    def forward(self, x):
        x = x.float()
        h1 = self.input_layer(x)
        h2 = self.hidden_layer(h1)
        output = self.conv_output(h2)
        return output.reshape(*x.shape[:-3], -1)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        count = float(y.shape[0])
        loss = F.cross_entropy(
            y_hat.reshape(-1, y_hat.shape[-1]), y, reduction="sum"
        ).item()
        probs = F.log_softmax(y_hat, dim=-1)
        corrects = (probs.argmax(dim=-1) == y).sum().float().item()
        return {"count": count, "loss": loss, "corrects": corrects}

    def validation_epoch_end(self, validation_step_outputs):
        total_count = sum([res["count"] for res in validation_step_outputs])
        avg_loss = (
            sum([res["loss"] for res in validation_step_outputs]) / total_count
        )
        accuracy = (
            sum([res["corrects"] for res in validation_step_outputs])
            / total_count
        )
        self.log(
            "valid_loss", avg_loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "valid_accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.reshape(-1, y_hat.shape[-1]), y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
