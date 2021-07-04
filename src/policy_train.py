import logging
from dataclasses import dataclass
from typing import Optional

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from lib.kifu_dataset import KifuDataset
from network.policy import PolicyNet


@dataclass
class Config:
    kifulist_train: str = "./data/dataset_information/kifulist_train.pickle"
    kifulist_valid: str = "./data/dataset_information/kifulist_valid.pickle"
    checkpoint: Optional[str] = None

    gpus: int = 0
    num_workers_dataloader: int = 6
    batch_size_train: int = 128
    batch_size_valid: int = 512
    epoch: int = 10
    lr: float = 1e-3
    log_interval: int = 100

    middle_ch: int = 192
    ksize: int = 3
    hidden_layer_num: int = 11
    batch_norm: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def train(
    path_kifulist_train,
    batch_size_train,
    path_kifulist_valid,
    batch_size_valid,
    gpus,
    num_workers_dataloader,
    epoch,
    lr,
    checkpoint,
    middle_ch,
    ksize,
    hidden_layer_num,
    batch_norm,
):
    dataset_train = KifuDataset(path_kifulist_train)
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size_train, shuffle=True
    )
    dataset_valid = KifuDataset(path_kifulist_valid)
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size_valid,
        num_workers=num_workers_dataloader,
    )

    if checkpoint is None:
        model = PolicyNet(
            middle_ch=middle_ch,
            ksize=ksize,
            hidden_layer_num=hidden_layer_num,
            batch_norm=batch_norm,
            lr=lr,
        )
    else:
        model = PolicyNet.load_from_checkpoint(checkpoint)

    trainer = pl.Trainer(
        gpus=gpus,
        val_check_interval=0.1,
        callbacks=[
            EarlyStopping(monitor="valid_loss"),
            ModelCheckpoint(monitor="valid_loss"),
        ],
        limit_val_batches=5,
        num_sanity_val_steps=1,
        max_epochs=epoch,
    )
    trainer.fit(model, loader_train, loader_valid)


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG,
    )
    train(
        to_absolute_path(cfg.kifulist_train),
        cfg.batch_size_train,
        to_absolute_path(cfg.kifulist_valid),
        cfg.batch_size_valid,
        cfg.gpus,
        cfg.num_workers_dataloader,
        cfg.epoch,
        cfg.lr,
        to_absolute_path(cfg.checkpoint) if cfg.checkpoint else None,
        cfg.middle_ch,
        cfg.ksize,
        cfg.hidden_layer_num,
        cfg.batch_norm,
    )


if __name__ == "__main__":
    main()
