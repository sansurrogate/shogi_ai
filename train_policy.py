import logging
from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from lib.kifu_dataset import KifuDataset
from lib.network.policy import PolicyNet


@dataclass
class Config:
    kifulist_train: str = "./data/dataset_information/kifulist_train.pickle"
    kifulist_valid: str = "./data/dataset_information/kifulist_valid.pickle"
    log: str = "./data/artiafcts/log"
    model: str = "./data/artifacts/model"
    optim: str = "./data/artifacts/optimizer"
    # load_model: str
    # load_optim: str

    batch_size_train: int = 128
    batch_size_valid: int = 512
    epoch: int = 10
    lr: float = 1e-3
    log_interval: int = 100


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def train(
    path_kifulist_train,
    batch_size_train,
    path_kifulist_valid,
    batch_size_valid,
    epoch,
    lr,
):
    dataset_train = KifuDataset(path_kifulist_train)
    loader_train = DataLoader(
        dataset_train, batch_size=batch_size_train, shuffle=True
    )
    dataset_valid = KifuDataset(path_kifulist_valid)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size_valid)

    model = PolicyNet(lr=lr)
    trainer = pl.Trainer(
        val_check_interval=0.1,
        callbacks=[EarlyStopping(monitor="valid_loss")],
        limit_val_batches=5,
        max_epochs=epoch,
    )
    trainer.fit(model, loader_train, loader_valid)


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y%m%d %H:%M:%S",
        filename=cfg.log,
        level=logging.DEBUG,
    )
    train(
        to_absolute_path(cfg.kifulist_train),
        cfg.batch_size_train,
        to_absolute_path(cfg.kifulist_valid),
        cfg.batch_size_valid,
        cfg.epoch,
        cfg.lr,
    )


if __name__ == "__main__":
    main()
