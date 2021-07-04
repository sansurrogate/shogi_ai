import logging
import os
import pickle

from torch.utils.data import Dataset

from .read_kifu import read_kifu
from .features import make_input_features, make_output_label

logging.basicConfig(level=logging.DEBUG)


class KifuDataset(Dataset):
    def __init__(self, filepath, limit=None) -> None:
        super().__init__()
        self.limit = limit
        self.positions = None

        file_suffix = "_positions"
        abs_filepath = os.path.abspath(filepath)
        filebase = os.path.splitext(abs_filepath)[0]
        self.filepath = (
            abs_filepath
            if filebase.endswith(file_suffix)
            else filebase + file_suffix + ".pickle"
        )

        logging.info(f"start to load dataset {abs_filepath}.")
        if os.path.exists(self.filepath):
            logging.info(
                f"positions dataset (path={self.filepath}) "
                "has already been pickled. load it."
            )
            with open(self.filepath, "rb") as f:
                self.positions = pickle.load(f)

        if self.positions is None or (limit and len(self.positions) < limit):
            logging.info(
                f"positions dataset (path={self.filepath}) "
                "has not yet been pickled. start to construct."
            )
            with open(abs_filepath, "rb") as f:
                kifulist = pickle.load(f)
            if limit:
                kifulist = kifulist[:limit]
            self.positions = read_kifu(kifulist)
            logging.info(f"save positions dataset to {self.filepath}")
            with open(self.filepath, "wb") as f:
                pickle.dump(self.positions, f)

        if limit:
            self.positions = self.positions[:limit]

        logging.info("finish to load dataset.")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        (piece_bb, occupied, pieces_in_hand, move, win) = self.positions[idx]
        feature = make_input_features(piece_bb, occupied, pieces_in_hand)
        return feature, move, win
