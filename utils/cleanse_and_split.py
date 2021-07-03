import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_information", type=str)
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    args = parser.parse_args()
    return args


def clense(dataset_information):
    pd_kifu_info = pd.read_pickle(dataset_information)
    pd_kifu_info["high_quality"] = pd_kifu_info.apply(
        lambda x: (
            x["move_len"] > 50
            and x["toryo"]
            and x.get("rate_black", 0) >= 2500
            and x.get("rate_white", 0) >= 2500
        ),
        axis=1,
    )
    return pd_kifu_info[pd_kifu_info["high_quality"]]


def split(df, train_ratio, valid_ratio):
    mask = np.random.rand(len(df)) < train_ratio / (train_ratio + valid_ratio)
    df_train = df[mask]
    df_valid = df[~mask]
    return df_train, df_valid


def main(args):
    filepath = os.path.abspath(args.dataset_information)
    logging.info(f"cleanse and split {filepath}.")
    dirname = os.path.dirname(filepath)
    df = clense(filepath)
    if args.limit:
        logging.info(f"limit record num to {args.limit}.")
        df = df.sample(n=args.limit)
    df_train, df_valid = split(df, args.train_ratio, args.valid_ratio)

    filepath_train = os.path.join(
        dirname, f"kifulist_train{args.suffix}.pickle"
    )
    filepath_valid = os.path.join(
        dirname, f"kifulist_valid{args.suffix}.pickle"
    )
    logging.info(f"size: train={len(df_train)}, valid={len(df_valid)}")
    logging.info(f"save train to {filepath_train}, valid to {filepath_valid}.")
    with open(filepath_train, "wb") as f:
        pickle.dump(list(df_train["filename"]), f)
    with open(filepath_valid, "wb") as f:
        pickle.dump(list(df_valid["filename"]), f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
