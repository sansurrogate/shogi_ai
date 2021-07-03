import argparse
import logging
import os
import re

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--output_df", type=str)
    args = parser.parse_args()
    return args


def find_all_files(directory):
    yield from (
        os.path.join(root, f)
        for root, dirs, files in os.walk(os.path.abspath(directory))
        for f in files
    )


def parse_csa(filename, ptn_rate):
    move_len = 0
    toryo = False
    rate = {}

    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        m = ptn_rate.match(line)
        if m:
            rate[m.group(1)] = float(m.group(2))
        if line[:1] == "+" or line[:1] == "-":
            move_len += 1
        if line == "%TORYO":
            toryo = True

    return (filename, move_len, toryo, rate.get('black'), rate.get('white'))


def main(args):
    logging.info(f"search kifu dir: {os.path.abspath(args.dir)}")
    ptn_rate = re.compile(r"^'(black|white)_rate:.*:(.*)$")
    pd_kifu_info = pd.DataFrame(
        (
            parse_csa(f, ptn_rate)
            for f in tqdm(find_all_files(args.dir))
            if os.path.splitext(f)[-1] == ".csa"
        ),
        columns=["filename", "move_len", "toryo", "rate_black", "rate_white"],
    )
    logging.info(f"saving dataframe to {args.output_df}")
    pd_kifu_info.to_pickle(args.output_df)


if __name__ == "__main__":
    args = parse_args()
    main(args)
