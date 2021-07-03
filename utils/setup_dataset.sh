#!/bin/bash

echo "download dataset"
curl -L https://osdn.net/projects/shogi-server/downloads/68500/wdoor2016.7z/ \
    -o $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/raw/wdoor2016.7z
curl -L http://wdoor.c.u-tokyo.ac.jp/shogi/x/wdoor2017.7z \
    -o $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/raw/wdoor2017.7z

echo "extract dataset"
7z x $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/raw/wdoor2016.7z \
    -o$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/kifu_csa
7z x $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/raw/wdoor2017.7z \
    -o$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/kifu_csa

echo "extract information from dataset"
poetry run python \
    $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/utils/extract_information_from_csa.py \
    --dir $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/kifu_csa \
    --output_df $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/dataset_information/kifu_info.pickle

echo "clense and split dataset"
poetry run python \
    $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/utils/cleanse_and_split.py \
    --dataset_information $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/dataset_information/kifu_info.pickle

echo "clense and split dataset for dev"
poetry run python \
    $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/utils/cleanse_and_split.py \
    --dataset_information $(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)/data/dataset_information/kifu_info.pickle \
     --limit=32 --suffix=_dev
