#!/usr/bin/env bash

ROOT_DIR=/shared/nitishg/data/drop_iclr

RAW_DIR=raw
PREPROCESS_DIR=preprocess

mkdir ${ROOT_DIR}/${PREPROCESS_DIR}

# PREPROCESS-TRAIN
python -m datasets.drop.preprocess.tokenize  --input_json ${ROOT_DIR}/${RAW_DIR}/drop_dataset_train.json \
                                             --output_json ${ROOT_DIR}/${PREPROCESS_DIR}/drop_dataset_train.json \
                                             --nump 20 &

# PREPROCESS-DEV
python -m datasets.drop.preprocess.tokenize  --input_json ${ROOT_DIR}/${RAW_DIR}/drop_dataset_dev.json \
                                             --output_json ${ROOT_DIR}/${PREPROCESS_DIR}/drop_dataset_dev.json \
                                             --nump 20