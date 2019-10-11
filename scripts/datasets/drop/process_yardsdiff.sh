#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_iclr

PREPROCESS_DIR=${ROOT_DIR}/preprocess

YARDSDIFF_DIR=${ROOT_DIR}/num/yardsdiff

python -m datasets.drop.preprocess.how_many_yards.yards_difference  --input_dir ${PREPROCESS_DIR} \
                                                                    --output_dir ${YARDSDIFF_DIR}