#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

PREPROCESS_DIR=${ROOT_DIR}/preprocess

YEARDIFF_DIR=${ROOT_DIR}/date/year_diff

mkdir ${YEARDIFF_DIR}

python -m datasets.drop.preprocess.year_diff.year_diff  --input_dir ${PREPROCESS_DIR} \
                                                        --output_dir ${YEARDIFF_DIR}
