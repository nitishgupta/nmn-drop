#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

NOEXEC=no_exec

INPUT_DATASET=date_numcq_hmvy_cnt_relprog_500

INPUT_DIR=${ROOT_DIR}/date_num/${INPUT_DATASET}
OUTPUT_DIR=${ROOT_DIR}/date_num/${INPUT_DATASET}_${NOEXEC}

mkdir ${OUTPUT_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.remove_execution_supervision --input_dir ${INPUT_DIR} \
                                                     --output_dir ${OUTPUT_DIR} \
