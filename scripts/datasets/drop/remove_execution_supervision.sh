#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

NOEXEC=no_exec

INPUT_DATASET=date_ydNEW_num_hmyw_cnt_rel_600

INPUT_DIR=${ROOT_DIR}/date_num/${INPUT_DATASET}
OUTPUT_DIR=${ROOT_DIR}/date_num/${INPUT_DATASET}_NOEXC

mkdir ${OUTPUT_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.remove_execution_supervision --input_dir ${INPUT_DIR} \
                                                     --output_dir ${OUTPUT_DIR} \
