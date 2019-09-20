#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PERC_SPLIT=0.2

INPUT_DIR=${ROOT_DIR}/date_num/date_ydNEW_num_hmyw_cnt_rel_600
OUTPUT_DIR=${ROOT_DIR}/date_num/date_ydNEW_num_hmyw_cnt_rel_600


python -m datasets.drop.split_dev --input_dir=${INPUT_DIR} \
                                  --output_dir=${OUTPUT_DIR} \
                                  --perc_split=${PERC_SPLIT}