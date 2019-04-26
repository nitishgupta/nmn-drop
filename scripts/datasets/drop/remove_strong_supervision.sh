#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_re

ANNOTATION_FOR_PARAS=100

INPUT_DIR=${ROOT_DIR}/date/datecomp_full
OUTPUT_DIR=${ROOT_DIR}/date/datecomp_${ANNOTATION_FOR_PARAS}

mkdir ${OUTPUT_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.remove_strong_supervision --input_dir ${INPUT_DIR} \
                                                  --output_dir ${OUTPUT_DIR} \
                                                  --annotation_for_numpassages ${ANNOTATION_FOR_PARAS}