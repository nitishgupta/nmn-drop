#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

ANNOTATION_FOR_PARAS=200

INPUT_DIR=${ROOT_DIR}/date_num
OUTPUT_DIR=${ROOT_DIR}/date_num_${ANNOTATION_FOR_PARAS}

mkdir ${OUTPUT_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.remove_strong_supervision --input_trnfp ${INPUT_DIR}/${TRAIN_FILENAME} \
                                                             --input_devfp ${INPUT_DIR}/${DEV_FILENAME} \
                                                             --output_trnfp ${OUTPUT_DIR}/${TRAIN_FILENAME} \
                                                             --output_devfp ${OUTPUT_DIR}/${DEV_FILENAME}  \
                                                             --annotation_for_numpassages ${ANNOTATION_FOR_PARAS}