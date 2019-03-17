#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PREPROCESS_DIR=${ROOT_DIR}/preprocess

DATE_DIR=${ROOT_DIR}/date_subset
NUM_DIR=${ROOT_DIR}/num_subset
DATE_NUM_DIR=${ROOT_DIR}/date_num_subset

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json


python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${TRAIN_FILENAME} \
                                              --output_json ${DATE_DIR}/${TRAIN_FILENAME} \
                                              --keep_date

python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${DEV_FILENAME} \
                                              --output_json ${DATE_DIR}/${DEV_FILENAME} \
                                              --keep_date

python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${TRAIN_FILENAME} \
                                              --output_json ${NUM_DIR}/${TRAIN_FILENAME} \
                                              --keep_num

python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${DEV_FILENAME} \
                                              --output_json ${NUM_DIR}/${DEV_FILENAME} \
                                              --keep_num

python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${TRAIN_FILENAME} \
                                              --output_json ${DATE_NUM_DIR}/${TRAIN_FILENAME} \
                                              --keep_date --keep_num

python -m datasets.drop.preprocess.prune_ques --input_json ${PREPROCESS_DIR}/${DEV_FILENAME} \
                                              --output_json ${DATE_NUM_DIR}/${DEV_FILENAME} \
                                              --keep_date --keep_num