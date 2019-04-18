#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DATE_NUM_DIR=${ROOT_DIR}/date_num
NUM_DIR=${ROOT_DIR}/num

NUM_PRUNE_DIR=${NUM_DIR}/num_prune
NUM_PRUNE_SUP_DIR=${NUM_DIR}/num_prune_supervised

mkdir ${NUM_PRUNE_DIR}
mkdir ${NUM_PRUNE_SUP_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.preprocess.numcomp.prune_numcomp  --input_trnfp ${DATE_NUM_DIR}/${TRAIN_FILENAME} \
                                                          --input_devfp ${DATE_NUM_DIR}/${DEV_FILENAME} \
                                                          --output_trnfp ${NUM_PRUNE_DIR}/${TRAIN_FILENAME} \
                                                          --output_devfp ${NUM_PRUNE_DIR}/${DEV_FILENAME}  \
                                                          --output_trntxt ${NUM_PRUNE_DIR}/train.txt  \
                                                          --output_devtxt ${NUM_PRUNE_DIR}/dev.txt  \


python -m datasets.drop.preprocess.numcomp.add_supervision  --input_trnfp ${NUM_PRUNE_DIR}/${TRAIN_FILENAME} \
                                                            --input_devfp ${NUM_PRUNE_DIR}/${DEV_FILENAME} \
                                                            --output_trnfp ${NUM_PRUNE_SUP_DIR}/${TRAIN_FILENAME} \
                                                            --output_devfp ${NUM_PRUNE_SUP_DIR}/${DEV_FILENAME}  \
                                                            --output_trntxt ${NUM_PRUNE_SUP_DIR}/train.txt  \
                                                            --output_devtxt ${NUM_PRUNE_SUP_DIR}/dev.txt