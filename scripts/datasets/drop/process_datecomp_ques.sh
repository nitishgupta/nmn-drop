#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DATE_DIR=${ROOT_DIR}/date

PRUNE_DATE_DIR=${DATE_DIR}/date_prune
PRUNE_DATE_AUGMENT_DIR=${DATE_DIR}/date_prune_augment

mkdir ${PRUNE_DATE_DIR}
mkdir ${PRUNE_DATE_AUGMENT_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.preprocess.datecomp.prune_date_comparison --input_trnfp ${DATE_DIR}/${TRAIN_FILENAME} \
                                                                  --input_devfp ${DATE_DIR}/${DEV_FILENAME} \
                                                                  --output_trnfp ${PRUNE_DATE_DIR}/${TRAIN_FILENAME} \
                                                                  --output_devfp ${PRUNE_DATE_DIR}/${DEV_FILENAME}  \


python -m datasets.drop.preprocess.datecomp.date_data_augmentation --input_trnfp ${PRUNE_DATE_DIR}/${TRAIN_FILENAME} \
                                                                   --input_devfp ${PRUNE_DATE_DIR}/${DEV_FILENAME} \
                                                                   --output_trnfp ${PRUNE_DATE_AUGMENT_DIR}/${TRAIN_FILENAME} \
                                                                   --output_devfp ${PRUNE_DATE_AUGMENT_DIR}/${DEV_FILENAME}  \
