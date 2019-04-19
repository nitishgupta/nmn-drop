#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PREPROCESS_DIR=${ROOT_DIR}/preprocess_new

DATE_DIR=${ROOT_DIR}/date

PRUNE_DATE_DIR=${DATE_DIR}/date_prune
PRUNE_DATE_AUGMENT_DIR=${DATE_DIR}/date_prune_augment

PRUNE_DATE_AUGMENT_50_DIR=${DATE_DIR}/date_prune_augment_50
PRUNE_DATE_AUGMENT_100_DIR=${DATE_DIR}/date_prune_augment_100

mkdir ${PRUNE_DATE_DIR}
mkdir ${PRUNE_DATE_AUGMENT_DIR}
mkdir ${PRUNE_DATE_AUGMENT_50_DIR}
mkdir ${PRUNE_DATE_AUGMENT_100_DIR}

TRAIN_FILENAME=drop_dataset_train.json
DEV_FILENAME=drop_dataset_dev.json

python -m datasets.drop.preprocess.datecomp.prune_date_comparison --input_trnfp ${PREPROCESS_DIR}/${TRAIN_FILENAME} \
                                                                  --input_devfp ${PREPROCESS_DIR}/${DEV_FILENAME} \
                                                                  --output_trnfp ${PRUNE_DATE_DIR}/${TRAIN_FILENAME} \
                                                                  --output_devfp ${PRUNE_DATE_DIR}/${DEV_FILENAME}  \


python -m datasets.drop.preprocess.datecomp.date_data_augmentation --input_trnfp ${PRUNE_DATE_DIR}/${TRAIN_FILENAME} \
                                                                   --input_devfp ${PRUNE_DATE_DIR}/${DEV_FILENAME} \
                                                                   --output_trnfp ${PRUNE_DATE_AUGMENT_DIR}/${TRAIN_FILENAME} \
                                                                   --output_devfp ${PRUNE_DATE_AUGMENT_DIR}/${DEV_FILENAME}  \


python -m datasets.drop.preprocess.datecomp.remove_dateques_supervision \
                                                        --input_trnfp ${PRUNE_DATE_AUGMENT_DIR}/${TRAIN_FILENAME} \
                                                        --input_devfp ${PRUNE_DATE_AUGMENT_DIR}/${DEV_FILENAME} \
                                                        --output_trnfp ${PRUNE_DATE_AUGMENT_100_DIR}/${TRAIN_FILENAME} \
                                                        --output_devfp ${PRUNE_DATE_AUGMENT_100_DIR}/${DEV_FILENAME}  \
                                                        --annotation_for_numpassages 100


python -m datasets.drop.preprocess.datecomp.remove_dateques_supervision \
                                                        --input_trnfp ${PRUNE_DATE_AUGMENT_DIR}/${TRAIN_FILENAME} \
                                                        --input_devfp ${PRUNE_DATE_AUGMENT_DIR}/${DEV_FILENAME} \
                                                        --output_trnfp ${PRUNE_DATE_AUGMENT_50_DIR}/${TRAIN_FILENAME} \
                                                        --output_devfp ${PRUNE_DATE_AUGMENT_50_DIR}/${DEV_FILENAME}  \
                                                        --annotation_for_numpassages 50

