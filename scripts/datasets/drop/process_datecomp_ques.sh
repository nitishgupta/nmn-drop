#!/usr/bin/env bash

ROOT_DIR=/shared/nitishg/data/drop_re

PREPROCESS_DIR=${ROOT_DIR}/preprocess

DATE_DIR=${ROOT_DIR}/date

PRUNE_DATE_DIR=${DATE_DIR}/datecomp_prune_new
PRUNE_DATE_AUGMENT_DIR=${DATE_DIR}/datecomp_full_new


python -m datasets.drop.preprocess.datecomp.date_comparison_prune --input_dir ${PREPROCESS_DIR} \
                                                                  --output_dir ${PRUNE_DATE_DIR}


python -m datasets.drop.preprocess.datecomp.date_data_augmentation --input_dir ${PRUNE_DATE_DIR} \
                                                                   --output_dir ${PRUNE_DATE_AUGMENT_DIR}