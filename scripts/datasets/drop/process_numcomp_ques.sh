#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

PREPROCESS_DIR=${ROOT_DIR}/preprocess

NUM_PRUNE_DIR=${ROOT_DIR}/num/numcomp_prune
NUM_PRUNE_SUP_DIR=${ROOT_DIR}/num/numcomp_full

mkdir ${NUM_PRUNE_DIR}
mkdir ${NUM_PRUNE_SUP_DIR}

python -m datasets.drop.preprocess.numcomp.prune_numcomp  --input_dir ${PREPROCESS_DIR} \
                                                          --output_dir ${NUM_PRUNE_DIR}


python -m datasets.drop.preprocess.numcomp.add_supervision  --input_dir ${NUM_PRUNE_DIR} \
                                                            --output_dir ${NUM_PRUNE_SUP_DIR}