#!/usr/bin/env bash

ROOT_DIR=/shared/nitishg/data/drop/iclr21

ICLR20_DIR=${ROOT_DIR}/iclr20_full-v3
QDMR_DIR=${ROOT_DIR}/qdmr-v3

MERGED_DATA=${ROOT_DIR}/iclr_qdmr-v3

MERGED_DATA_NOEXC=${MERGED_DATA}-noexc

# Splitting ICLR20 into 90 / 10
python -m datasets.drop.split_dev \
  --input_dir ${ICLR20_DIR} \
  --keeporig_dirname original \
  --dev_perc 0.1

# Splitting QDMR into 85 / 15 -- since QDMR is 4762/773(15%) train/dev
python -m datasets.drop.split_dev \
  --input_dir ${QDMR_DIR} \
  --keeporig_dirname original \
  --dev_perc 0.15


python -m datasets.drop.merge_datasets \
  --dir1 ${ICLR20_DIR} \
  --dir2 ${QDMR_DIR} \
  --outputdir ${MERGED_DATA}

python -m datasets.drop.remove_execution_supervision \
  --input_dir ${MERGED_DATA} \
  --output_dir ${MERGED_DATA_NOEXC}
