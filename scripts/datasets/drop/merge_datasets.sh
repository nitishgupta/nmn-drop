#!/usr/bin/env bash

DATASET_1=/shared/nitishg/data/drop-w-qdmr/qdmr-v1-ss

DATASET_2=/shared/nitishg/data/drop-w-qdmr/drop_iclr_full-ss

OUTPUT_DATASET=/shared/nitishg/data/drop-w-qdmr/qdmr-v1_iclrfull-ss

python -m datasets.drop.merge_datasets --dir1 ${DATASET_1} \
                                       --dir2 ${DATASET_2} \
                                       --outputdir ${OUTPUT_DATASET}