#!/usr/bin/env bash

DATASET_1=/shared/nitishg/data/drop-w-qdmr/qdmr-v2-ss

DATASET_2=/shared/nitishg/data/drop-w-qdmr/drop_iclr_full_v2-ss

OUTPUT_DATASET=/shared/nitishg/data/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss

python -m datasets.drop.merge_datasets --dir1 ${DATASET_1} \
                                       --dir2 ${DATASET_2} \
                                       --outputdir ${OUTPUT_DATASET}