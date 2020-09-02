#!/usr/bin/env bash

DROP_PREPROCESS=/shared/nitishg/data/drop/preprocess
QDMR_DROP_DIR=/shared/nitishg/data/drop/qdmr-processed/drop-programs

QDMR_SUBSET_DIR=/shared/nitishg/data/drop/iclr21/qdmr_subsets-v2

QDMR_OUTPUT_DIR=/shared/nitishg/data/drop/iclr21

QDMR_DATASET_DIRNAME=qdmr-v2

python -m datasets.qdmr.process_drop_qdmr \
    --qdmr_json ${QDMR_DROP_DIR}/train.json \
    --drop_json ${DROP_PREPROCESS}/drop_dataset_train.json \
    --output_json ${QDMR_SUBSET_DIR}/processed/drop_dataset_train.json


python -m datasets.qdmr.process_drop_qdmr \
    --qdmr_json ${QDMR_DROP_DIR}/dev.json \
    --drop_json ${DROP_PREPROCESS}/drop_dataset_dev.json \
    --output_json ${QDMR_SUBSET_DIR}/processed/drop_dataset_dev.json


printf "\n\nFiltering QDMR data\n\n"
python -m datasets.qdmr.filter_dataset \
    --input_dir ${QDMR_SUBSET_DIR}/processed \
    --output_dir ${QDMR_SUBSET_DIR}/${QDMR_DATASET_DIRNAME} \

#printf "\n\nPost-processing QDMR data\n"
#python -m datasets.qdmr.postprocess_programs \
#    --input_dir ${QDMR_SUBSET_DIR}/filtered \
#    --output_dir ${QDMR_SUBSET_DIR}/${QDMR_DATASET_DIRNAME}

printf "\n\nAdding intermediate execution supervision\n"
# Avoid num_comparison supervision for it is noisy and hurts performance
# python -m datasets.drop.aux_supervision.num_comparison --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1
python -m datasets.drop.aux_supervision.date_comparison --input_dir ${QDMR_SUBSET_DIR}/${QDMR_DATASET_DIRNAME}
python -m datasets.drop.aux_supervision.hmyw --input_dir ${QDMR_SUBSET_DIR}/${QDMR_DATASET_DIRNAME}

cp -r ${QDMR_SUBSET_DIR}/${QDMR_DATASET_DIRNAME} ${QDMR_OUTPUT_DIR}

