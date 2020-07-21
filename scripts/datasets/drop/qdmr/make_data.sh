#!/usr/bin/env bash

DROP_PREPROCESS=/shared/nitishg/data/drop-w-qdmr/preprocess
QDMR_DROP_DIR=/shared/nitishg/data/qdmr-processed/QDMR-high-level/DROP/drop-programs

QDMR_OUTPUT=/shared/nitishg/data/drop-w-qdmr/qdmr

python -m datasets.qdmr.process_drop_qdmr \
    --qdmr_json ${QDMR_DROP_DIR}/train.json \
    --drop_json ${DROP_PREPROCESS}/drop_dataset_train.json \
    --output_json /shared/nitishg/data/drop-w-qdmr/qdmr/processed/drop_dataset_train.json


python -m datasets.qdmr.process_drop_qdmr \
    --qdmr_json ${QDMR_DROP_DIR}/dev.json \
    --drop_json ${DROP_PREPROCESS}/drop_dataset_dev.json \
    --output_json /shared/nitishg/data/drop-w-qdmr/qdmr/processed/drop_dataset_dev.json


python -m datasets.qdmr.filter_dataset \
    --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/processed \
    --output_dir /shared/nitishg/data/drop-w-qdmr/qdmr/filtered \

python -m datasets.qdmr.postprocess_programs \
    --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/filtered \
    --output_dir /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1


python -m datasets.drop.aux_supervision.num_comparison --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1
python -m datasets.drop.aux_supervision.date_comparison --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1
python -m datasets.drop.aux_supervision.hmyw --input_dir /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1

cp -r /shared/nitishg/data/drop-w-qdmr/qdmr/qdmr-v1 /shared/nitishg/data/drop-w-qdmr/qdmr-v1

