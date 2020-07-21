#!/usr/bin/env bash

# This should be output from datasets.qdmr.process_drop_qdmr and datasets.qdmr.postprocess_programs
INPUT_DIR=/shared/nitishg/data/drop-w-qdmr/qdmr-v1

# This should be output from datasets.qdmr.process_drop_qdmr and datasets.qdmr.postprocess_programs
OUTPUT_DIR=/shared/nitishg/data/drop-w-qdmr/qdmr-filter-v2


python -m datasets.qdmr.filter_dataset --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR}

# Aux supervision
python -m datasets.drop.aux_supervision.num_comparison --input_dir ${OUTPUT_DIR}
python -m datasets.drop.aux_supervision.date_comparison --input_dir ${OUTPUT_DIR}
python -m datasets.drop.aux_supervision.hmyw --input_dir ${OUTPUT_DIR}

