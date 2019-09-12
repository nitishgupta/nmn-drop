#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

ANNOTATION_FOR_PARAS=500

INPUT_DIR=${ROOT_DIR}/date_num/date_ydnew2_num_hmyw_cnt_rel
OUTPUT_DIR=${ROOT_DIR}/date_num/date_ydnew2_num_hmyw_cnt_rel_${ANNOTATION_FOR_PARAS}


python -m datasets.drop.remove_strong_supervision --input_dir ${INPUT_DIR} \
                                                  --output_dir ${OUTPUT_DIR} \
                                                  --annotation_for_numpassages ${ANNOTATION_FOR_PARAS}