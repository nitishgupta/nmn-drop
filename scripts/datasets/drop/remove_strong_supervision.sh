#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

ANNOTATION_FOR_PARAS=600

INPUT_DIR=${ROOT_DIR}/date_num/date_ydre_num_hmyw_cnt_relre
OUTPUT_DIR=${ROOT_DIR}/date_num/date_ydre_num_hmyw_cnt_relre_${ANNOTATION_FOR_PARAS}


python -m datasets.drop.remove_strong_supervision --input_dir ${INPUT_DIR} \
                                                  --output_dir ${OUTPUT_DIR} \
                                                  --annotation_for_numpassages ${ANNOTATION_FOR_PARAS}