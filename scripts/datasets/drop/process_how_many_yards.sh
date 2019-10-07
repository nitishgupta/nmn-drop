#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PREPROCESS_DIR=${ROOT_DIR}/preprocess

HOWMANYYARDS_DIR=${ROOT_DIR}/num/how_many_yards_was

COUNT_DIR=${ROOT_DIR}/num/count

DIFF_DIR=${ROOT_DIR}/num/yardsdifference

python -m datasets.drop.preprocess.how_many_yards.how_many_yards  --input_dir ${PREPROCESS_DIR} \
                                                                  --output_dir ${HOWMANYYARDS_DIR} \
                                                                  --qattn \
                                                                  --numground


python -m datasets.drop.preprocess.how_many_yards.count_ques    --input_dir ${PREPROCESS_DIR} \
                                                                --output_dir ${COUNT_DIR}


python -m datasets.drop.preprocess.how_many_yards.yards_difference  --input_dir ${PREPROCESS_DIR} \
                                                                    --output_dir ${DIFF_DIR}


