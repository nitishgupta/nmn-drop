#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

PREPROCESS_DIR=${ROOT_DIR}/preprocess

HOWMANYYARDS_DIR=${ROOT_DIR}/num/how_many_yards_was

LONGEST_DIR=${ROOT_DIR}/num/longest_shortest_yards

mkdir ${HOWMANYYARDS_DIR}
mkdir ${LONGEST_DIR}

python -m datasets.drop.preprocess.how_many_yards.how_many_yards    --input_dir ${PREPROCESS_DIR} \
                                                                    --output_dir ${HOWMANYYARDS_DIR}


python -m datasets.drop.preprocess.how_many_yards.longestshortest_ques    --input_dir ${HOWMANYYARDS_DIR} \
                                                                          --output_dir ${LONGEST_DIR}
