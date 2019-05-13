#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

PREPROCESS_DIR=${ROOT_DIR}/preprocess

WHORELOCATE_DIR=${ROOT_DIR}/num/who_relocate

mkdir ${WHORELOCATE_DIR}

python -m datasets.drop.preprocess.who_relocate.who_relocate --input_dir ${PREPROCESS_DIR} \
                                                             --output_dir ${WHORELOCATE_DIR}
