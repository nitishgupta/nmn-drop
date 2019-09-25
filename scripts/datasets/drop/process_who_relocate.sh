#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PREPROCESS_DIR=${ROOT_DIR}/preprocess

WHORELOCATE_DIR=${ROOT_DIR}/num/who_relocate_re

python -m datasets.drop.preprocess.who_relocate.relocate_wprogs --input_dir ${PREPROCESS_DIR} \
                                                                --output_dir ${WHORELOCATE_DIR}