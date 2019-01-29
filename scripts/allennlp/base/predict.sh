#!/usr/bin/env bash

TEST_FILE=$1
MODEL_TAR=$2
OUTPUT_FILE=$3
PREDICTOR=$4
GPU=$5
INCLUDE_PACKAGE=$6

allennlp predict --output-file ${OUTPUT_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 ${MODEL_TAR} ${TEST_FILE}

