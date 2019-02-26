#!/usr/bin/env bash

TEST_FILE=$1
MODEL_TAR=$2
OUTPUT_FILE=$3
PREDICTOR=$4
GPU=$5
INCLUDE_PACKAGE=$6

DEBUG=$7
BEAMSIZE=$8

# "{"model": {"decoder_beam_search": {"beam_size": 16}, "wsideargs": true, "debug": ${DEBUG}}}"

allennlp predict --output-file ${OUTPUT_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 --overrides "{"model": {"decoder_beam_search": {"beam_size": ${BEAMSIZE}}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${TEST_FILE}

# --overrides "{"model": {"decoder_beam_search": {"beam_size": 16}}}" \