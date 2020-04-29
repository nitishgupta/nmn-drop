#!/usr/bin/env bash

TEST_FILE=$1
MODEL_TAR=$2
OUTPUT_FILE=$3
GPU=$4
INCLUDE_PACKAGE=$5

BEAMSIZE=$6

# "{"model": {"decoder_beam_search": {"beam_size": 16}, "wsideargs": true, "debug": ${DEBUG}}}"

allennlp evaluate --output-file ${OUTPUT_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  --overrides "{"model": {"decoder_beam_search": {"beam_size": ${BEAMSIZE} }}}" \
                  ${MODEL_TAR} ${TEST_FILE}

# --overrides "{"model": {"decoder_beam_search": {"beam_size": 16}}}" \