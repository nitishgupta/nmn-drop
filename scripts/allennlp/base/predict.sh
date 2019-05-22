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
                 --overrides "{ "model": {"debug": ${DEBUG}} }" \
                 ${MODEL_TAR} ${TEST_FILE}

# --overrides "{"model": {"decoder_beam_search": {"beam_size": ${BEAMSIZE}}, "debug": ${DEBUG}}}" \
# --weights-file ./resources/semqa/checkpoints/hotpotqa_model/bool_wosame/hotpotqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_false/DA_NOPROJ_true/DA_WT_true/QSPANEMB_true/AUXLOSS_false/QENTLOSS_true_cdist_cov/model_state_epoch_1.th \