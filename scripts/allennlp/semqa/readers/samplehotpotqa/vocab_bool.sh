#!/usr/bin/env bash

INCLUDE_PACKAGE=semqa

# DATASET FILES
DATASET_DIR=/srv/local/data/nitishg/data/hotpotqa/processed
TRAINFILE=${DATASET_DIR}/train_bool.jsonl

# VOCAB DATASET_READER CONFIG
CONFIGFILE=allenconfigs/semqa/vocab/vocab_tokens.jsonnet

#########    MODEL PARAMS  ######################
# Check CONFIGFILE for environment variables to set
export DATASET_READER="sample_hotpot"
export TOKEN_MIN_CNT=0
export TRAINING_DATA_FILE=${TRAINFILE}

# OUTPUT DIR
VOCABDIR=/srv/local/data/nitishg/semqa/vocabs/hotpotqa_bool/sample_reader

#######################################################################################################################
# Code below this shouldn't require changing for a reader
bash scripts/allennlp/base/vocab.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${VOCABDIR}