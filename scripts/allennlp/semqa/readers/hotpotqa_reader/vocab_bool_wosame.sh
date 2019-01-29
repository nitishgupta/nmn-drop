#!/usr/bin/env bash

INCLUDE_PACKAGE=semqa

# DATASET FILES
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
TRAINFILE=${DATASET_DIR}/train_goldcontexts.jsonl

# VOCAB DATASET_READER CONFIG
CONFIGFILE=allenconfigs/semqa/vocab/hotpotqa_vocab.jsonnet

#########    MODEL PARAMS  ######################
# Check CONFIGFILE for environment variables to set
export DATASET_READER="hotpotqa_reader"
export TOKEN_MIN_CNT=0
export TRAINING_DATA_FILE=${TRAINFILE}
export WORD_EMBED_FILE=./resources/embeddings/glove/glove.6B.50d.txt.gz

# OUTPUT DIR
VOCABDIR=./resources/semqa/vocabs/hotpotqa/bool_wosame/gold_contexts_onlyand

#######################################################################################################################
# Code below this shouldn't require changing for a reader
bash scripts/allennlp/base/vocab.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${VOCABDIR}
