#!/usr/bin/env bash

INCLUDE_PACKAGE=semqa

# DATASET FILES
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
TRAINFILE=${DATASET_DIR}/train_resplit.jsonl

# VOCAB DATASET_READER CONFIG
CONFIGFILE=allenconfigs/semqa/vocab/hotpotqa_vocab.jsonnet

#########    MODEL PARAMS  ######################
# Check CONFIGFILE for environment variables to set
export DATASET_READER="hotpotqa_reader"
export TRAINING_DATA_FILE=${TRAINFILE}
# These are used to extend an existing vocab. For eg. When using pretrained bidaf, this would be bidaf's vocabulary
export EXISTING_VOCAB_DIR=./resources/semqa/pretrained_bidaf/vocabulary
export EXTEND_VOCAB=true
export W_SIDEARGS=false

# OUTPUT DIR
VOCABDIR=./resources/semqa/vocabs/hotpotqa/bool_wosame/gold_contexts/wosideargs_${W_SIDEARGS}_resplit

# Don't need since we're extending vocab from bidaf
#export TOKEN_MIN_CNT=0
#export WORD_EMBED_FILE=./resources/embeddings/glove/glove.6B.50d.txt.gz

#######################################################################################################################
# Code below this shouldn't require changing for a reader
bash scripts/allennlp/base/vocab.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${VOCABDIR}
