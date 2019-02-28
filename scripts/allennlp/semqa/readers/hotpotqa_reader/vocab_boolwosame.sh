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
export TRAINING_DATA_FILE=${TRAINFILE}

############  WARNING - Only one of the glove / bidaf / elmo blocks should be used #############################
export TOKENIDX="elmoglove"
export WORD_EMBED_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

#export TOKENIDX="bidaf"
#export EXTEND_VOCAB=true
#export EXISTING_VOCAB_DIR=./resources/semqa/pretrained_bidaf/vocabulary

#export TOKENIDX="elmo"

# These are used to extend an existing vocab. For eg. When using pretrained bidaf, this would be bidaf's vocabulary

export W_SIDEARGS=false

# OUTPUT DIR
VOCABDIR=./resources/semqa/vocabs/hotpotqa/bool_wosame/gold_contexts/wosideargs_${W_SIDEARGS}/tokens_${TOKENIDX}

# Don't need since we're extending vocab from bidaf
#export TOKEN_MIN_CNT=0
#export WORD_EMBED_FILE=./resources/embeddings/glove/glove.6B.50d.txt.gz

#######################################################################################################################
# Code below this shouldn't require changing for a reader
bash scripts/allennlp/base/vocab.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${VOCABDIR}
