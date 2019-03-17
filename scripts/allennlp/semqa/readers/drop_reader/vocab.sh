#!/usr/bin/env bash

INCLUDE_PACKAGE=semqa

# DATASET FILES
DATASET_DIR=./resources/data/drop/date_num_subset
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json

# VOCAB DATASET_READER CONFIG
CONFIGFILE=allenconfigs/semqa/vocab/drop_vocab.jsonnet

#########    MODEL PARAMS  ######################
# Check CONFIGFILE for environment variables to set
export DATASET_READER="drop_reader"
export TRAINING_DATA_FILE=${TRAINFILE}

############  WARNING - Only one of the glove / bidaf / elmo blocks should be used #############################
export TOKENIDX="glove"
export WORD_EMBED_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

#export TOKENIDX="bidaf"
#export EXTEND_VOCAB=true
#export EXISTING_VOCAB_DIR=./resources/semqa/pretrained_bidaf/vocabulary

#export TOKENIDX="elmo"

# These are used to extend an existing vocab. For eg. When using pretrained bidaf, this would be bidaf's vocabulary

# OUTPUT DIR
VOCABDIR=./resources/semqa/vocabs/drop/date_num/tokens_${TOKENIDX}

#######################################################################################################################
# Code below this shouldn't require changing for a reader
bash scripts/allennlp/base/vocab.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${VOCABDIR}
