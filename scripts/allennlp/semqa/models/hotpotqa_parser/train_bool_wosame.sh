#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
TRAINFILE=${DATASET_DIR}/train_goldcontexts.jsonl
VALFILE=${DATASET_DIR}/devds_goldcontexts.jsonl
# TESTFILE=${DATASET_DIR}/test.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/hotpotqa_parser.jsonnet

export DATASET_READER=hotpotqa_reader

# Check CONFIGFILE for environment variables to set
export GPU=0
export VOCABDIR=./resources/semqa/vocabs/hotpotqa/bool_wosame/gold_contexts_onlyand/vocabulary
export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}
# export TEST_DATA_FILE=${TESTFILE}

export WORD_EMBED_FILE=./resources/embeddings/glove/glove.6B.100d.txt.gz

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

export BS=2
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BEAMSIZE=32
export MAX_DECODE_STEP=12
export EPOCHS=10

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa/bool_wosame
MODEL_DIR=hotpotqa_parser
PARAMETERS_DIR=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/BeamSize_${BEAMSIZE}/MaxDecodeStep_${MAX_DECODE_STEP}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}_onlyAND

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
