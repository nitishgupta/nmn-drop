#!/usr/bin/env


### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed
TRAINFILE=${DATASET_DIR}/train_bool_wosame.jsonl
VALFILE=${DATASET_DIR}/devds_bool_wosame.jsonl
# TESTFILE=${DATASET_DIR}/test.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/sample_parser.jsonnet

export DATASET_READER=sample_hotpot

# Check CONFIGFILE for environment variables to set
export GPU=0
export VOCABDIR=./resources/semqa/vocabs/hotpotqa_bool_wosame/sample_reader/vocabulary
export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}
# export TEST_DATA_FILE=${TESTFILE}

export BS=1
export LR=0.001
export DROPOUT=0.2

export BEAMSIZE=32
export MAX_DECODE_STEP=7


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints

SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa_bool_wosame
MODEL_DIR=sample_parser
PARAMETERS_DIR=BS_${BS}/LR_${LR}/Drop_${DROPOUT}/BeamSize_${BEAMSIZE}/MaxDecodeStep_${MAX_DECODE_STEP}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
