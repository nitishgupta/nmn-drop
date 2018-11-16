#!/usr/bin/env


### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=/srv/local/data/nitishg/data/hotpotqa/processed
TRAINFILE=${DATASET_DIR}/train.jsonl
VALFILE=${DATASET_DIR}/devds.jsonl
# TESTFILE=${DATASET_DIR}/test.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/sample_parser.jsonnet

export DATASET_READER=sample_hotpot

# Check CONFIGFILE for environment variables to set
export VOCABDIR=/srv/local/data/nitishg/semqa/vocabs/semqa/hotpotqa/sample_reader/vocabulary
export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}
# export TEST_DATA_FILE=${TESTFILE}

export BS=4
export LR=0.0005
export DROPOUT=0.3

export BEAMSIZE=4
export MAX_DECODE_STEP=32


####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=/srv/local/data/nitishg/semqa/checkpoints

SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa
MODEL_DIR=sample_parser
PARAMETERS_DIR=BS_${BS}/LR_${LR}/Drop_${DROPOUT}/BeamSize_${BEAMSIZE}/MaxDecodeStep_${MAX_DECODE_STEP}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}