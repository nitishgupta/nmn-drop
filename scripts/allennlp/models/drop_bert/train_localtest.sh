#!/usr/bin/env

# export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_NAME=merged_data/my1200_full

DATASET_DIR=./resources/data/drop_acl/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/small_sample_train.json
VALFILE=${DATASET_DIR}/small_sample_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_bert_test.jsonnet

export DATASET_READER="drop_reader_bert"

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}
export TEST_DATA_FILE=${TESTFILE}

export COUNT_FIXED=false
export AUXLOSS=true

export DENLOSS=true
export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

# 0 will not run HardEM
export HARDEM_EPOCH=0

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=0

export BS=4
export DROPOUT=0.2

export SEED=1

export BEAMSIZE=2
export MAX_DECODE_STEP=14
export EPOCHS=42

export GC_FREQ=500
export PROFILE_FREQ=0
export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
SERIALIZATION_DIR=./resources/semqa/checkpoints/test

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
