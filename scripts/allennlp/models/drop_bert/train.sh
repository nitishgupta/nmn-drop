#!/usr/bin/env

# export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_NAME=date_num/date_yd_num_hmyw_cnt_whoarg_600
# DATASET_NAME=date/datefull_yd_new2

DATASET_DIR=./resources/data/drop_iclr/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_mydev.json
TESTFILE=${DATASET_DIR}/drop_dataset_mytest.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_parser_bert.jsonnet

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

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=5

export BS=4
export DROPOUT=0.2

export SEED=1000

export BEAMSIZE=1
export MAX_DECODE_STEP=14
export EPOCHS=45

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_parser_bert
PD_1=CNTFIX_${COUNT_FIXED}/EXCLOSS_${EXCLOSS}/MMLLOSS_${MMLLOSS}/aux_${AUXLOSS}/SUPEPOCHS_${SUPEPOCHS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/S_${SEED}/BertModel_wTest_ICLR

# SERIALIZATION_DIR=./resources/semqa/checkpoints/test

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
