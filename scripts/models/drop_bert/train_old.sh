#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_NAME=date_num/date_yd_num_hmyw_cnt_whoarg_1200

DATASET_DIR=./resources/data/drop_post_iclr/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_mydev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=training_config/drop_parser_bert_v2.jsonnet

export DATASET_READER="drop_reader_bert"

export SCALING_BERT=false

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export COUNT_FIXED=false
export AUXLOSS=true

export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

export INTERPRET=false

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=5

# -1 will not run HardEM; HardEM will kick after EPOCH num of epochs
export HARDEM_EPOCH=5

export BS=4
export DROPOUT=0.2

export SEED=1

export BEAMSIZE=1
export MAX_DECODE_STEP=14
export EPOCHS=40

export GC_FREQ=500
export PROFILE_FREQ=0
export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_parser_bert
PD_1=EXCLOSS_${EXCLOSS}/MMLLOSS_${MMLLOSS}/aux_${AUXLOSS}/SUPEPOCHS_${SUPEPOCHS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/S_${SEED}/BeamSize1

SERIALIZATION_DIR=./resources/semqa/checkpoints/test-1.0

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}
