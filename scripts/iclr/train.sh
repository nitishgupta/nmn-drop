#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/iclr_cameraready
DATASET_NAME=iclr_drop_data

TRAINFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mydev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_parser_bert.jsonnet

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
SERIALIZATION_DIR=./resources/iclr_cameraready/my_ckpt

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
