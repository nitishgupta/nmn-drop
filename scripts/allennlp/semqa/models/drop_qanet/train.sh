#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_NAME=date_prune
DATASET_DIR=./resources/data/drop/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

CONFIGFILE=allenconfigs/semqa/train/drop_qanet.jsonnet
export TOKENIDX="qanet"

export DATASET_READER=drop_reader

export WORDEMB_FILE="./resources/embeddings/glove.840B.300d.lower.converted.zip"

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export BS=16
export DROPOUT=0.2

export SEED=13

export EPOCHS=50

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_qanet
PARAMETERS_DIR=BS_${BS}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}/S_${SEED}

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}

#RESUME_SER_DIR=${SERIALIZATION_DIR}/Resume
#
#MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz
#
#bash scripts/allennlp/base/resume.sh ${CONFIGFILE} \
#                                     ${INCLUDE_PACKAGE} \
#                                     ${RESUME_SER_DIR} \
#                                     ${MODEL_TAR_GZ}