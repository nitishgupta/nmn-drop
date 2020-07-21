#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=qdmr-filter-v2

TRAINFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=training_config/drop_ques_parser_bert.jsonnet

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export QATTLOSS=true

export BS=16
export DROPOUT=0.2

export SEED=5

export BEAMSIZE=1
export MAX_DECODE_STEP=14
export EPOCHS=40

export GC_FREQ=500
export PROFILE_FREQ=0
export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop-w-qdmr/${DATASET_NAME}
MODEL_DIR=drop_ques_parser
PD_1=BS_${BS}/Qattn_${QATTLOSS}/BM_${BEAMSIZE}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/S_${SEED}

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}
