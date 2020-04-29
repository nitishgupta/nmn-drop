#!/usr/bin/env bash

GPU=0

DATASET_NAME=date_num/date_ydNEW_num_hmyw_cnt_rel_600
# DATASET_NAME=date/datefull_yd_new2

DATASET_DIR=./resources/data/drop/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_mydev.json
TESTFILE=${DATASET_DIR}/drop_dataset_mytest.json

CONFIGFILE=training_config/naqanet.jsonnet

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}
export TEST_DATA_FILE=${TESTFILE}


export SEED=1

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/

CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=naqanet
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/S_${SEED}/NAQANET_wTest


allennlp train ${CONFIGFILE} -s ${SERIALIZATION_DIR}