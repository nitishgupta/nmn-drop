#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/date_num/posticrl_1200/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5/S_10/Nov19Data-HardEM5-SentFilter
# iclr_1200/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5/S_1/IclrData-HardEM5-SentFilter
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=./resources/data/interpret-drop

# This should contain:
# 1. drop_dataset_mydev.json and drop_dataset_mytest.json
# 2. A folder containing multiple sub-dataset folders, each with dev and test .json
DATASET_NAME=interpret_dev

VALDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/interpret_dev_manual_wanno.json

VAL_METRICS_FILE=${PREDICTION_DIR}/interpret_dev_metrics.json

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALDATA_FILE}