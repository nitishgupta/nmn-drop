#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=/shared/nitishg/semqa/checkpoints/drop/merged_data/date_yd_num_hmyw_cnt_whoarg_0_full/drop_parser_bert/EXCLOSS_false/MMLLOSS_true/aux_true/SUPEPOCHS_0/S_1/BeamSize2_HEM0_wSpanAns
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}


DATASET_DIR=./resources/data/drop_post_iclr
DATASET_NAME=merged_data/date_yd_num_hmyw_cnt_whoarg_1200_full
VALDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json
VAL_METRICS_FILE=${PREDICTION_DIR}/dev_metrics.json

# This should contain:
# 1. drop_dataset_mydev.json and drop_dataset_mytest.json
# 2. A folder containing multiple sub-dataset folders, each with dev and test .json
# DATASET_DIR=./resources/data/interpret-drop
# DATASET_NAME=interpret_dev

# VALDATA_FILE=/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno_v2.json
# VAL_METRICS_FILE=${PREDICTION_DIR}/interpret_dev_metrics.json

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALDATA_FILE}