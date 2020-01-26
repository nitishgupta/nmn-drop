#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=-1
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/iclr_cameraready/ckpt
MODEL_TAR=${MODEL_DIR}/model.tar.gz

PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=./resources/iclr_cameraready
DATASET_NAME=iclr_drop_data

VALDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mydev.json

VAL_METRICS_FILE=${PREDICTION_DIR}/drop_mydev_metrics.json

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --overrides "{"model": {"cuda_device": ${GPU} }}" \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALDATA_FILE}