#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
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

# Prediction output including verbose execution logs
PREDICTOR=drop_parser_predictor
PREDICTION_FILE=${PREDICTION_DIR}/drop_mydev_verbosepred.txt

# Prediction output in a JSON-L file similar to MTMSN
#PREDICTOR=drop_mtmsnstyle_predictor
#PREDICTION_FILE=${PREDICTION_DIR}/drop_dev_preds_wDAS.jsonl

allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${VALDATA_FILE}

echo -e "Dev predictions file saved at: ${PREDICTION_FILE}"
