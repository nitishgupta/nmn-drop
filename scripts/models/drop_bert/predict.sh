#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/test-1.0
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=drop_wqdmr_programs

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

# Prediction output including verbose execution logs
PREDICTOR=drop_parser_predictor
PREDICTION_FILE=${PREDICTION_DIR}/drop_dev_verbosepred.txt

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
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG} }}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

echo -e "Dev predictions file saved at: ${PREDICTION_FILE}"
