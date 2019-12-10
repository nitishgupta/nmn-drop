#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/merged_data/nov19_100_full/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_3/S_1/NoSpanAnsMod
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=./resources/data/drop_post_iclr
DATASET_NAME=preprocess

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
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

echo -e "Dev predictions file saved at: ${PREDICTION_FILE}"
