#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/merged_data/iclr20_full/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_0/S_1/al0.9-composed-num_Cl-20
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=./resources/data/drop_acl/merged_data

DATASET_NAME=iclr20_full

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json
# PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_numstepanalysis.tsv
# PREDICTOR=drop_analysis_predictor
PREDICTOR=drop_parser_predictor

PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_pred.txt

allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 8 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

echo -e "Dev predictions file saved at: ${PREDICTION_FILE}"