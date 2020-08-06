#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/squad-nmn/squad/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_HEM_0_BM_1/S_42_PreBIO
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data
DATASET_NAME=squad

TRAIN_OR_DEV=dev

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/squad-${TRAIN_OR_DEV}-v1.1_drop.json

VIS_PREDICTOR=drop_parser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_visualize.txt

JSONL_PREDICTOR=drop_parser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_predictions.jsonl

METRICS_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_metrics.json


allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 8 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &


#allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
#                 --predictor ${JSONL_PREDICTOR} \
#                 --cuda-device ${GPU} \
#                 --include-package ${INCLUDE_PACKAGE} \
#                 --silent \
#                 --batch-size 4 \
#                 --use-dataset-reader \
#                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
#                 ${MODEL_TAR} ${FULL_VALFILE} &
#
#
#allennlp evaluate --output-file ${METRICS_FILE} \
#                  --cuda-device ${GPU} \
#                  --include-package ${INCLUDE_PACKAGE} \
#                  ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
echo -e "Metrics file saved at: ${METRICS_FILE}"
