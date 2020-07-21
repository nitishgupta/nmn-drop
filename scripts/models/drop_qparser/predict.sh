#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-w-qdmr/qdmr-filter-v2/drop_ques_parser/BS_16/Qattn_true/BM_1/S_5
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=qdmr-filter-v2
# qdmr-filter-v2-ss
# qdmr-filter-post-v6
# drop_iclr_600

TRAIN_OR_DEV=train

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_${TRAIN_OR_DEV}.json

VIS_PREDICTOR=drop_qparser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_visualize.txt

JSONL_PREDICTOR=drop_qparser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_predictions.jsonl

METRICS_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${TRAIN_OR_DEV}_metrics.json


#allennlp predict --output-file ${VIS_PREDICTION_FILE} \
#                 --predictor ${VIS_PREDICTOR} \
#                 --cuda-device ${GPU} \
#                 --include-package ${INCLUDE_PACKAGE} \
#                 --silent \
#                 --batch-size 4 \
#                 --use-dataset-reader \
#                 --overrides "{"model": { "beam_size": ${BEAMSIZE}}}" \
#                 ${MODEL_TAR} ${FULL_VALFILE}

allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
                 --predictor ${JSONL_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}
#
#allennlp evaluate --output-file ${METRICS_FILE} \
#                  --cuda-device ${GPU} \
#                  --include-package ${INCLUDE_PACKAGE} \
#                  ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
echo -e "Metrics file saved at: ${METRICS_FILE}"
