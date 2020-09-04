#!/usr/bin/env

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true
export INTERPRET=false

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-iclr21/iclr_qdmr-v2-noexc-aug/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FGS-DCYD

PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop/iclr21
DATASET_NAME=iclr_qdmr-v2-noexc
# iclr_qdmr-v1
# qdmr-v1
# iclr20_full-v1

# train/dev/test
SPLIT=dev

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_${SPLIT}.json

VIS_PREDICTOR=drop_parser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_visualize.txt

JSONL_PREDICTOR=drop_parser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_predictions.jsonl

METRICS_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_metrics.json


allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &

allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
                 --predictor ${JSONL_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &

allennlp evaluate --output-file ${METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                  ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
echo -e "Metrics file saved at: ${METRICS_FILE}"
