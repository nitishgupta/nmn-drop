#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-w-qdmr/ss-minmax/drop_parser_bert/Qattn_true/EXCLOSS_false/aux_true/IO_false/SHRDSUB_true/SUPEPOCHS_0_HEM_0_BM_1/S_10_sumattn
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=ss-minmax
# qdmr-filter-post-v6
# drop_iclr_600

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

VIS_PREDICTOR=drop_parser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_visualize.txt

JSONL_PREDICTOR=drop_parser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_predictions.jsonl


allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

#allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
#                 --predictor ${JSONL_PREDICTOR} \
#                 --cuda-device ${GPU} \
#                 --include-package ${INCLUDE_PACKAGE} \
#                 --silent \
#                 --batch-size 4 \
#                 --use-dataset-reader \
#                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
#                 ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
