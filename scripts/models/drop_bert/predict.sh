#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/drop_iclr600_wqdmr/drop_parser_bert/Qattn_true/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5_HEM_3_BM_1/S_42
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=drop_wqdmr_programs-ns

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

# Prediction output including verbose execution logs
VIS_PREDICTOR=drop_parser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/drop_qdmr_verbosepred.txt

JSONL_PREDICTOR=drop_parser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/drop_qdmr_predictions.jsonl

# Prediction output in a JSON-L file similar to MTMSN
#PREDICTOR=drop_mtmsnstyle_predictor
#PREDICTION_FILE=${PREDICTION_DIR}/drop_dev_preds_wDAS.jsonl

allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
                 --predictor ${JSONL_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE}

echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
