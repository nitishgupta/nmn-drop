#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-w-qdmr/qdmr-filter-post-v5/drop_parser_bert/Qattn_true/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_0_HEM_0_BM_1/S_42_IO
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=qdmr-filter-post-v5
# drop_wqdmr_programs-ns-filtered
# drop_wqdmr_programs-ns
# drop_iclr_600

VALDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

VAL_METRICS_FILE=${PREDICTION_DIR}/${DATASET_NAME}_metrics.json

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALDATA_FILE}

echo -e "Metrics file at ${VAL_METRICS_FILE}"