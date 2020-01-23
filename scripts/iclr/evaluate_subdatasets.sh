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
TESTDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mytest.json

VAL_METRICS_FILE=${PREDICTION_DIR}/drop_mydev_metrics.json
TEST_METRICS_FILE=${PREDICTION_DIR}/drop_mytest_metrics.json

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALDATA_FILE}

# Test over complete dataset
allennlp evaluate --output-file ${TEST_METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TESTDATA_FILE}

QUESTYPE_SETS_DIR=questype_datasets
for EVAL_DATASET in datecomp_full year_diff numcomp_full who_arg count how_many_yards_was
do
    VALFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_mydev.json
    TESTFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_mytest.json

    VAL_METRICS_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_mydev_metrics.json
    TEST_METRICS_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_mytest_metrics.json

    ###################################################################################################################

    allennlp evaluate --output-file ${VAL_METRICS_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${VALFILE}

    allennlp evaluate --output-file ${TEST_METRICS_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${TESTFILE}

    echo -e "Dev Evaluations file saved at: ${VAL_METRICS_FILE}"
    # echo -e "Test Evaluations file saved at: ${TEST_EVALUATION_FILE}"
done
