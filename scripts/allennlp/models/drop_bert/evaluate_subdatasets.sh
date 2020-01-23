#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/date_num/date_yd_num_hmyw_cnt_whoarg_600/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5/S_1/BeamSize1
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=./resources/data/drop_post_iclr
DATASET_NAME=date_num/date_yd_num_hmyw_cnt_whoarg_600

VALDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mydev.json
TESTDATA_FILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mytest.json

VAL_METRICS_FILE=${PREDICTION_DIR}/drop_dev_metrics.json
TEST_METRICS_FILE=${PREDICTION_DIR}/drop_test_metrics.json

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

    VAL_METRICS_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_metrics.json
    TEST_METRICS_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_test_metrics.json

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
