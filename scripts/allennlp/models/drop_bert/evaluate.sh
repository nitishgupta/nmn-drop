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

DATASET_DIR=./resources/data/drop_acl

# This should contain:
# 1. drop_dataset_mydev.json and drop_dataset_mytest.json
# 2. A folder containing multiple sub-dataset folders, each with dev and test .json
DATASET_NAME=preprocess
QUESTYPE_SETS_DIR=questype_datasets

VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json
TESTFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mytest.json

VAL_EVAL_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_eval.txt
TEST_EVAL_FILE=${PREDICTION_DIR}/${DATASET_NAME}_test_eval.txt

# Validation over complete dataset
allennlp evaluate --output-file ${VAL_EVAL_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${VALFILE}

# Test over complete dataset
#allennlp evaluate --output-file ${TEST_EVAL_FILE} \
#                  --cuda-device ${GPU} \
#                  --include-package ${INCLUDE_PACKAGE} \
#                  ${MODEL_TAR} ${TESTFILE}
#
#
#for EVAL_DATASET in datecomp_full year_diff count how_many_yards_was who_arg numcomp_full
#do
#    TRAINFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_train.json
#    VALFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_mydev.json
#    TESTFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_mytest.json
#
#    VAL_EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_eval.txt
#    TEST_EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_test_eval.txt
#
#    ###################################################################################################################
#
#    allennlp evaluate --output-file ${VAL_EVALUATION_FILE} \
#                      --cuda-device ${GPU} \
#                      --include-package ${INCLUDE_PACKAGE} \
#                      ${MODEL_TAR} ${VALFILE}
#
#    allennlp evaluate --output-file ${TEST_EVALUATION_FILE} \
#                      --cuda-device ${GPU} \
#                      --include-package ${INCLUDE_PACKAGE} \
#                      ${MODEL_TAR} ${TESTFILE}
#
#    echo -e "Dev Evaluations file saved at: ${VAL_EVALUATION_FILE}"
#    echo -e "Test Evaluations file saved at: ${TEST_EVALUATION_FILE}"
#done
