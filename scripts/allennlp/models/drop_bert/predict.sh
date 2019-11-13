#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=/shared/nitishg/semqa/checkpoints/drop/iclr_subm/model_checkpoints_cleaned/S_100/BertModel_wTest
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop_iclr/iclr_subm
QUESTYPE_SETS_DIR=questype_datasets

# This should contain:
# 1. drop_dataset_mydev.json and drop_dataset_mytest.json
# 2. A folder containing multiple sub-dataset folders, each with dev and test .json
DATASET_NAME=date_ydNEW_num_hmyw_cnt_rel_600

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_mydev.json
PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_dev_preds.txt
# PREDICTOR=drop_analysis_predictor
PREDICTOR=drop_parser_predictor

#allennlp predict --output-file ${PREDICTION_FILE} \
#                     --predictor ${PREDICTOR} \
#                     --cuda-device ${GPU} \
#                     --include-package ${INCLUDE_PACKAGE} \
#                     --silent \
#                     --batch-size 4 \
#                     --use-dataset-reader \
#                     --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
#                    ${MODEL_TAR} ${FULL_VALFILE}


for EVAL_DATASET in datecomp_full year_diff_re count how_many_yards_was who_relocate_re numcomp_full
# for EVAL_DATASET in datecomp_full year_diff_re numcomp_full how_many_yards_was
do
    VALFILE=${DATASET_DIR}/${DATASET_NAME}/${QUESTYPE_SETS_DIR}/${EVAL_DATASET}/drop_dataset_mydev.json

    PREDICTOR=drop_parser_predictor
    VAL_PRED_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_preds.txt

    ###################################################################################################################

    allennlp predict --output-file ${VAL_PRED_FILE} \
                     --predictor ${PREDICTOR} \
                     --cuda-device ${GPU} \
                     --include-package ${INCLUDE_PACKAGE} \
                     --silent \
                     --batch-size 4 \
                     --use-dataset-reader \
                     --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                    ${MODEL_TAR} ${VALFILE}

    echo -e "Predictions file saved at: ${VAL_PRED_FILE}"
done


echo -e "Full mydev predictions file saved at: ${PREDICTION_FILE}"