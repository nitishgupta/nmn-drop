#!/usr/bin/env bash

MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/naqanet-2019.03.01.tar.gz'
GPU=0

CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop
MODEL_DIR=naqanet
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}

PREDICTION_DIR=${SERIALIZATION_DIR}/predictions

mkdir -p ${PREDICTION_DIR}

SUBFOLDER=date_num

for EVAL_DATASET in date_numcq_hmvy_cnt_filter
do
    DATASET_DIR=./resources/data/drop_s/${SUBFOLDER}/${EVAL_DATASET}
    TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
    VALFILE=${DATASET_DIR}/drop_dataset_dev.json

    TESTFILE=${VALFILE}

    PREDICTION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_pred.txt
    EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_eval.txt

    allennlp evaluate --output-file ${EVALUATION_FILE} \
                      --cuda-device ${GPU} \
                      ${MODEL_TAR} ${TESTFILE}


#    allennlp predict --output-file ${PREDICTION_FILE} \
#                     --predictor 'machine-comprehension' \
#                     --cuda-device ${GPU} \
#                     --silent \
#                     --batch-size 1 \
#                     --use-dataset-reader \
#                     ${MODEL_TAR} ${TESTFILE}

    echo -e "Evaluation file saved at: ${EVALUATION_FILE}"
    echo -e "Predictions file saved at: ${PREDICTION_FILE}"
done
