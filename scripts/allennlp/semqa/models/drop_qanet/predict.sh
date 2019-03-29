#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/drop/date_subset
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

# All parameters here are used to fetch the correct serialization_dir
export TOKENIDX="qanet"

export BS=16
export DROPOUT=0.1

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/date_num
MODEL_DIR=drop_qanet
PARAMETERS_DIR=BS_${BS}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}


# PREDICTION DATASET
PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
mkdir ${PREDICT_OUTPUT_DIR}

#*****************    PREDICTION FILENAME   *****************
PRED_FILENAME=dev_predictions.txt

TESTFILE=${VALFILE}
MODEL_TAR=${SERIALIZATION_DIR}/model.tar.gz
PREDICTION_FILE=${PREDICT_OUTPUT_DIR}/${PRED_FILENAME}
PREDICTOR=drop_qanet_predictor

#######################################################################################################################


allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 --overrides "{ "model": {"debug": ${DEBUG}} }" \
                 ${MODEL_TAR} ${TESTFILE} \
                 --use-dataset-reader


echo -e "Predictions file saved at: ${PREDICTION_FILE}"
