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

export GOLDACTIONS=true

export BS=8
export DROPOUT=0.2

export WEMB_DIM=300
export LR=0.001
export RG=1e-4

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/date_num
MODEL_DIR=drop_parser
PARAMETERS_DIR1=BS_${BS}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/GOLDAC_${GOLDACTIONS}/ED_${WEMB_DIM}/LR_${LR}/RG_${RG}
# PARAMETERS_DIR2=AUXGPLOSS_${AUXGPLOSS}/ATTCOV_${ATTCOVLOSS}/PTREX_${PTREX}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}_qan_m3_test


# PREDICTION DATASET
PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
mkdir ${PREDICT_OUTPUT_DIR}

#*****************    PREDICTION FILENAME   *****************
PRED_FILENAME=dev_evaluation.txt

TEST_FILE=${VALFILE}
MODEL_TAR=${SERIALIZATION_DIR}/model.tar.gz
PREDICTION_FILE=${PREDICT_OUTPUT_DIR}/${PRED_FILENAME}
PREDICTOR=drop_parser_predictor

#######################################################################################################################

allennlp evaluate --output-file ${PREDICTION_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TEST_FILE}

# --overrides "{"model": {"decoder_beam_search": {"beam_size": ${BEAMSIZE} }}}"

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
