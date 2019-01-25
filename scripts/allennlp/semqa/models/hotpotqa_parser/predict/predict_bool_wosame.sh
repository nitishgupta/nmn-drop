#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed
VALFILE=${DATASET_DIR}/devds_bool_wosame.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=2
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BEAMSIZE=32
export MAX_DECODE_STEP=12

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints

SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa_bool_wosame
MODEL_DIR=sample_parser
PARAMETERS_DIR=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/BeamSize_${BEAMSIZE}/MaxDecodeStep_${MAX_DECODE_STEP}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR}_WClamp

PREDICT_OUTPUT_ROOT=./resources/semqa/predictions

DATASET=hotpotqa_bool_wosame
PREDICT_OUTPUT_DIR=${PREDICT_OUTPUT_ROOT}/${DATASET}/${MODEL_DIR}/${PARAMETERS_DIR}
mkdir -p ${PREDICT_OUTPUT_DIR}


TESTFILE=${VALFILE}
MODEL_TAR=${SERIALIZATION_DIR}/model.tar.gz
PREDICTION_FILE=${PREDICT_OUTPUT_DIR}/predictions.txt
PREDICTOR=hotpotqa_predictor


#######################################################################################################################

bash scripts/allennlp/base/predict.sh ${TESTFILE} \
                                      ${MODEL_TAR} \
                                      ${PREDICTION_FILE} \
                                      ${PREDICTOR} \
                                      ${GPU} \
                                      ${INCLUDE_PACKAGE}
