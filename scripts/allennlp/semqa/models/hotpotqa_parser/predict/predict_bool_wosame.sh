#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
VALFILE=${DATASET_DIR}/devds_resplit.jsonl

#*****************    PREDICTION FILENAME   *****************
PRED_FILENAME=devds_resplit.txt

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=4
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BEAMSIZE=1
export MAX_DECODE_STEP=12

export BIDAF_CONTEXT_KEY='encoded_passage'
export BOOL_QSTRQENT_FUNC='rawbidaf'
export W_SIDEARGS=false
export GOLDACTIONS=true

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints

SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa/bool_wosame
MODEL_DIR=hotpotqa_parser
PARAMETERS_DIR1=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/B_CONTEXT_${BIDAF_CONTEXT_KEY}/FUNC_${BOOL_QSTRQENT_FUNC}
PARAMETERS_DIR2=SIDEARG_${W_SIDEARGS}/GOLDAC_${GOLDACTIONS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/${PARAMETERS_DIR2}_resplit_bigram

# PREDICTION DATASET
PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
mkdir ${PREDICT_OUTPUT_DIR}

TESTFILE=${VALFILE}
MODEL_TAR=${SERIALIZATION_DIR}/model.tar.gz
PREDICTION_FILE=${PREDICT_OUTPUT_DIR}/${PRED_FILENAME}
PREDICTOR=hotpotqa_predictor

#######################################################################################################################

bash scripts/allennlp/base/predict.sh ${TESTFILE} \
                                      ${MODEL_TAR} \
                                      ${PREDICTION_FILE} \
                                      ${PREDICTOR} \
                                      ${GPU} \
                                      ${INCLUDE_PACKAGE}

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
