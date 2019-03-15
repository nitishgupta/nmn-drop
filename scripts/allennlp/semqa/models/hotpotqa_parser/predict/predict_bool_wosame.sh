#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
VALFILE=${DATASET_DIR}/devds_goldcontexts.jsonl
TRAINFILE=${DATASET_DIR}/train_goldcontexts.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

# All parameters here are used to fetch the correct serialization_dir
export TOKENIDX="glove"
export W_SIDEARGS=true

export BS=4
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BOOL_QSTRQENT_FUNC='snli'
export QTK='encoded'
export CTK='modeled'

export DA_WT=true
export DA_NOPROJ=true

export GOLDACTIONS=false
export AUXGPLOSS=false
export QENTLOSS=true
export ATTCOVLOSS=false

export PTREX=true
# export PTRWTS="./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/PTREX_false/best.th"
# export PTRWTS=""

# These parameters are passed as overrides to the predict call
export DEBUG=false
export BEAMSIZE=1000

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints

CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hpqa/b_wsame
MODEL_DIR=hpqa_parser
PARAMETERS_DIR1=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/FUNC_${BOOL_QSTRQENT_FUNC}
PARAMETERS_DIR2=SIDEARG_${W_SIDEARGS}/GOLDAC_${GOLDACTIONS}
PARAMETERS_DIR3=AUXGPLOSS_${AUXGPLOSS}/QENTLOSS_${QENTLOSS}/ATTCOV_${ATTCOVLOSS}/PTREX_${PTREX}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/${PARAMETERS_DIR2}/${PARAMETERS_DIR3}_bool

# PREDICTION DATASET
PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
mkdir ${PREDICT_OUTPUT_DIR}

#*****************    PREDICTION FILENAME   *****************
PRED_FILENAME=devds_goldcontexts.txt

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
                                      ${INCLUDE_PACKAGE} \
                                      ${DEBUG} \
                                      ${BEAMSIZE}

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
