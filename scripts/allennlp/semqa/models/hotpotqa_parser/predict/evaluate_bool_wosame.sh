#!/usr/bin/env

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
VALFILE=${DATASET_DIR}/devds_goldcontexts.jsonl

#*****************    PREDICTION FILENAME   *****************
PRED_FILENAME=metrics_devds_goldcontexts.txt

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

export TOKENIDX="glove"
export W_SIDEARGS=false

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=4
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BOOL_QSTRQENT_FUNC='snli'
export QTK='encoded'
export CTK='modeled'
export GOLDACTIONS=false

export DA_NORMEMB=false
export DA_WT=true
export DA_NOPROJ=true

export BEAMSIZE=16

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints

CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa/bool_wosame
MODEL_DIR=hotpotqa_parser
PARAMETERS_DIR1=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/FUNC_${BOOL_QSTRQENT_FUNC}
PARAMETERS_DIR2=SIDEARG_${W_SIDEARGS}/GOLDAC_${GOLDACTIONS}/DA_NOPROJ_${DA_NOPROJ}/DA_WT_${DA_WT}/DA_NORMEMB_${DA_NORMEMB}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/${PARAMETERS_DIR2}

# PREDICTION DATASET
PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
mkdir ${PREDICT_OUTPUT_DIR}

TESTFILE=${VALFILE}
MODEL_TAR=${SERIALIZATION_DIR}/model.tar.gz
OUTPUT_FILE=${PREDICT_OUTPUT_DIR}/${PRED_FILENAME}

#######################################################################################################################

bash scripts/allennlp/base/evaluate.sh ${TESTFILE} \
                                       ${MODEL_TAR} \
                                       ${OUTPUT_FILE} \
                                       ${GPU} \
                                       ${INCLUDE_PACKAGE} \
                                       ${BEAMSIZE}

echo -e "Predictions file saved at: ${OUTPUT_FILE}"
