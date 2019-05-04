#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp
#
#### DATASET PATHS -- should be same across models for same dataset
#TRAINDATASET_NAME=date_num/dc_nc_100_yeardiff
#
#EVAL_DATASET=date/year_diff
#
#DATASET_DIR=./resources/data/drop_s/${EVAL_DATASET}
#TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
#VALFILE=${DATASET_DIR}/drop_dataset_dev.json
#
## PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
#INCLUDE_PACKAGE=semqa
#
## Check CONFIGFILE for environment variables to set
#export GPU=0
#
## All parameters here are used to fetch the correct serialization_dir
#export TOKENIDX="qanet"
#
#export BS=8
#export DROPOUT=0.2
#export LR=0.001
#
#export WEMB_DIM=100
#export RG=1e-4
#
## Which kind of similarity to use in Ques-Passage attention - raw / encoded / raw-enc
#export QP_SIM_KEY="raw"
#
#export GOLDACTIONS=false
#export GOLDPROGS=false
#export DENLOSS=true
#export EXCLOSS=true
#export QATTLOSS=true
#export MMLLOSS=true
#
## Whether strong supervison instances should be trained on first, if yes for how many epochs
#export SUPFIRST=true
#export SUPEPOCHS=5
#
#export SEED=100
#
#export BEAMSIZE=2
#
#export DEBUG=true
#
#####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
#CHECKPOINT_ROOT=./resources/semqa/checkpoints
#SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop_old/${TRAINDATASET_NAME}
#MODEL_DIR=drop_parser
#PD_1=BS_${BS}/LR_${LR}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/ED_${WEMB_DIM}/RG_${RG}/GACT_${GOLDACTIONS}/GPROGS_${GOLDPROGS}
#PD_2=QPSIMKEY_${QP_SIM_KEY}/QAL_${DENLOSS}/EXL_${EXCLOSS}/QATL_${QATTLOSS}/MML_${MMLLOSS}/SUPFIRST_${SUPFIRST}/SUPEPOCHS_${SUPEPOCHS}
#SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/${PD_2}/S_${SEED}/no_qsa
#
## PREDICTION DATASET
#PREDICT_OUTPUT_DIR=${SERIALIZATION_DIR}/predictions
#mkdir ${PREDICT_OUTPUT_DIR}
#
#mkdir -p ${PREDICT_OUTPUT_DIR}/${EVAL_DATASET}

##*****************    PREDICTION FILENAME   *****************
#PRED_FILENAME=${EVAL_DATASET}.dev_pred.txt
#EVAL_FILENAME=${EVAL_DATASET}.dev_eval.txt
#TESTFILE=${VALFILE}
##PRED_FILENAME=train_predictions.txt
##TESTFILE=${TRAINFILE}


# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/savedmodels/dateq_numcq_hmvy_ydiff_count
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

# EVALUATION DATASET
SUBFOLDER=date
EVAL_DATASET=datecomp_full
DATASET_DIR=./resources/data/drop_s/${SUBFOLDER}/${EVAL_DATASET}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

TESTFILE=${VALFILE}

PREDICTION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_train_pred.txt
EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_train_eval.txt
PREDICTOR=drop_parser_predictor

#######################################################################################################################


allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}}}" \
                 ${MODEL_TAR} ${TESTFILE}

allennlp evaluate --output-file ${EVALUATION_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TESTFILE}

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
echo -e "Evaluations file saved at: ${EVALUATION_FILE}"
