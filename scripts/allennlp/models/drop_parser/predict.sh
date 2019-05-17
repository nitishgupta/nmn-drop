#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/num/hmyw_relprog/drop_parser/TOKENS_qanet/ED_100/RG_1e-07/MODELTYPE_encoded/CNTFIX_false/aux_true/SUPEPOCHS_0/S_10/NewModel
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

# EVALUATION DATASET
SUBFOLDER=num

for EVAL_DATASET in who_relocate
# for EVAL_DATASET in numcomp_full count_filterqattn hmyw_filter who_relocate
# for EVAL_DATASET in datecomp_full year_diff
do
    DATASET_DIR=./resources/data/drop_s/${SUBFOLDER}/${EVAL_DATASET}
    TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
    VALFILE=${DATASET_DIR}/drop_dataset_dev.json

    TESTFILE=${VALFILE}

    PREDICTION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_pred.txt
    EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_eval.txt
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
done
