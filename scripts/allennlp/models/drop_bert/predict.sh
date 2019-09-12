#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/date/datefull_yd/drop_parser_bert/CNTFIX_false/aux_true/SUPEPOCHS_5/S_100/BertModel15RelAux15_2x
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

# EVALUATION DATASET
SUBFOLDER=alldatasets

# for EVAL_DATASET in datecomp_full year_diff count_filterqattn hmyw_filter relocate_wprog numcomp_full
for EVAL_DATASET in year_diff datecomp_full
do
    DATASET_DIR=./resources/data/drop/${SUBFOLDER}/${EVAL_DATASET}
    TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
    VALFILE=${DATASET_DIR}/drop_dataset_dev.json

    TESTFILE=${VALFILE}
	
    ANALYSIS_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_analysis.tsv	
    PREDICTION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_pred.txt
    EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_eval.txt
    PREDICTOR=drop_parser_predictor
    # PREDICTOR=drop_analysis_predictor

    ###################################################################################################################



    # allennlp predict --output-file ${ANALYSIS_FILE} \
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
