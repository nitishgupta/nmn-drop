#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true

# SAVED MODEL
MODEL_DIR=./resources/semqa/checkpoints/drop/date_num/date_ydNEW_num_hmyw_cnt_rel_600/drop_parser_bert/CNTFIX_false/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5/S_1000/BertModel_wTest
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

# EVALUATION DATASET
# SUBFOLDER=alldatasets
SUBFOLDER=date_num

# for EVAL_DATASET in datecomp_full year_diff count_filterqattn hmyw_filter relocate_wprog numcomp_full
for EVAL_DATASET in date_ydNEW_num_hmyw_cnt_rel_600
do
    DATASET_DIR=./resources/data/drop/${SUBFOLDER}/${EVAL_DATASET}
    VALFILE=${DATASET_DIR}/drop_dataset_mydev.json
    TESTFILE=${DATASET_DIR}/drop_dataset_mytest.json

    VAL_EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_dev_eval.txt
    TEST_EVALUATION_FILE=${PREDICTION_DIR}/${EVAL_DATASET}_test_eval.txt

    ###################################################################################################################

    allennlp evaluate --output-file ${VAL_EVALUATION_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${VALFILE}

    allennlp evaluate --output-file ${TEST_EVALUATION_FILE} \
                      --cuda-device ${GPU} \
                      --include-package ${INCLUDE_PACKAGE} \
                      ${MODEL_TAR} ${TESTFILE}



    echo -e "Val Evaluations file saved at: ${VAL_EVALUATION_FILE}"
    echo -e "Test Evaluations file saved at: ${TEST_EVALUATION_FILE}"
done
