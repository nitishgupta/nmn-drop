#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

# All parameters here are used to fetch the correct serialization_dir
export BS=8
export DROPOUT=0.2

DATASET_DIR=./resources/data/drop_s/synthetic/pattn2count
TRAINFILE=${DATASET_DIR}/train.json
VALFILE=${DATASET_DIR}/dev.json

TEST_DATA=${VALFILE}

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
MODEL_DIR=./resources/semqa/checkpoints/drop_pattn2count/T_gru/Isize_4/Hsize_20/Layers_2/S_100/t600_v600
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}
PREDICTION_FILE=${PREDICTION_DIR}/preds.txt
EVALUATION_FILE=${PREDICTION_DIR}/eval.txt

#######################################################################################################################

allennlp evaluate --output-file ${EVALUATION_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TEST_DATA}



allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor pattn2count_predictor \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 --use-dataset-reader \
                 --overrides "{"dataset_reader": { "samples_per_bucket_count": 5}}" \
                 ${MODEL_TAR} ${TEST_DATA}
#

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
