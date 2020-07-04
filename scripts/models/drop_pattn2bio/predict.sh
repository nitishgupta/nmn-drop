#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# Check CONFIGFILE for environment variables to set
export GPU=0

# All parameters here are used to fetch the correct serialization_dir
export BS=8
export DROPOUT=0.2

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr/preprocess
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

TEST_DATA=${VALFILE}

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
MODEL_DIR=/shared/nitishg/checkpoints/drop_pattn2bio/T_gru/Isize_4/Hsize_20/Layers_3/S_1_small_v0.1
MODEL_TAR=${MODEL_DIR}/model.tar.gz

PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

PREDICTION_FILE=${PREDICTION_DIR}/preds1.txt
EVALUATION_FILE=${PREDICTION_DIR}/eval1.txt

#######################################################################################################################


allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor pattn2bio_predictor \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 1 \
                 --use-dataset-reader \
                 ${MODEL_TAR} ${TEST_DATA}

allennlp evaluate --output-file ${EVALUATION_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  ${MODEL_TAR} ${TEST_DATA}

echo -e "Predictions file saved at: ${PREDICTION_FILE}"
echo -e "Eval metrics file saved at: ${EVALUATION_FILE}"
