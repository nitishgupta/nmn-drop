#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/squad-qgen/BS_6/BEAM_1/MASKQ_false/S_42
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

FULL_VALFILE=/shared/nitishg/data/squad/squad-dev-v1.1_drop.json

VIS_PREDICTOR=question_generation
VIS_PREDICTION_FILE=${PREDICTION_DIR}/predictions.jsonl

allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --use-dataset-reader \
                 --silent \
                 --batch-size 4 \
                 ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
