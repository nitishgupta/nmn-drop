#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0

# SAVED MODEL
MODEL_DIR=/shared/nitishg/checkpoints/qgen
PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/qgen_dand

FULL_VALFILE=/shared/nitishg/data/qgen_dand/pred.jsonl

VIS_PREDICTOR=question_generation
VIS_PREDICTION_FILE=${PREDICTION_DIR}/predictions.txt

allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"

# --overrides "{"model": { "beam_size": ${BEAMSIZE} }}" \
