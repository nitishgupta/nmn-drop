#!/usr/bin/env bash

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

DATASET_DIR=/shared/nitishg/data/squad

TRAINFILE=${DATASET_DIR}/squad-train-v1.1_drop.json
VALFILE=${DATASET_DIR}/squad-dev-v1.1_drop.json

INCLUDE_PACKAGE=semqa
CONFIGFILE=training_config/qgen/squad_qgen.jsonnet

export GPU=0

export TRAIN_DATA=${TRAINFILE}
export VAL_DATA=${VALFILE}

export MASKQ=false

export BS=6
export BEAMSIZE=1

export EPOCHS=5

export SEED=42


CHECKPOINT_ROOT=./resources/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}
MODEL_DIR=squad-qgen
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/BS_${BS}/BEAM_${BEAMSIZE}/MASKQ_${MASKQ}/S_${SEED}_E5

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}
