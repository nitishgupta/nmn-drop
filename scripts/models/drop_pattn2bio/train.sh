#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

DATASET_DIR=/shared/nitishg/data/drop-w-qdmr/preprocess
# qdmr-filter-post-v6
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

export TRAIN_FILE=${TRAINFILE}
export VAL_FILE=${VALFILE}

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=training_config/passage_attn2bio.jsonnet

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=16

export TYPE=gru
export ISIZE=4
export HSIZE=20
export LAYERS=3

export SEED=1

export EPOCHS=50


CHECKPOINT_ROOT=/shared/nitishg/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}
MODEL_DIR=drop_pattn2bio
PARAMETERS_DIR1=T_${TYPE}/Isize_${ISIZE}/Hsize_${HSIZE}/Layers_${LAYERS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/S_${SEED}

##############

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}


#for LAYERS in 2 3 4 5
#do
#    for TYPE in gru
#    do
#        ####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
#        CHECKPOINT_ROOT=/shared/nitishg/checkpoints
#        SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}
#        MODEL_DIR=drop_pattn2bio
#        PARAMETERS_DIR1=T_${TYPE}/Isize_${ISIZE}/Hsize_${HSIZE}/Layers_${LAYERS}
#        SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/S_${SEED}
#
#        #######################################################################################################################
#
#        allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR} &
#
#    done
#done