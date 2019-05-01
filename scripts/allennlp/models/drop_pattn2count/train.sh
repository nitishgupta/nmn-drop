#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

DATASET_NAME=num/yardscount_wqattn

DATASET_DIR=./resources/data/drop_s/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/passage_attn2count.jsonnet

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=8

export TYPE=gru
export ISIZE=4
export HSIZE=20
export NL=3
export SEED=10

export NORM=true
export NOISE=true

export EPOCHS=20

for SEED in 100
do
    for TYPE in gru
    do
        ####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
        CHECKPOINT_ROOT=./resources/semqa/checkpoints
        SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/date_num
        MODEL_DIR=drop_pattn2count
        PARAMETERS_DIR1=BS_${BS}/NORM_${NORM}/T_${TYPE}/I_${ISIZE}/H_${HSIZE}/NL_${NL}/NOISE_${NOISE}/S_${SEED}
        SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}

        SERIALIZATION_DIR=./resources/semqa/checkpoints/savedmodels/count_pretrn_nobias

        #######################################################################################################################

        allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR}

#        bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
#                                            ${INCLUDE_PACKAGE} \
#                                            ${SERIALIZATION_DIR} &

    done
done

#RESUME_SER_DIR=${SERIALIZATION_DIR}/Resume
#
#MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz
#
#bash scripts/allennlp/base/resume.sh ${CONFIGFILE} \
#                                     ${INCLUDE_PACKAGE} \
#                                     ${RESUME_SER_DIR} \
#                                     ${MODEL_TAR_GZ}