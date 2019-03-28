#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/passage_attn2span.jsonnet

# Check CONFIGFILE for environment variables to set
export GPU=0

export BS=8
export NORM=true
export AVAL=1.0

export TYPE=gru
export ISIZE=4
export HSIZE=10
export NL=2
export SEED=10

export NOISE=true
export SCALING=true

export EPOCHS=20

for SEED in 10 100 1000
do
    for TYPE in gru lstm
    do
        ####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
        CHECKPOINT_ROOT=./resources/semqa/checkpoints
        SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/date_num
        MODEL_DIR=drop_pattn2span
        PARAMETERS_DIR1=BS_${BS}/NORM_${NORM}/AVAL_${AVAL}/T_${TYPE}/I_${ISIZE}/H_${HSIZE}/NL_${NL}/NOISE_${NOISE}/SC_${SCALING}/SEED_${SEED}
        SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}

        #######################################################################################################################

        allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR} &

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