#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_NAME=date_num/date_numcq_hmvy_cnt_filter
# DATASET_NAME=num/hmyw_filter

DATASET_DIR=./resources/data/drop_s/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
# CONFIGFILE=allenconfigs/semqa/train/drop_parser_wmodel.jsonnet
CONFIGFILE=allenconfigs/semqa/train/drop_parser_wmodel.jsonnet

export TOKENIDX="qanet"

export DATASET_READER=drop_reader

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export WEMB_DIM=100
# export WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.lower.converted.zip"
export WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

export MODELTYPE=modeled
export COUNT_FIXED=false

export DENLOSS=true
export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=5

export BS=8
export DROPOUT=0.2

export LR=0.001
export RG=1e-07

export SEED=100

export BEAMSIZE=4
export MAX_DECODE_STEP=14
export EPOCHS=60

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_parser
PD_1=TOKENS_${TOKENIDX}/ED_${WEMB_DIM}/RG_${RG}/MODELTYPE_${MODELTYPE}/CNTFIX_${COUNT_FIXED}
PD_2=SUPEPOCHS_${SUPEPOCHS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/${PD_2}/S_${SEED}/NewMinMax

# SERIALIZATION_DIR=./resources/semqa/checkpoints/test/hmywcount_mod_sgfilter_filterlater5_cntfix
# SERIALIZATION_DIR=./resources/semqa/checkpoints/test/test

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
