#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
# DATASET_NAME=num/synthetic_count_num
DATASET_NAME=date_num/dc_nc_yd_num_syn

DATASET_DIR=./resources/data/drop_s/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_parser.jsonnet

export TOKENIDX="qanet"

export DATASET_READER=drop_reader

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export WEMB_DIM=300
export WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.lower.converted.zip"
# export WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

# Which kind of similarity to use in Ques-Passage attention - raw / encoded / raw-enc
export QP_SIM_KEY="enc"
export SIM_KEY="ma"

export GOLDACTIONS=false
export GOLDPROGS=false
export DENLOSS=true
export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=5

# export PTREX=false
# export PTRWTS="./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/PTREX_false/best.th"

export BS=8
export DROPOUT=0.2

export LR=0.001
export RG=1e-07

export SEED=100

export BEAMSIZE=1
export MAX_DECODE_STEP=14
export EPOCHS=60

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_parser_new
PD_1=TOKENS_${TOKENIDX}/ED_${WEMB_DIM}/RG_${RG}
PD_2=QPSIMKEY_${QP_SIM_KEY}/SIM_KEY_${SIM_KEY}/SUPEPOCHS_${SUPEPOCHS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/${PD_2}/S_${SEED}_c3

# SERIALIZATION_DIR=./resources/semqa/checkpoints/test

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}
