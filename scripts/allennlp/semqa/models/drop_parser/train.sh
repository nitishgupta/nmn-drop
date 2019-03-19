#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/drop/date_num_subset
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_parser.jsonnet
export TOKENIDX="glove"
export VOCABDIR=./resources/semqa/vocabs/drop/date_num/tokens_${TOKENIDX}/vocabulary

export DATASET_READER=drop_reader

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export WORD_EMBED_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

export GOLDACTIONS=false
export AUXGPLOSS=false
export ATTCOVLOSS=false

export PTREX=false
# export PTRWTS="./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/PTREX_false/best.th"

export BS=4
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BEAMSIZE=4
export MAX_DECODE_STEP=12
export EPOCHS=30

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/date_num
MODEL_DIR=drop_parser
PARAMETERS_DIR1=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/GOLDAC_${GOLDACTIONS}
PARAMETERS_DIR2=AUXGPLOSS_${AUXGPLOSS}/ATTCOV_${ATTCOVLOSS}/PTREX_${PTREX}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/${PARAMETERS_DIR2}

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}


#RESUME_SER_DIR=${SERIALIZATION_DIR}/Resume
#
#MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz
#
#bash scripts/allennlp/base/resume.sh ${CONFIGFILE} \
#                                     ${INCLUDE_PACKAGE} \
#                                     ${RESUME_SER_DIR} \
#                                     ${MODEL_TAR_GZ}