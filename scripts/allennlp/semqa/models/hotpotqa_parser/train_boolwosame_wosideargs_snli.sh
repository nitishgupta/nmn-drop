#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=./resources/data/hotpotqa/processed/bool_wosame
TRAINFILE=${DATASET_DIR}/train_goldcontexts.jsonl
VALFILE=${DATASET_DIR}/devds_goldcontexts.jsonl

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

# NO SIDEARGS -- Model with discrete actions
export W_SIDEARGS=false

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/hotpotqa_parser_wosideargs_snli.jsonnet
export TOKENIDX="glove"
export VOCABDIR=./resources/semqa/vocabs/hotpotqa/bool_wosame/gold_contexts/wosideargs_${W_SIDEARGS}/tokens_${TOKENIDX}/vocabulary

export DATASET_READER=hotpotqa_reader

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export WORD_EMBED_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE='./resources/embeddings/glove/glove.6B.100d.txt.gz'
export BOOL_QSTRQENT_FUNC='snli'
export QTK='encoded'
export CTK='modeled'
export GOLDACTIONS=false

export DA_NORMEMB=false
export DA_WT=true
export DA_NOPROJ=true

export BS=4
export LR=0.001
export OPT=adam
export DROPOUT=0.2

export BEAMSIZE=1
export MAX_DECODE_STEP=12
export EPOCHS=35

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/hotpotqa/bool_wosame
MODEL_DIR=hotpotqa_parser
PARAMETERS_DIR1=BS_${BS}/OPT_${OPT}/LR_${LR}/Drop_${DROPOUT}/TOKENS_${TOKENIDX}/FUNC_${BOOL_QSTRQENT_FUNC}
PARAMETERS_DIR2=SIDEARG_${W_SIDEARGS}/GOLDAC_${GOLDACTIONS}/DA_NOPROJ_${DA_NOPROJ}/DA_WT_${DA_WT}/DA_NORMEMB_${DA_NORMEMB}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PARAMETERS_DIR1}/${PARAMETERS_DIR2}_test

#######################################################################################################################

bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
                                    ${INCLUDE_PACKAGE} \
                                    ${SERIALIZATION_DIR}