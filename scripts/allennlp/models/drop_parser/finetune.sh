#!/usr/bin/env

export TMPDIR=/srv/local/data/nitishg/tmp

### DATASET PATHS -- should be same across models for same dataset
# DATASET_NAME=num/longest_shortest_yards
DATASET_NAME=date_num/date_numcq_hmvy_cnt_filter_no_exec

DATASET_DIR=./resources/data/drop_s/${DATASET_NAME}
TRAINFILE=${DATASET_DIR}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=allenconfigs/semqa/train/drop_parser_wmodel.jsonnet
export TOKENIDX="qanet"

export DATASET_READER=drop_reader

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

export WEMB_DIM=100
# export WORDEMB_FILE="./resources/embeddings/glove.840B.300d.lower.converted.zip"
export WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

export BIDAF_MODEL_TAR='https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz'
export BIDAF_WORDEMB_FILE="https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"

export MODELTYPE=encoded
export COUNT_FIXED=false
export AUXLOSS=false

export DENLOSS=true
export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=0

# export PTREX=false
# export PTRWTS="./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/PTREX_false/best.th"

export BS=8
export DROPOUT=0.2

export LR=0.001
export RG=1e-4

export SEED=100

export BEAMSIZE=4
export MAX_DECODE_STEP=14
export EPOCHS=20

export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/semqa/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop/${DATASET_NAME}
MODEL_DIR=drop_parser
PD_1=TOKENS_${TOKENIDX}/ED_${WEMB_DIM}/RG_${RG}/MODELTYPE_${MODELTYPE}/CNTFIX_${COUNT_FIXED}
PD_2=SUPEPOCHS_${SUPEPOCHS}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/${PD_2}/S_${SEED}/NewMinMax_aux_${AUXLOSS}

#######################################################################################################################

#bash scripts/allennlp/base/train.sh ${CONFIGFILE} \
#                                    ${INCLUDE_PACKAGE} \
#                                    ${SERIALIZATION_DIR}

RESUME_SER_DIR=./resources/semqa/checkpoints/drop/date_num/date_numcq_hmvy_cnt_filter_no_exec/drop_parser/TOKENS_qanet/ED_100/RG_1e-07/MODELTYPE_encoded/CNTFIX_false/SUPEPOCHS_5/S_100/NewMinMax_aux_false/Resume
MODEL_TAR_GZ=./resources/semqa/checkpoints/drop/date_num/date_numcq_hmvy_cnt_filter_no_exec/drop_parser/TOKENS_qanet/ED_100/RG_1e-07/MODELTYPE_encoded/CNTFIX_false/SUPEPOCHS_5/S_100/NewMinMax_aux_false/model.tar.gz

allennlp fine-tune --extend-vocab -c ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${RESUME_SER_DIR} -m ${MODEL_TAR_GZ}