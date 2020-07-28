#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

### DATASET PATHS -- should be same across models for same dataset
DATASET_DIR=/shared/nitishg/data/drop-w-qdmr
DATASET_NAME=qdmr-filter-post-v6
# qdmr-filter-post-v6
# qdmr-v6_iclr600
# drop_iclr600
# drop_iclr_full

TRAINFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_train.json
VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_dev.json

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa

### TRAINING MODEL CONFIG -- should be same across datasets for the same model
CONFIGFILE=training_config/drop_parser_bert_biocount.jsonnet

export DATASET_READER="drop_reader_bert"

# Options: bert_joint_qp_encoder (default) ; bert_independent_qp_encoding
export QP_ENC="bert_joint_qp_encoder"

# Options: attn (default) ; decont ; decont_qp
export Q_REPR="attn"

# Check CONFIGFILE for environment variables to set
export GPU=0

export TRAINING_DATA_FILE=${TRAINFILE}
export VAL_DATA_FILE=${VALFILE}

# export BIO_TAG=true   --- this is now fixed at true for drop_parser_bert_biocount.jsonnet
export BIO_LABEL=IO

export COUNT_FIXED=false
export AUXLOSS=true

export EXCLOSS=true
export QATTLOSS=true
export MMLLOSS=true

export INTERPRET=false

# Whether strong supervison instances should be trained on first, if yes for how many epochs
export SUPFIRST=true
export SUPEPOCHS=0

# -1 will not run HardEM; HardEM will kick after EPOCH num of epochs
export HARDEM_EPOCH=0

export BS=4
export DROPOUT=0.2

export SEED=10

export BEAMSIZE=1
export MAX_DECODE_STEP=14
export EPOCHS=40

export GC_FREQ=500
export PROFILE_FREQ=0
export DEBUG=false

####    SERIALIZATION DIR --- Check for checkpoint_root/task/dataset/model/parameters/
CHECKPOINT_ROOT=./resources/checkpoints
SERIALIZATION_DIR_ROOT=${CHECKPOINT_ROOT}/drop-w-qdmr/${DATASET_NAME}
MODEL_DIR=drop_parser_bert
PD_1=Qattn_${QATTLOSS}/EXCLOSS_${EXCLOSS}/aux_${AUXLOSS}/${BIO_LABEL}_true/SUPEPOCHS_${SUPEPOCHS}_HEM_${HARDEM_EPOCH}_BM_${BEAMSIZE}
SERIALIZATION_DIR=${SERIALIZATION_DIR_ROOT}/${MODEL_DIR}/${PD_1}/S_${SEED}_biocnt-tied_v001_4000

# SERIALIZATION_DIR=./resources/checkpoints/test1

#######################################################################################################################

bash scripts/allennlp/train.sh ${CONFIGFILE} \
                               ${INCLUDE_PACKAGE} \
                               ${SERIALIZATION_DIR}