#!/usr/bin/env bash

INCLUDE_PACKAGE=semqa

# DATASET FILES
DATASET_DIR=/srv/local/data/nitishg/data/hotpotqa/processed
TRAINFILE=${DATASET_DIR}/train.jsonl

# VOCAB DATASET_READER CONFIG
CONFIGFILE=allenconfigs/semqa/vocab/vocab_tokens.jsonnet

#########    MODEL PARAMS  ######################
# Check CONFIGFILE for environment variables to set
export DATASET_READER="sample_hotpot"
export TOKEN_MIN_CNT=0
export TRAINING_DATA_FILE=${TRAINFILE}

# OUTPUT DIR
VOCABDIR=/srv/local/data/nitishg/semqa/vocabs/semqa/hotpotqa/sample_reader

#######################################################################################################################
# Code below this shouldn't require changing for a reader
if [ -d "${VOCABDIR}/vocabulary" ]; then
    echo "Vocabulary Dir already exists: ${VOCABDIR}/vocabulary"
    read -p "Delete (Y/N) " delete

    if [ "${delete}" = "y" ] || [ "${delete}" = "Y" ]; then
        echo "Deleting ${SERIALIZATION_DIR}"
        rm -r ${SERIALIZATION_DIR}
    else
        echo "Not deleting ${SERIALIZATION_DIR}"
        echo "Cannot continue with non-empty serialization dir. Exiting"
        exit 1
    fi
fi

allennlp make-vocab ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${VOCABDIR}

# This is to copy the evaluated jsonnet in the vocab dir
# This should be same across vocabscripts
VOCABCONFIG_JSON=${VOCABDIR}/vocabconfig.json
echo "Copying evaluated Jsonnet config in ${VOCABCONFIG_JSON}"
python utils/evaluate_jsonnet.py ${CONFIGFILE} ${VOCABCONFIG_JSON}