#!/usr/bin/env bash

CONFIGFILE=$1
INCLUDE_PACKAGE=$2
VOCABDIR=$3

if [ -d "${VOCABDIR}/vocabulary" ]; then
    echo "Vocabulary Dir already exists: ${VOCABDIR}/vocabulary"
    read -p "Delete (Y/N) " delete

    if [ "${delete}" = "y" ] || [ "${delete}" = "Y" ]; then
        echo "Deleting ${VOCABDIR}/vocabulary"
        rm -r ${VOCABDIR}/vocabulary
    else
        echo "Not deleting ${VOCABDIR}"
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