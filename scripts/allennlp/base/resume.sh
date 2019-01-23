#!/usr/bin/env bash

CONFIGFILE=$1
INCLUDE_PACKAGE=$2
SERIALIZATION_DIR=$3
MODEL_TAR_GZ=$4

echo ""
echo "CONFIG FILE: ${CONFIGFILE}"
echo "SERIALIZATION_DIR_ROOT: ${SERIALIZATION_DIR}"
echo ""

read -p "Continue (Y/N) " continue
if [ "${continue}" != "Y" ]; then exit 1; else echo "Continuing ... "; fi

allennlp fine-tune -c ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR} -m ${MODEL_TAR_GZ}
