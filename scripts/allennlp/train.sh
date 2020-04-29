#!/usr/bin/env bash

CONFIGFILE=$1
INCLUDE_PACKAGE=$2
SERIALIZATION_DIR=$3

echo ""
echo "CONFIG FILE: ${CONFIGFILE}"
echo "SERIALIZATION_DIR_ROOT: ${SERIALIZATION_DIR}"
echo ""

read -p "Continue (Y/N) " continue
if ! ( [ "${continue}" = "y" ] || [ "${continue}" = "Y" ] ); then exit 1; else echo "Continuing ... "; fi

# Simple logic to make sure existing serialization dir is safely deleted
if [ -d "${SERIALIZATION_DIR}" ]; then
  echo "SERIALIZATION_DIR EXISTS: ${SERIALIZATION_DIR}"
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

allennlp train ${CONFIGFILE} --include-package ${INCLUDE_PACKAGE} -s ${SERIALIZATION_DIR}
