#!/usr/bin/env bash
#
#ROOT_DIR=./resources/data/hotpotqa
#RAW_JSON_DIR=${ROOT_DIR}/raw
#
#TRAIN_JSON=hotpot_train_v1.json
#DEVDS_JSON=hotpot_dev_distractor_v1.json
#SAMPLE_JSON=sample.json
#
#TRAIN_GO_JSON=train_goldonly.json
#DEVDS_GO_JSON=devds_goldonly.json
#SAMPLE_GO_JSON=sample_goldonly.json
#
#NUMPROC=15
#
#read -p "Split to process: (train / devds / sample): " SPLIT
#
#if [ "${SPLIT}" = "train" ]; then
#    INPUT_JSON=${TRAIN_JSON}
#    OUTPUT_JSON=${TRAIN_GO_JSON}
#    echo "SPLIT: ${SPLIT}"
#elif [ "${SPLIT}" = "devds" ]; then
#    INPUT_JSON=${DEVDS_JSON}
#    OUTPUT_JSON=${DEVDS_GO_JSON}
#    echo "SPLIT: ${SPLIT}"
#elif [ "${SPLIT}" = "sample" ]; then
#    INPUT_JSON=${SAMPLE_JSON}
#    OUTPUT_JSON=${SAMPLE_GO_JSON}
#    echo "SPLIT: ${SPLIT}"
#else
#    echo "UNRECOGNIZED SPLIT: ${SPLIT}"
#    exit 1
#fi

INPUT_JSON=$1
OUTPUT_JSON=$2
NUMPROC=$3

echo -e "\nRAW OUTPUT WITH GOLD CONTEXTS ONLY\n"
time python -m datasets.hotpotqa.preprocess.raw_with_goldcontexts --input_json ${INPUT_JSON} \
                                                                  --output_json ${OUTPUT_JSON} \
                                                                  --nump ${NUMPROC}