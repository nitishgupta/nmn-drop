#!/usr/bin/env bash

ROOT_DIR=/srv/local/data/nitishg/data/hotpotqa
RAW_JSON_DIR=${ROOT_DIR}/raw
TOKENIZED_DIR=${ROOT_DIR}/tokenized
PROCESSED_DIR=${ROOT_DIR}/processed

TRAIN_JSON=hotpot_train_v1.json
DEVDS_JSON=hotpot_dev_distractor_v1.json
DEVFW_JSON=hotpot_dev_fullwiki_v1.json

TRAIN_JSONL=train.jsonl
DEVDS_JSONL=devds.jsonl
DEVFW_JSONL=devfw.jsonl

NUMPROC=15
F1THRESH=0.6


mkdir ${PROCESSED_DIR}

read -p "Split to preprocess: (train / devds / devfw): " SPLIT

if [ "${SPLIT}" = "train" ]; then
    INPUT_JSON=${TRAIN_JSON}
    INPUT_JSONL=${TRAIN_JSONL}
    echo "SPLIT: ${SPLIT}"
elif [ "${SPLIT}" = "devds" ]; then
    INPUT_JSON=${DEVDS_JSON}
    INPUT_JSONL=${DEVDS_JSONL}
    echo "SPLIT: ${SPLIT}"
elif [ "${SPLIT}" = "devfw" ]; then
    INPUT_JSON=${DEVFW_JSON}
    INPUT_JSONL=${DEVFW_JSONL}
    echo "SPLIT: ${SPLIT}"
else
    echo "UNRECOGNIZED SPLIT: ${SPLIT}"
    exit 1
fi

echo -e "\nTOKENIZATION AND NER\n"
# Tokenize and NER the raw files using spacy
time python -m datasets.hotpotqa.preprocess.tokenize_mp --input_json ${RAW_JSON_DIR}/${INPUT_JSON} \
                                                        --output_jsonl ${TOKENIZED_DIR}/${INPUT_JSONL} \
                                                        --nump ${NUMPROC}


echo -e "\nNORMALIZATION AND CLEANING OF ENTITY MENTIONS\n"
time python -m datasets.hotpotqa.preprocess.clean_ners_mp --input_json ${TOKENIZED_DIR}/${INPUT_JSONL} \
                                                          --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                          --nump ${NUMPROC}


echo -e "\nFLATTEN CONTEXTS INTO SINGLE STRING\n"
time python -m datasets.hotpotqa.preprocess.flatten_contexts --input_json ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                             --replace


echo -e "\nCROSS-DOC-COREF  and  GROUNDING CONTEXT AND QUESTION MENTIONS\n"
time python -m datasets.hotpotqa.preprocess.cdcr --input_json ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                 --replace

echo -e "\nANSWER GROUNDING and TYPING\n"
time python -m datasets.hotpotqa.preprocess.ans_grounding --input_json ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                          --replace --f1thresh ${F1THRESH}





