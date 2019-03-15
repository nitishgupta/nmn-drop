#!/usr/bin/env bash

ROOT_DIR=./resources/data/hotpotqa
RAW_JSON_DIR=${ROOT_DIR}/raw


TOKENIZED_DIR=${ROOT_DIR}/processed/all_tokenized
PROCESSED_DIR=${ROOT_DIR}/processed/all

RAW_TRAIN_JSON=hotpot_train_v1.json
RAW_DEVDS_JSON=hotpot_dev_distractor_v1.json
RAW_DEVFW_JSON=hotpot_dev_fullwiki_v1.json
RAW_TRAIN_GOLDONLY_JSON=train_goldcontexts.json
RAW_DEVDS_GOLDONLY_JSON=devds_goldcontexts.json


TRAIN_JSONL=train.jsonl
DEVDS_JSONL=devds.jsonl
DEVFW_JSONL=devfw.jsonl
TRAIN_GOLDONLY_JSONL=train_goldcontexts.jsonl
DEVDS_GOLDONLY_JSONL=devds_goldcontexts.jsonl

NUMPROC=15

F1THRESH=0.6

TASK=$1
SPLIT=$2


if [ "${TASK}" = "goldcontexts" ]; then
    if [ "${SPLIT}" = "train" ]; then
        JSON_FILE=${RAW_JSON_DIR}/${RAW_TRAIN_JSON}
        OUTPUT_JSON=${RAW_JSON_DIR}/${RAW_TRAIN_GOLDONLY_JSON}
    elif [ "${SPLIT}" = "devds" ]; then
        JSON_FILE=${RAW_JSON_DIR}/${RAW_DEVDS_JSON}
        OUTPUT_JSON=${RAW_JSON_DIR}/${RAW_DEVDS_GOLDONLY_JSON}
    else
        echo "UNRECOGNIZED SPLIT - ${SPLIT} FOR TASK ${TASK}"
        exit 1
    fi
    bash ./scripts/datasets/hotpotqa/raw_gold_contexts.sh ${JSON_FILE} ${OUTPUT_JSON} ${NUMPROC}
    printf "\nOUTPUT: ${OUTPUT_JSON}\n\n";
fi

if [ "${SPLIT}" = "train" ]; then
    JSON_FILE=${RAW_TRAIN_JSON}
    JSONL_FILE=${TRAIN_JSONL}
elif [ "${SPLIT}" = "devds" ]; then
    JSON_FILE=${RAW_DEVDS_JSON}
    JSONL_FILE=${DEVDS_JSONL}
elif [ "${SPLIT}" = "devfw" ]; then
    JSON_FILE=${RAW_DEVFW_JSON}
    JSONL_FILE=${DEVFW_JSONL}
elif [ "${SPLIT}" = "train_goldcontexts" ]; then
    JSON_FILE=${RAW_TRAIN_GOLDONLY_JSON}
    JSONL_FILE=${TRAIN_GOLDONLY_JSONL}
elif [ "${SPLIT}" = "devds_goldcontexts" ]; then
    JSON_FILE=${RAW_DEVDS_GOLDONLY_JSON}
    JSONL_FILE=${DEVDS_GOLDONLY_JSONL}
else
    echo "UNRECOGNIZED SPLIT - ${SPLIT} FOR TASK ${TASK}"
    exit 1
fi


if [ "${TASK}" = "preprocess" ]; then
    echo -e "Preprocessing raw json to jsonl"
    echo -e "Input: ${JSON_FILE}"
    echo -e "Output: ${JSONL_FILE}\n"
    bash ./scripts/datasets/hotpotqa/preprocess.sh ${RAW_JSON_DIR} ${JSON_FILE} ${TOKENIZED_DIR} ${PROCESSED_DIR} \
                                                   ${JSONL_FILE} ${F1THRESH} ${NUMPROC};
    echo "OUTPUT: ${JSONL_FILE}"

elif [ "${TASK}" = "extract_bool" ]; then
    echo -e "Extracting bool questions from processed jsonl"
    echo -e "Input: ${PROCESSED_DIR}/${JSONL_FILE}"
    bash ./scripts/datasets/hotpotqa/extract_bool_ques.sh ${PROCESSED_DIR}/${JSONL_FILE}

elif [ "${TASK}" = "extract_either" ]; then
    echo -e "Extracting bool questions from processed jsonl"
    echo -e "Input: ${PROCESSED_DIR}/${JSONL_FILE}"
    bash ./scripts/datasets/hotpotqa/extract_eitheror_ques.sh ${PROCESSED_DIR}/${JSONL_FILE}

else
    echo "UNRECOGNIZED TASK ${TASK}"
    exit 1
fi


#elif [ "${SPLIT}" = "devds" ]; then
#    INPUT_JSON=${DEVDS_JSON}
#    INPUT_JSONL=${DEVDS_JSONL}
#    echo "SPLIT: ${SPLIT}"
#elif [ "${SPLIT}" = "devfw" ]; then
#    INPUT_JSON=${DEVFW_JSON}
#    INPUT_JSONL=${DEVFW_JSONL}
#    echo "SPLIT: ${SPLIT}"
#elif [ "${SPLIT}" = "sample" ]; then
#    INPUT_JSON=${SAMPLE_JSON}
#    INPUT_JSONL=${SAMPLE_JSONL}
#    echo "SPLIT: ${SPLIT}"
#else
#    echo "UNRECOGNIZED SPLIT: ${SPLIT}"
#    exit 1
#fi
