#!/usr/bin/env bash
#
#ROOT_DIR=/srv/local/data/nitishg/data/hotpotqa
#RAW_JSON_DIR=${ROOT_DIR}/raw
#TOKENIZED_DIR=${ROOT_DIR}/tokenized
#PROCESSED_DIR=${ROOT_DIR}/processed
#PROCESSED_GOLDONLY_DIR=${ROOT_DIR}/processed/goldcontexts
#
#TRAIN_JSON=hotpot_train_v1.json
#DEVDS_JSON=hotpot_dev_distractor_v1.json
#DEVFW_JSON=hotpot_dev_fullwiki_v1.json
#SAMPLE_JSON=sample.json
#TRAIN_GOLDONLY_JSON=train_goldonly.json
#DEVDS_GOLDONLY_JSON=devds_goldonly.json
#
#TRAIN_JSONL=train.jsonl
#DEVDS_JSONL=devds.jsonl
#DEVFW_JSONL=devfw.jsonl
#SAMPLE_JSONL=sample.jsonl
#
#NUMPROC=15
#F1THRESH=0.6
#
#mkdir ${TOKENIZED_DIR}
#mkdir ${PROCESSED_DIR}
#
#read -p "Split to preprocess: (train / devds / devfw): " SPLIT
#
#if [ "${SPLIT}" = "train" ]; then
#    INPUT_JSON=${TRAIN_JSON}
#    INPUT_JSONL=${TRAIN_JSONL}
#    echo "SPLIT: ${SPLIT}"
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

RAW_JSON_DIR=$1
INPUT_JSON=$2
TOKENIZED_DIR=$3
PROCESSED_DIR=$4
INPUT_JSONL=$5
F1THRESH=$6
NUMPROC=$7


echo -e "\nTOKENIZATION AND NER\n"
time python -m datasets.hotpotqa.preprocess.tokenize_mp --input_json ${RAW_JSON_DIR}/${INPUT_JSON} \
                                                        --output_jsonl ${TOKENIZED_DIR}/${INPUT_JSONL} \
                                                        --nump ${NUMPROC}


echo -e "\nNORMALIZATION AND CLEANING OF ENTITY MENTIONS\n"
time python -m datasets.hotpotqa.preprocess.clean_ners_mp --input_json ${TOKENIZED_DIR}/${INPUT_JSONL} \
                                                          --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                          --nump ${NUMPROC}

echo -e "\nFLATTEN CONTEXTS INTO SINGLE STRING\n"
time python -m datasets.hotpotqa.preprocess.flatten_contexts --input_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                             --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL}.tmp

mv ${PROCESSED_DIR}/${INPUT_JSONL}.tmp ${PROCESSED_DIR}/${INPUT_JSONL}

echo -e "\nCROSS-DOC-COREF  and  GROUNDING CONTEXT AND QUESTION MENTIONS\n"
time python -m datasets.hotpotqa.preprocess.cdcr --input_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                 --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL}.tmp
mv ${PROCESSED_DIR}/${INPUT_JSONL}.tmp ${PROCESSED_DIR}/${INPUT_JSONL}

echo -e "\nANSWER GROUNDING and TYPING\n"
time python -m datasets.hotpotqa.preprocess.ans_grounding --input_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                          --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL}.tmp \
                                                          --f1thresh ${F1THRESH}

mv ${PROCESSED_DIR}/${INPUT_JSONL}.tmp ${PROCESSED_DIR}/${INPUT_JSONL}

echo -e "\nREPLACING BRACKETS\n"
time python -m datasets.hotpotqa.preprocess.replace_brackets --input_jsonl ${PROCESSED_DIR}/${INPUT_JSONL} \
                                                             --output_jsonl ${PROCESSED_DIR}/${INPUT_JSONL}.tmp
mv ${PROCESSED_DIR}/${INPUT_JSONL}.tmp ${PROCESSED_DIR}/${INPUT_JSONL}