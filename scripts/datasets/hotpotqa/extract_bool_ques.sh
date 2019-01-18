#!/usr/bin/env bash

ROOT_DIR=/srv/local/data/nitishg/data/hotpotqa
PROCESSED_DIR=${ROOT_DIR}/processed

TRAIN_JSONL=train.jsonl
DEVDS_JSONL=devds.jsonl
DEVFW_JSONL=devfw.jsonl

echo -e "Extracting bool questions from Training\n"
time python -m datasets.hotpotqa.analysis.extract_bool_ques --input_json ${PROCESSED_DIR}/${TRAIN_JSONL}

echo -e "Extracting bool questions from Dev-ds\n"
time python -m datasets.hotpotqa.analysis.extract_bool_ques --input_json ${PROCESSED_DIR}/${DEVDS_JSONL}


echo -e "Extracting bool questions from Training\n"
time python -m datasets.hotpotqa.analysis.extract_bool_ques --input_json ${PROCESSED_DIR}/${DEVFW_JSONL}
