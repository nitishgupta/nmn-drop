#!/usr/bin/env bash

INPUT_DIR=./resources/data/hotpotqa/processed/all

filenames=( 'train.jsonl' 'devds.jsonl' 'devfw.jsonl' 'train_goldcontexts.jsonl' 'devds_goldcontexts.jsonl')

OUTPUTDIR_BOOL=./resources/data/hotpotqa/processed/bool
OUTPUTDIR_BOOL_WSAME=./resources/data/hotpotqa/processed/bool_wsame
OUTPUTDIR_BOOL_WOSAME=./resources/data/hotpotqa/processed/bool_wosame

for INPUT_JSONL in "${filenames[@]}"
do
    if [ -f ${INPUT_DIR}/${INPUT_JSONL} ]; then
        echo -e "\nExtracting bool questions from ${INPUT_DIR}/${INPUT_JSONL} \n"
        time python -m datasets.hotpotqa.analysis.extract_bool_ques --input_json ${INPUT_DIR}/${INPUT_JSONL} \
                                                                    --bool_outdir ${OUTPUTDIR_BOOL} \
                                                                    --bool_wsame_outdir ${OUTPUTDIR_BOOL_WSAME} \
                                                                    --bool_wosame_outdir ${OUTPUTDIR_BOOL_WOSAME}

    else
        echo -e "\nInput file doesn't exist: ${INPUT_DIR}/${INPUT_JSONL} \n"
    fi
done