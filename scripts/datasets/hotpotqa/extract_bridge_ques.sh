#!/usr/bin/env bash

INPUT_DIR=./resources/data/hotpotqa/processed

filenames=( 'train.jsonl' 'devds.jsonl' 'devfw.jsonl' 'train_goldcontexts.jsonl' 'devds_goldcontexts.jsonl' )

OUTPUTDIR_BRIDGEEN=./resources/data/hotpotqa/processed/bridge_en

for INPUT_JSONL in "${filenames[@]}"
do
    if [ -f ${INPUT_DIR}/${INPUT_JSONL} ]; then
        echo -e "\nExtracting brideg entity questions from ${INPUT_DIR}/${INPUT_JSONL} \n"
        time python -m datasets.hotpotqa.analysis.extract_bridgeen_ques --input_json ${INPUT_DIR}/${INPUT_JSONL} \
                                                                        --output_dir ${OUTPUTDIR_BRIDGEEN}

    else
        echo -e "\nInput file doesn't exist: ${INPUT_DIR}/${INPUT_JSONL} \n"
    fi
done