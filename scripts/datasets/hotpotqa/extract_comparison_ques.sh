#!/usr/bin/env bash

INPUT_DIR=./resources/data/hotpotqa/processed

filenames=( 'train.jsonl' 'devds.jsonl' 'devfw.jsonl' 'train_goldcontexts.jsonl' 'devds_goldcontexts.jsonl' )

OUTPUT_DIR=./resources/data/hotpotqa/processed/comparison_entity

for INPUT_JSONL in "${filenames[@]}"
do
    if [ -f ${INPUT_DIR}/${INPUT_JSONL} ]; then
        echo -e "\nExtracting comparison questions from ${INPUT_DIR}/${INPUT_JSONL} \n"
        time python -m datasets.hotpotqa.analysis.extract_comparison_ques --input_json ${INPUT_DIR}/${INPUT_JSONL} \
                                                                          --output_dir ${OUTPUT_DIR}

    else
        echo -e "\nInput file doesn't exist: ${INPUT_DIR}/${INPUT_JSONL} \n"
    fi
done