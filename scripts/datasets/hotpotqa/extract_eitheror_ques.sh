#!/usr/bin/env bash

# Output file has the same name as the input file in the output_dir.
# Make sure output dir is not the same as the input dir.

INPUT_DIR=./resources/data/hotpotqa/processed/bool_wosame
filenames=( 'train.jsonl' 'devds.jsonl' 'devfw.jsonl' 'train_goldcontexts.jsonl' 'devds_goldcontexts.jsonl' 'qnitish.jsonl')

OUTPUT_DIR=./resources/data/hotpotqa/processed/bool_wosame_eitheror

for INPUT_JSONL in "${filenames[@]}"
do
    if [ -f ${INPUT_DIR}/${INPUT_JSONL} ]; then
        echo -e "\nExtracting bool questions from ${INPUT_DIR}/${INPUT_JSONL} \n"
        time python -m datasets.hotpotqa.analysis.extract_either_ques --input_json ${INPUT_DIR}/${INPUT_JSONL} \
                                                                      --output_dir ${OUTPUT_DIR}
    else
        echo -e "\nInput file doesn't exist: ${INPUT_DIR}/${INPUT_JSONL} \n"
    fi
done