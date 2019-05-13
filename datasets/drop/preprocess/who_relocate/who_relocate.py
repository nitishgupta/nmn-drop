from typing import List, Dict, Tuple
import json
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse


WHO_RELOCATE_NGRAMS = ["which player scored", "who kicked the", "who threw the", "who scored the", "who caught the"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def prune_WhoRelocate(dataset):
    """ Extract all questions that contain any one of the 'Who' relocate-style questions based on the n-grams"""

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()

            if any(span in question_lower for span in WHO_RELOCATE_NGRAMS):
                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"After Pruning:{num_passages_after_prune} After pruning:{after_pruning_ques}")

    return new_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    print(f"\nTrain Output: {output_trnfp}")
    new_train_dataset = prune_WhoRelocate(train_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    print(f"\nDev Output: {output_trnfp}")
    new_dev_dataset = prune_WhoRelocate(dev_dataset)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("\nWritten augmented datasets")
