import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
import random
from datasets.drop import constants

random.seed(100)

""" 
    Remove Execution Supervised key from each question in the Train data (only)
"""


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def removeExecutionSupervision(dataset):
    """ Given a dataset, remove execution supervision from all questions. """

    total_num_passages = len(dataset)

    supervision_dict = defaultdict(int)
    exec_supervision_keys = [constants.exection_supervised, constants.strongly_supervised]
    supervision_keys = [constants.program_supervised, constants.qattn_supervised, constants.exection_supervised,
                        constants.strongly_supervised]

    total_num_qa = 0

    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        for qa in passage_info[constants.qa_pairs]:
            for key in exec_supervision_keys:
                qa[key] = False

            for key in supervision_keys:
                if key in qa:
                    supervision_dict[key] += 1 if qa[key] is True else 0


    print()
    print(f"TotalNumPassages: {total_num_passages}")
    print(f"Num of original question: {total_num_qa}")
    print(f"Supervision Dict: {supervision_dict}")

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    print("Training questions .... ")
    print(output_dir)
    new_train_dataset = removeExecutionSupervision(train_dataset)
    # new_dev_dataset = removeDateCompPassageWeakAnnotations(train_dataset, 0)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(dev_dataset, f, indent=4)



