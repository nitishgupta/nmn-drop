import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
import random

random.seed(100)

from datasets.drop import constants


FILES_TO_MERGE = ['drop_dataset_train.json', 'drop_dataset_dev.json']


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset



def make_supervision_dict(dataset, supervision_keys):
    total_num_qa = 0
    supervision_dict = defaultdict(int)
    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        for qa in passage_info[constants.qa_pairs]:
            for key in supervision_keys:
                if key in qa:
                    supervision_dict[key] += 1 if qa[key] else 0

    return supervision_dict


def removeDateCompPassageWeakAnnotations(dataset, annotation_for_numpassages):
    """ Given a dataset containing date-comparison questions that are heuristically strongly annotated
        and the number of passages that need to remain strongly annotated, we remove the strong annotations for other
        passages. These annotations include: question-attention, event-date-groundings, etc.
        Fields from which annotation is removed:
        1. constants.datecomp_ques_event_date_groundings, constants.datecomp_ques_event_date_values
        2. constants.datecomp_ques_event_attentions
        3. constants.strongly_annotated - is set to False for all questions

    """

    total_num_passages = len(dataset)
    if annotation_for_numpassages == -1:
        annotation_for_numpassages = total_num_passages

    passage_idxs = list(dataset.keys())
    random.shuffle(passage_idxs)
    choosen_passage_idxs = passage_idxs[0:annotation_for_numpassages]

    total_num_qa = 0

    supervision_keys = [constants.program_supervised, constants.qattn_supervised, constants.exection_supervised,
                        constants.strongly_supervised]

    orig_supervision_dict = make_supervision_dict(dataset, supervision_keys)

    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        if passage_idx not in choosen_passage_idxs:
            # Removing the strong annotations for all QAs in this passage
            for qa in passage_info[constants.qa_pairs]:
                # Setting all keys for supervised = False
                for key in supervision_keys:
                    qa[key] = False
                if constants.qtype in qa:
                    qa.pop(constants.qtype)


    pruned_supervision_dict = make_supervision_dict(dataset, supervision_keys)

    print()
    print(f"TotalNumPassages: {total_num_passages}  Passages remaining annotated: {annotation_for_numpassages}")
    print(f"Num of original question: {total_num_qa}")
    print(f"Original Supervision Dict: {orig_supervision_dict}")
    print(f"Supervision Dict: {pruned_supervision_dict}")

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--annotation_for_numpassages', type=int, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    annotation_for_numpassages = args.annotation_for_numpassages

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
    new_train_dataset = removeDateCompPassageWeakAnnotations(train_dataset, annotation_for_numpassages)
    # new_dev_dataset = removeDateCompPassageWeakAnnotations(train_dataset, 0)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(dev_dataset, f, indent=4)



