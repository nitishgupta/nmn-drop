import os
import json
import argparse
from collections import defaultdict
import random
from utils import util

random.seed(100)

from datasets.drop import constants


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def count_num_exec_sup(dataset, choosen_pids):
    num_exec_sup = 0
    for passage_idx in choosen_pids:
        passage_info = dataset[passage_idx]
        for qa in passage_info[constants.qa_pairs]:
            if constants.execution_supervised in qa:
                if qa[constants.execution_supervised]:
                    num_exec_sup += 1
    return num_exec_sup


def compute_pid2supervisioncount(dataset):
    """Program and Exectution are counted as 1 each."""
    pid2supexamples = defaultdict(int)
    for passage_idx, passage_info in dataset.items():
        num_sup_examples = 0
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervised in qa:
                if qa[constants.program_supervised]:
                    num_sup_examples += 1

            if constants.execution_supervised in qa:
                if qa[constants.execution_supervised]:
                    num_sup_examples += 1
        pid2supexamples[passage_idx] = num_sup_examples

    return pid2supexamples


def compute_choosen_pids(pid2supexamples, perc: float):
    sorted_pid2numsup = util.sortDictByValue(pid2supexamples, decreasing=True)
    num_pids = len(sorted_pid2numsup)
    num_choosen_pids = int(perc * num_pids)
    choosen_pids = [x[0] for x in sorted_pid2numsup[0:num_choosen_pids]]
    return choosen_pids


def make_supervision_dict(dataset):
    basic_keys = [constants.program_supervised, constants.qattn_supervised, constants.execution_supervised]
    qtype_dict = defaultdict(int)
    total_num_qa = 0
    supervision_dict = defaultdict(int)
    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        for qa in passage_info[constants.qa_pairs]:
            if constants.qtype in qa:
                qtype_dict[qa[constants.qtype]] += 1

            all_basic_true = False
            for key in basic_keys:
                if key in qa:
                    supervision_dict[key] += 1 if qa[key] else 0
                    all_basic_true = True if qa[key] else False
                else:
                    all_basic_true = False
            if all_basic_true:
                supervision_dict[constants.strongly_supervised] += 1

    return supervision_dict, qtype_dict


def sample_dataset(dataset, perc):
    """ Given a dataset containing date-comparison questions that are heuristically strongly annotated
        and the number of passages that need to remain strongly annotated, we remove the strong annotations for other
        passages. These annotations include: question-attention, event-date-groundings, etc.
        Fields from which annotation is removed:
        1. constants.datecomp_ques_event_date_groundings, constants.datecomp_ques_event_date_values
        2. constants.datecomp_ques_event_attentions
        3. constants.strongly_annotated - is set to False for all questions

    """

    total_num_passages = len(dataset)

    pid2supexamples = compute_pid2supervisioncount(dataset)
    # choosen_pids = compute_choosen_pids(pid2supexamples, perc)

    passage_idxs = list(dataset.keys())
    num_pids = len(passage_idxs)
    num_choosen_pids = int(perc * num_pids)

    num_exec_sup = 0
    while num_exec_sup < 80:
        random.shuffle(passage_idxs)
        choosen_pids = passage_idxs[0:num_choosen_pids]
        num_exec_sup = count_num_exec_sup(dataset, choosen_pids)

    # choosen_pids = passage_idxs[0:num_choosen_pids]

    new_dataset = {}
    for passage_idx, passage_info in dataset.items():
        if passage_idx in choosen_pids:
            new_dataset[passage_idx] = passage_info

    pruned_supervision_dict, pruned_qtype_dict = make_supervision_dict(new_dataset)
    print()
    print(f"TotalNumPassages: {total_num_passages}  Passages remaining: {len(new_dataset)}")
    print(f"Supervision Dict: {pruned_supervision_dict}")
    print(f"Ques Type Dict: {pruned_qtype_dict}")

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--perc", type=float, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_json = "drop_dataset_train.json"

    perc = args.perc

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    output_trnfp = os.path.join(output_dir, train_json)

    train_dataset = readDataset(input_trnfp)

    print("Training questions .... ")
    print(output_dir)
    new_train_dataset = sample_dataset(train_dataset, perc)
    # new_dev_dataset = removeDateCompPassageWeakAnnotations(train_dataset, 0)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)
