import os
import json
import argparse
from collections import defaultdict
import random
from utils import util

random.seed(100)


""" Split the dev data into mydev and mytest. """


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def splitDataset(dataset, perc_split: float):
    """Split dataset based on given percentage"""

    total_num_passages = len(dataset)
    num_dev_paras = int(perc_split * total_num_passages)
    num_test_paras = total_num_passages - num_dev_paras

    passage_idxs = list(dataset.keys())
    random.shuffle(passage_idxs)
    dev_passage_idxs = passage_idxs[0:num_dev_paras]
    test_passage_idxs = passage_idxs[num_dev_paras:]

    total_num_qa = 0

    dev_dataset = {}
    test_dataset = {}

    for passage_idx, passage_info in dataset.items():
        if passage_idx in dev_passage_idxs:
            dev_dataset[passage_idx] = passage_info
        elif passage_idx in test_passage_idxs:
            test_dataset[passage_idx] = passage_info
        else:
            raise NotImplementedError

    print(f"TotalNumPassages: {total_num_passages}")
    print(f"Size of dev: {len(dev_dataset)} Size of test: {len(test_dataset)}")

    return dev_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--perc_split", type=float, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    dev_json = "drop_dataset_dev.json"

    perc_split = args.perc_split

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_devfp = os.path.join(input_dir, dev_json)
    output_devfp = os.path.join(output_dir, "drop_dataset_mydev.json")
    output_testfp = os.path.join(output_dir, "drop_dataset_mytest.json")

    input_dev_dataset = readDataset(input_devfp)

    print("Training questions .... ")
    print(output_dir)
    new_dev_dataset, new_test_dataset = splitDataset(input_dev_dataset, perc_split)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    with open(output_testfp, "w") as f:
        json.dump(new_test_dataset, f, indent=4)
