import os
import json
import argparse
import random

random.seed(42)


""" Split the dev data into mydev and mytest. """


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def splitDataset(dataset, perc_split: float):
    """Split dataset to make perc_split of train paragraphs into dev paragraphs."""

    total_num_passages = len(dataset)
    num_dev_paras = int(perc_split * total_num_passages)

    passage_idxs = list(dataset.keys())
    passage_idxs = sorted(passage_idxs)
    random.shuffle(passage_idxs)
    dev_passage_idxs = passage_idxs[0:num_dev_paras]
    train_passage_idxs = passage_idxs[num_dev_paras:]

    dev_dataset = {}
    train_dataset = {}

    total_passages, total_questions = 0, 0
    train_passages, dev_passages, train_questions, dev_questions = 0, 0, 0, 0
    for passage_idx, passage_info in dataset.items():
        total_passages += 1
        numq = len(passage_info["qa_pairs"])
        total_questions += numq
        if passage_idx in dev_passage_idxs:
            dev_dataset[passage_idx] = passage_info
            dev_questions += numq
            dev_passages += 1
        else:
            train_dataset[passage_idx] = passage_info
            train_questions += numq
            train_passages += 1

    print(f"\nBefore split: P: {total_passages} Q: {total_questions}")
    print(f"After Split:\nTrain P: {train_passages} Q: {train_questions}\nDev: P:{dev_passages} Q:{dev_questions}")

    return train_dataset, dev_dataset


def write_drop_data(dataset, filepath):
    print(f"Writing dataset: {filepath}")
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--keeporig_dirname")
    parser.add_argument("--dev_perc", type=float, required=True)
    args = parser.parse_args()

    """ The new train/dev/test will replace the train/dev split in input_dir. 
        Original train/dev split will be moved to `keeporig_dirname` inside the input_dir. 
    """

    input_dir = args.input_dir
    keeporig_dirname = args.keeporig_dirname
    keeporig_dir = os.path.join(input_dir, keeporig_dirname)
    os.makedirs(keeporig_dir, exist_ok=True)

    print(f"\nSplitting train data into train/dev from: {input_dir}")

    # Split train into dev and test; rename dev into test

    # Number of paragraphs from train to put into dev
    dev_perc = args.dev_perc

    input_train_json = os.path.join(input_dir, "drop_dataset_train.json")
    input_dev_json = os.path.join(input_dir, "drop_dataset_dev.json")

    train_data = readDataset(input_train_json)
    dev_data = readDataset(input_dev_json)

    print("\nMoving original train/dev split to: {}".format(keeporig_dir))
    write_drop_data(train_data, os.path.join(keeporig_dir, "drop_dataset_train.json"))
    write_drop_data(dev_data, os.path.join(keeporig_dir, "drop_dataset_dev.json"))

    print("\nSplitting train data into train/dev")
    new_train_dataset, new_dev_dataset = splitDataset(train_data, dev_perc)

    test_data = readDataset(input_dev_json)
    num_test_passages = len(test_data)
    num_test_questions = sum([len(pinfo["qa_pairs"]) for _, pinfo in test_data.items()])
    print(f"Test data; P: {num_test_passages}  Q:{num_test_questions}")

    output_train_json = os.path.join(input_dir, "drop_dataset_train.json")
    output_dev_json = os.path.join(input_dir, "drop_dataset_dev.json")
    output_test_json = os.path.join(input_dir, "drop_dataset_test.json")

    print(f"Writing training dataset: {output_train_json}")
    with open(output_train_json, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    print(f"Writing validation dataset: {output_dev_json}")
    with open(output_dev_json, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print(f"Writing test dataset: {output_test_json}")
    with open(output_test_json, "w") as f:
        json.dump(test_data, f, indent=4)
