import os
import json
import argparse
import datasets.drop.constants as constants
from collections import defaultdict


SUPERVISION_KEYS = [constants.program_supervised, constants.qattn_supervised, constants.exection_supervised]


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def supervisionStats(input_json):
    """ For this dataset, find the following:
        1. How many questions have a qtype, i.e. program_supervised
        2. For each qtype, find the number of questions with qattn_supervised and execution_supervised
    """
    dataset = readDataset(input_json)

    numparas = 0
    numques = 0

    qtype_dict = defaultdict(int)
    supervision_dict = defaultdict(int)
    supervision_dict["TOTAL"] = defaultdict(int)

    for pid, pinfo in dataset.items():
        numparas += 1
        passage = pinfo[constants.tokenized_passage]

        qa_pairs = pinfo[constants.qa_pairs]

        for qa in qa_pairs:
            numques += 1

            if constants.qtype in qa:
                qtype = qa[constants.qtype]
                assert qa[constants.program_supervised] is True, f"Qtype: {qtype}, program_supervised: False"
            else:
                qtype = "UNK"

            qtype_dict[qtype] += 1
            if qtype not in supervision_dict:
                supervision_dict[qtype] = defaultdict(int)

            for supervision_key in SUPERVISION_KEYS:
                if supervision_key in qa:
                    qa_supervionkey_val = 1 if qa[supervision_key] is True else 0
                    supervision_dict[qtype][supervision_key] += qa_supervionkey_val
                    supervision_dict["TOTAL"][supervision_key] += qa_supervionkey_val

    print(f"Paras: {numparas}  NumQues:{numques}")
    print(f"Qtypes Count: {json.dumps(qtype_dict, indent=2)}")
    print(f"Per Qtype supervision amount:\n {json.dumps(supervision_dict, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir")
    args = parser.parse_args()

    inputdir = args.inputdir

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_trnfp = os.path.join(inputdir, train_json)
    input_devfp = os.path.join(inputdir, dev_json)

    print("\n\n")
    print(input_trnfp)
    supervisionStats(input_trnfp)

    print("\n\n")

    print(input_devfp)
    supervisionStats(input_devfp)
