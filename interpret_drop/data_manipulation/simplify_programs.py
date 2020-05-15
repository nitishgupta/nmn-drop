import os
import json
import copy
import argparse

import datasets.drop.constants as constants


""" This script is used to modify data so that mix-max modules are never used """

supervision_keys = [
    constants.program_supervised,
    constants.qattn_supervised,
    constants.execution_supervised,
    constants.strongly_supervised,
]


ORIG_TO_SIMPLE_MAPPING = {
    constants.MAX_find_qtype: constants.NUM_find_qtype,
    constants.MIN_find_qtype: constants.NUM_find_qtype,
    constants.MAX_filter_find_qtype: constants.NUM_filter_find_qtype,
    constants.MIN_filter_find_qtype: constants.NUM_filter_find_qtype,
    constants.RELOC_maxfind_qtype: constants.RELOC_find_qtype,
    constants.RELOC_minfind_qtype: constants.RELOC_find_qtype,
    constants.RELOC_maxfilterfind_qtype: constants.RELOC_filterfind_qtype,
    constants.RELOC_minfilterfind_qtype: constants.RELOC_filterfind_qtype,
}


REMOVE_PROGRAM_SUP = {
    constants.DATECOMP_QTYPE,
    constants.NUMCOMP_QTYPE,
}


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def replace_program_supervision(dataset):
    for para_id, para_info in dataset.items():
        qa_pairs = para_info["qa_pairs"]
        for qa in qa_pairs:
            if "qtype" in qa:
                qtype = qa["qtype"]
                if qtype in ORIG_TO_SIMPLE_MAPPING:
                    new_qtype = ORIG_TO_SIMPLE_MAPPING[qtype]
                    qa["qtype"] = new_qtype
                if qtype in REMOVE_PROGRAM_SUP:
                    qa.pop("qtype")
                    for key in supervision_keys:
                        qa[key] = False
    return dataset


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_tr_dataset = replace_program_supervision(train_dataset)
    new_dev_dataset = replace_program_supervision(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_tr_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    main(args)