import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from datasets.drop import constants


def mergeDatasets(input_json1: str, input_json2: str):
    """ Merge DROP datasets that contain paired examples.

    Assumption that both datasets contain the same paragraphs and questions; only that the paired questions need to be
    merged.
    """

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json1, "r") as f:
        dataset1 = json.load(f)

    with open(input_json2, "r") as f:
        dataset2 = json.load(f)

    output_passage_dict = {}
    num_ques_w_paired_examples = 0
    num_paired_examples = 0
    max_paired_examples = 0
    total_qa = 0

    for passage_id in dataset1:
        pinfo1 = dataset1[passage_id]
        pinfo2 = dataset2[passage_id]

        qids2qas2 = {}
        for qa in pinfo2[constants.qa_pairs]:
            qid = qa[constants.query_id]
            qids2qas2[qid] = qa

        for qa in pinfo1[constants.qa_pairs]:
            total_qa += 1
            qid = qa[constants.query_id]
            qa2 = qids2qas2[qid]

            paired2 = qa2.get(constants.shared_substructure_annotations, [])
            if len(paired2) > 0:
                # If paired2 is non-empty
                if constants.shared_substructure_annotations not in qa:
                    qa[constants.shared_substructure_annotations] = []
                qa[constants.shared_substructure_annotations].extend(paired2)

            if constants.shared_substructure_annotations in qa:
                qa[constants.shared_substructure_annotations] = qa[constants.shared_substructure_annotations][0:4]
                num_ques_w_paired_examples += 1
                num_paired_examples += len(qa[constants.shared_substructure_annotations])
                max_paired_examples = max(max_paired_examples, len(qa[constants.shared_substructure_annotations]))

    stats = {
        "total_qa": total_qa,
        "num_ques_w_paired_examples": num_ques_w_paired_examples,
        "total_paired_examples": num_paired_examples,
        "max_paired_examples": max_paired_examples,
    }

    print(stats)

    return dataset1, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json1")
    parser.add_argument("--input_json2")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    input_json1 = args.input_json1
    input_json2 = args.input_json2
    output_json = args.output_json

    print(f"\nMerging datasets from:\nD1: {input_json1}\nD2: {input_json2}\nOutDir:{output_json}\n")

    output_dataset, stats_dict = mergeDatasets(input_json1=input_json1, input_json2=input_json2)

    output_dir, output_filename = os.path.split(output_json)
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_json = os.path.join(stats_dir, output_filename)

    print(f"\nWriting paired-examples augmented drop data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)

    with open(stats_json, 'w') as outf:
        json.dump(stats_dict, outf, indent=4)

