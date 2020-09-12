from typing import List, Tuple, Dict
import os
import json
import random
import argparse

from collections import defaultdict

from allennlp.data.tokenizers import SpacyTokenizer

from utils.util import tokenize
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, convert_answer
from semqa.domain_languages.drop_language import Date
from datasets.drop.paired_data.utils import compute_number_support, get_year_difference_candidates, make_paired_qa_pair_dict, \
    get_question_generation_predictor


def get_contrastive_questions(drop_dataset: Dict) -> Dict:
    total_questions = 0

    num_p, num_q = 0, 0
    paras_to_remove = []
    for passage_id, passage_info in drop_dataset.items():
        qas_to_add = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision not in qa:
                continue

            program_node = node_from_dict(qa[constants.program_supervision])
            program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())

            if not any([x in program_lisp for x in ["select_min_num", "select_max_num", "aggregate_count"]]):
                continue

            if constants.shared_substructure_annotations not in qa:
                continue

            qas_to_add.extend(qa[constants.shared_substructure_annotations])

        if qas_to_add:
            passage_info[constants.qa_pairs] = qas_to_add
            num_q += len(qas_to_add)
        else:
            paras_to_remove.append(passage_id)

    for paraid in paras_to_remove:
        drop_dataset.pop(paraid)

    num_p = len(drop_dataset)

    print("Passages: {} Questions: {}".format(num_p, num_q))

    return drop_dataset


def main(args):
    input_json = args.input_json

    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    output_dataset = get_contrastive_questions(input_dataset)
    output_json = args.output_json
    print(f"\nWriting paired-examples to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)