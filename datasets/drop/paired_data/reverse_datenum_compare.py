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

# Imports unused but needed for allennlp; TODO: ask Matt how/why this works
from semqa.predictors.qgen_predictor import QuestionGenerationPredictor
from semqa.models.qgen.conditional_qgen_model import ConditionalQuestionGenerationModel
from semqa.data.dataset_readers.qgen.squad_qgen_reader import SquadConditionalQuestionGenerationReader

spacy_tokenizer = SpacyTokenizer()

Entity = Tuple[int, int, str]
CharOffsets = Tuple[int, int]

random.seed(42)


def compute_postorder_position(node: Node, position: int = 0):
    for c in node.children:
        position = compute_postorder_position(c, position)
    node.extras["post_order"] = position
    position += 1
    return position




def reverse_datenum_compare_selects(drop_dataset: Dict):
    relevant_lisps = ["(select_passagespan_answer (compare_date_lt select_passage select_passage))",
                      "(select_passagespan_answer (compare_date_gt select_passage select_passage))",
                      "(select_passagespan_answer (compare_num_lt select_passage select_passage))",
                      "(select_passagespan_answer (compare_num_gt select_passage select_passage))"]

    total_paras, total_q, relevant_q = 0, 0, 0

    for pid, pinfo in drop_dataset.items():
        total_paras += 1

        for qa in pinfo[constants.qa_pairs]:
            total_q += 1
            if constants.program_supervision not in qa or not qa[constants.program_supervision]:
                continue

            program_node = node_from_dict(qa[constants.program_supervision])
            program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
            if program_lisp not in relevant_lisps:
                continue

            relevant_q += 1

            # Flip the selects
            compare_node = program_node.children[0]
            select1, select2 = compare_node.children[0], compare_node.children[1]
            compare_node.children = []
            compare_node.add_child(select2)
            compare_node.add_child(select1)

            qa[constants.program_supervision] = program_node.to_dict()

            if constants.shared_substructure_annotations in qa:
                # These should be simple project(select) questions for the two events
                paired_qas = qa[constants.shared_substructure_annotations]
                if len(paired_qas) == 2:
                    paired_qa_1, paired_qa_2 = paired_qas[0], paired_qas[1]
                    if paired_qa_1["origprog_postorder_node_idx"] == 0:
                        paired_qa_1["origprog_postorder_node_idx"] = 1

                    if paired_qa_2["origprog_postorder_node_idx"] == 1:
                        paired_qa_2["origprog_postorder_node_idx"] = 0

                    qa[constants.shared_substructure_annotations] = [paired_qa_2, paired_qa_1]

    return drop_dataset


def main(args):
    input_json = args.input_json

    print("\nReversing date-comapre and num-compare select nodes... ")

    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    output_dataset = reverse_datenum_compare_selects(input_dataset)
    output_json = args.output_json

    print(f"\nWriting reveresed dataset to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)