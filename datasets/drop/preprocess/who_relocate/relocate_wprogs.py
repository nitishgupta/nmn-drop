from typing import List, Dict, Tuple
import json
import os
from collections import defaultdict
import argparse
from enum import Enum

import datasets.drop.constants as constants
from semqa.utils.qdmr_utils import Node, nested_expression_to_lisp

WHO_RELOCATE_NGRAMS = ["which player scored", "who kicked the", "who threw the", "who scored the", "who caught the"]


class MinMaxNum(Enum):
    min = 1
    max = 2
    num = 3



def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


FILTER_MODIFIERS = [
    "the first quarter",
    "the second quarter",
    "the third quarter",
    "the fourth quarter",
    "the first half",
    "the second half",
    "the first two quarters",
    "the last two quarters",
]


def stringarg_from_attention(attn, tokens):
    string_arg = " ".join([x for i, x in enumerate(tokens) if attn[i] == 1])
    return string_arg


def get_find_node(find_qattn, question_tokens):
    find_node = Node(predicate="select_passage")
    find_node.supervision["question_attention_supervision"] = find_qattn
    find_node.string_arg = stringarg_from_attention(find_qattn, question_tokens)
    return find_node


def get_filter_find_node(find_qattn, filter_qattn, question_tokens):
    find_node = Node(predicate="select_passage")
    find_node.supervision["question_attention_supervision"] = find_qattn
    find_node.string_arg = stringarg_from_attention(find_qattn, question_tokens)

    filter_node = Node(predicate="filter_passage")
    filter_node.supervision["question_attention_supervision"] = filter_qattn
    filter_node.string_arg = stringarg_from_attention(filter_qattn, question_tokens)
    filter_node.add_child(find_node)

    return filter_node


def get_project_minmax_filterfind_node(min_or_max: MinMaxNum, filter: bool, find_qattn, filter_qattn, reloc_qattn,
                                       question_tokens):
    min_max_str = "min" if min_or_max == MinMaxNum.min else "max"
    min_max_node = Node(predicate="select_{}_num".format(min_max_str))
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn, question_tokens)
    else:
        node = get_find_node(find_qattn, question_tokens)
    min_max_node.add_child(node)

    project_node = Node(predicate="project_passage")
    project_node.supervision["question_attention_supervision"] = reloc_qattn
    project_node.string_arg = stringarg_from_attention(reloc_qattn, question_tokens)
    project_node.add_child(min_max_node)
    return project_node


def get_project_filterfind_node(filter: bool, find_qattn, filter_qattn, reloc_qattn, question_tokens):
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn, question_tokens)
    else:
        node = get_find_node(find_qattn, question_tokens)
    project_node = Node(predicate="project_passage")
    project_node.supervision["question_attention_supervision"] = reloc_qattn
    project_node.string_arg = stringarg_from_attention(reloc_qattn, question_tokens)
    project_node.add_child(node)
    return project_node


def node_from_findfilter_maxminnum(find_or_filter: str, min_max_or_num: MinMaxNum,
                                   find_qattn, filter_qattn, reloc_qattn, question_tokens) -> Node:
    is_filter = True if find_or_filter == "filter" else False
    is_min_max = True if min_max_or_num in [MinMaxNum.min, MinMaxNum.max] else False
    min_or_max: MinMaxNum = None if not is_min_max else min_max_or_num

    if min_or_max is not None:
        project_node = get_project_minmax_filterfind_node(min_or_max, is_filter, find_qattn, filter_qattn, reloc_qattn,
                                                          question_tokens)
    else:
        project_node = get_project_filterfind_node(is_filter, find_qattn, filter_qattn, reloc_qattn, question_tokens)

    return project_node


def relocate_program_qattn(tokenized_queslower: str):
    """ Here we'll annotate questions with one/two attentions depending on if the program type is
        1. find(QuestionAttention)
        2. filter(QuestionAttention, find(QuestionAttention))
    """
    question_lower = tokenized_queslower
    question_tokens: List[str] = question_lower.split(" ")
    qlen = len(question_tokens)

    reloc_qattn = [0] * qlen
    find_qattn = [0] * qlen
    filter_qattn = None
    find_or_filter = None

    for i in range(0, 3):
        reloc_qattn[i] = 1

    if any([span in tokenized_queslower for span in FILTER_MODIFIERS]):
        filter_qattn = [0] * qlen
        for i in range(0, 4):
            filter_qattn[qlen - 1 - 1 - i] = 1  # extra -1 to avoid '?' at the end
        find_or_filter = "filter"
    else:
        find_or_filter = "find"

    for i in range(qlen - 1):   # avoid '?' with -1
        if reloc_qattn[i] != 1 and question_tokens[i] not in ["longest", "shortest"]:
            if filter_qattn:
                if filter_qattn[i] != 1:
                    find_qattn[i] = 1
            else:
                find_qattn[i] = 1

    # Using above, find would attend to "of the game", "in the game" -- removing that
    if question_tokens[-4:-1] == ["of", "the", "game"] or question_tokens[-4:-1] == ["in", "the", "game"]:
        find_qattn[-4:-1] = [0, 0, 0]

    if "longest" in question_tokens:
        min_max_or_num = MinMaxNum.max
    elif "shortest" in question_tokens:
        min_max_or_num = MinMaxNum.min
    else:
        min_max_or_num = MinMaxNum.num

    return find_or_filter, min_max_or_num, reloc_qattn, filter_qattn, find_qattn


def convert_span_to_attention(qlen, span):
    # span is inclusive on both ends
    qattn = [0.0] * qlen
    for x in range(span[0], span[1] + 1):
        qattn[x] = 1.0

    return qattn


def preprocess_Relocate_ques_wattn(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_attn = 0
    num_passages = len(dataset)
    qtype_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.question]
            question_lower = original_question.lower()
            question_tokens = question_answer[constants.question_tokens]
            tokenized_ques = " ".join(question_tokens)
            qlen = len(question_tokens)
            if any(span in question_lower for span in WHO_RELOCATE_NGRAMS):
                if any([t in question_tokens for t in ["more", "less", "most", "least"]]):
                    continue

                (find_or_filter, min_max_or_num, reloc_qattn, filter_qattn, find_qattn) = \
                    relocate_program_qattn(tokenized_queslower=tokenized_ques.lower())

                node: Node = node_from_findfilter_maxminnum(find_or_filter, min_max_or_num, find_qattn, filter_qattn,
                                                            reloc_qattn, question_tokens)
                program_node = Node(predicate="select_passagespan_answer")
                program_node.add_child(node)

                question_answer[constants.program_supervision] = program_node.to_dict()
                qtype = str(min_max_or_num) + "_" + find_or_filter

                qtype_dist[qtype] += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Ques with attn: {questions_w_attn}")
    print(f"QType distribution: {qtype_dist}")

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = preprocess_Relocate_ques_wattn(train_dataset)

    new_dev_dataset = preprocess_Relocate_ques_wattn(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")
