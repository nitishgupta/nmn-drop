from typing import List
import json
from nltk.corpus import stopwords
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse
from semqa.utils.qdmr_utils import Node

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

COUNT_TRIGRAMS = [
    "how many field goals did",
    "how many field goals were",
    "how many interceptions did",
    "how many passes",
    "how many rushing",
    "how many touchdown passes did",
    "how many touchdowns did the",
    "how many touchdowns were scored",
]


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def get_find_node(find_qattn):
    find_node = Node(predicate="select_passage")
    find_node.supervision["question_attention_supervision"] = find_qattn
    return find_node


def get_filter_find_node(find_qattn, filter_qattn):
    find_node = Node(predicate="select_passage")
    find_node.supervision["question_attention_supervision"] = find_qattn
    filter_node = Node(predicate="filter_passage")
    filter_node.supervision["question_attention_supervision"] = filter_qattn
    filter_node.add_child(find_node)
    return filter_node


def get_count_filterfind_node(filter: bool, find_qattn, filter_qattn):
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn)
    else:
        node = get_find_node(find_qattn)
    count_node = Node(predicate="aggregate_count")
    count_node.add_child(node)
    return count_node


def node_from_findfilter(find_or_filter: str, find_qattn, filter_qattn) -> Node:
    is_filter = True if find_or_filter == "filter" else False
    select_num_node = get_count_filterfind_node(is_filter, find_qattn, filter_qattn)
    return select_num_node


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


def count_program_qattn(tokenized_queslower: str):
    """ Here we'll annotate questions with one/two attentions depending on if the program type is
        1. find(QuestionAttention)
        2. filter(QuestionAttention, find(QuestionAttention))
    """
    question_lower = tokenized_queslower
    question_tokens: List[str] = question_lower.split(" ")
    qlen = len(question_tokens)

    find_qattn = [0] * qlen
    filter_qattn = None
    find_or_filter = None

    if any([span in tokenized_queslower for span in FILTER_MODIFIERS]):
        filter_qattn = [0] * qlen
        for i in range(0, 4):
            filter_qattn[qlen - 1 - 1 - i] = 1  # extra -1 to avoid '?' at the end
        find_or_filter = "filter"
    else:
        find_or_filter = "find"

    for i in range(2, qlen - 1):   # avoid "how many" and '?' with -1
        if filter_qattn:
            if filter_qattn[i] != 1:
                find_qattn[i] = 1
        else:
            find_qattn[i] = 1

    # Using above, find would attend to "of the game", "in the game" -- removing that
    if question_tokens[-4:-1] == ["of", "the", "game"] or question_tokens[-4:-1] == ["in", "the", "game"]:
        find_qattn[-4:-1] = [0, 0, 0]

    return find_or_filter, filter_qattn, find_qattn


def preprocess_HowManyYardsCount_ques(dataset):
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
            tokenizedquestion_lower = " ".join([t.lower() for t in question_tokens])
            if any(span in question_lower for span in COUNT_TRIGRAMS):
                (find_or_filter, filter_qattn, find_qattn) = count_program_qattn(tokenizedquestion_lower)
                qtype = find_or_filter + "_count"

                program_node: Node = node_from_findfilter(find_or_filter, find_qattn, filter_qattn)

                question_answer[constants.program_supervision] = program_node.to_dict()
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

    new_train_dataset = preprocess_HowManyYardsCount_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsCount_ques(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")
