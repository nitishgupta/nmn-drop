from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
import copy
from collections import defaultdict
import datasets.drop.constants as constants
import argparse
from semqa.utils.qdmr_utils import nested_expression_to_tree, lisp_to_nested_expression, Node
from enum import Enum


STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])


class MinMaxNum(Enum):
    min = 1
    max = 2
    num = 3


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def get_number_distribution_supervision(
    question_tokens,
    passage_tokens,
    num_answer,
    attention,
    passage_num_mens,
    passage_num_entidxs,
    passage_num_vals,
):
    WINDOW = 10

    # Only supervised longest / shortest questions -- cannot do the first / last kind of questions
    if "longest" not in question_tokens and "shortest" not in question_tokens:
        return None, None
    if num_answer is None:
        return None, None

    # These are the relevant tokens in the question. We'd like to find numbers that are surrounded by these tokens
    attended_tokens = [token for att, token in zip(attention, question_tokens) if att > 0]
    attended_tokens = set(attended_tokens)
    # Replacing TD with touchdown
    if "TD" in attended_tokens:
        attended_tokens.remove("TD")
        attended_tokens.add("touchdown")
    if "goals" in attended_tokens:
        attended_tokens.remove("goals")
        attended_tokens.add("goal")
    if "touchdowns" in attended_tokens:
        attended_tokens.remove("touchdowns")
        attended_tokens.add("touchdown")
    irrelevant_tokens = ["'", "'s", "of", "the", "game", "games", "in"]
    # Remove irrelevant tokens from attended-tokens
    for t in irrelevant_tokens:
        if t in attended_tokens:
            attended_tokens.remove(t)

    # Num of passage number tokens
    number_token_idxs = [x for (_, x, _) in passage_num_mens]

    relevant_number_tokenidxs = []
    relevant_number_entidxs = []
    relevant_number_values = []

    for menidx, number_token_idx in enumerate(number_token_idxs):
        try:
            if passage_tokens[number_token_idx + 1] != "-" or passage_tokens[number_token_idx + 2] != "yard":
                continue
        except:
            continue
        starting_tokenidx = max(0, number_token_idx - WINDOW)  # Inclusive
        ending_tokenidx = min(len(passage_tokens), number_token_idx + WINDOW + 1)  # Exclusive
        surrounding_passage_tokens = set(passage_tokens[starting_tokenidx:ending_tokenidx])
        if "TD" in surrounding_passage_tokens:
            surrounding_passage_tokens.remove("TD")
            surrounding_passage_tokens.add("touchdown")
        if "goals" in surrounding_passage_tokens:
            surrounding_passage_tokens.remove("goals")
            surrounding_passage_tokens.add("goal")
        if "touchdowns" in surrounding_passage_tokens:
            surrounding_passage_tokens.remove("touchdowns")
            surrounding_passage_tokens.add("touchdown")
        intersection_tokens = surrounding_passage_tokens.intersection(attended_tokens)
        if intersection_tokens == attended_tokens:
            relevant_number_tokenidxs.append(number_token_idx)
            relevant_number_entidxs.append(passage_num_entidxs[menidx])
            relevant_number_values.append(passage_num_vals[passage_num_entidxs[menidx]])

    if relevant_number_entidxs:
        relevant_number_entidxs = sorted(list(set(relevant_number_entidxs)))
        number_values = set()
        for entidx in relevant_number_entidxs:
            number_values.add(passage_num_vals[entidx])
        number_values = list(number_values)
        if num_answer not in number_values:  # It's now a list
            relevant_number_entidxs = None
            number_values = None

    else:
        relevant_number_entidxs = None
        number_values = None

    return relevant_number_entidxs, number_values


def get_question_attention(question_tokens: List[str]) -> Tuple[List[int], List[int]]:
    tokens_with_find_attention = [
        "touchdown",
        "run",
        "pass",
        "field",
        "goal",
        "passing",
        "TD",
        "td",
        "rushing",
        "kick",
        "scoring",
        "drive",
        "touchdowns",
        "reception",
        "interception",
        "return",
        "goals",
    ]
    tokens_with_no_attention = [
        "how",
        "How",
        "many",
        "yards",
        "was",
        "the",
        "longest",
        "shortest",
        "?",
        "of",
        "in",
        "game",
    ]
    qlen = len(question_tokens)
    find_qattn = [0.0] * qlen
    filter_qattn = [0.0] * qlen

    for i, token in enumerate(question_tokens):
        if token in tokens_with_no_attention:
            continue
        if token in tokens_with_find_attention:
            find_qattn[i] = 1.0
        else:
            filter_qattn[i] = 1.0

    if sum(find_qattn) == 0:
        find_qattn = None
    if sum(filter_qattn) == 0:
        filter_qattn = None

    return find_qattn, filter_qattn


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


def get_num_minmax_filterfind_node(min_or_max: MinMaxNum, filter: bool, find_qattn, filter_qattn):
    min_max_str = "min" if min_or_max == MinMaxNum.min else "max"
    min_max_node = Node(predicate="select_{}_num".format(min_max_str))
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn)
    else:
        node = get_find_node(find_qattn)
    min_max_node.add_child(node)

    select_num_node = Node(predicate="select_num")
    select_num_node.add_child(min_max_node)

    return select_num_node


def get_num_filterfind_node(filter: bool, find_qattn, filter_qattn):
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn)
    else:
        node = get_find_node(find_qattn)
    select_num_node = Node(predicate="select_num")
    select_num_node.add_child(node)
    return select_num_node



def node_from_findfilter_maxminnum(find_or_filter: str, min_max_or_num: MinMaxNum,
                                   find_qattn, filter_qattn) -> Node:
    is_filter = True if find_or_filter == "filter" else False
    is_min_max = True if min_max_or_num in [MinMaxNum.min, MinMaxNum.max] else False
    min_or_max: MinMaxNum = None if not is_min_max else min_max_or_num

    if min_or_max is not None:
        select_num_node = get_num_minmax_filterfind_node(min_or_max, is_filter, find_qattn, filter_qattn)
    else:
        select_num_node = get_num_filterfind_node(is_filter, find_qattn, filter_qattn)

    return select_num_node


def preprocess_HowManyYardsWasThe_ques(dataset):
    """ This function prunes questions that start with "How many yards was".

        Along with pruning, we also supervise the longest/shortest/second longest/second shortest questions
        by adding the question_type for those questions.

        Currently ---
        We prune out questions like `the second longest / shortest`.
        Still does prune questions like `Tom Brady's second longest/shortest` infact we label them as longest/shortest
        instead of second longest/shortest. But their size is minuscule

        Question-attention
        If the `ques_attn` flag is ON, we also add question-attention supervision

    """

    how_many_yards_was = "how many yards was"

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_qtypes = 0
    questions_w_attn = 0
    questions_w_numground = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)
    counter = 1

    for passage_id, passage_info in dataset.items():

        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_entidxs = passage_info[constants.passage_num_entidx]
        passage_num_vals = passage_info[constants.passage_num_normalized_values]
        passage_tokens = passage_info[constants.passage_tokens]

        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            answer = question_answer[constants.answer]
            total_ques += 1

            question = question_answer[constants.question]
            question_tokens = question_answer[constants.question_tokens]
            ques_lower_tokens = [t.lower() for t in question_tokens]
            question_lower = question.lower()

            # Keep questions that contain "how many yards was"
            if how_many_yards_was in question_lower:

                if "second longest" in question_lower or "second shortest" in question_lower:
                    continue

                # Rest of the questions can be of these kinds:
                # 1. Find or Filter(Find)
                # 2. Longest/Shortest/FindNum

                # We will find the ques-attentions for find vs. filter
                # Using the existence of longest / shortest word we can figure out between Max/Min/Num
                # Tuple[List[int], List[int]]
                (find_qattn, filter_qattn) = get_question_attention(
                    question_tokens=ques_lower_tokens)

                find_or_filter = None
                if find_qattn is None and filter_qattn is None:
                    pass
                elif find_qattn is None:
                    find_qattn = filter_qattn
                    filter_qattn = None
                    find_or_filter = "find"
                elif filter_qattn is None:
                    find_or_filter = "find"
                else:
                    # Both are not None
                    find_or_filter = "filter"

                if find_or_filter is None:
                    continue

                # Now need to figure out whether it's a findNumber / maxNumber / minNumber
                min_max_or_num = None
                if "longest" in question_lower:
                    min_max_or_num = MinMaxNum.max
                elif "shortest" in question_lower:
                    min_max_or_num = MinMaxNum.min
                else:
                    min_max_or_num = MinMaxNum.num

                program_node: Node = node_from_findfilter_maxminnum(find_or_filter, min_max_or_num,
                                                                    find_qattn, filter_qattn)
                qtype = min_max_or_num + "_" + find_or_filter
                qtype_dist[qtype] += 1

                num_answer_str = answer["number"]
                num_answer = float(num_answer_str) if num_answer_str else None

                qattn = copy.deepcopy(find_qattn)
                if filter_qattn is not None:
                    qattn = [min(1, x + y) for (x, y) in zip(qattn, filter_qattn)]

                relevant_number_entidxs, number_values = get_number_distribution_supervision(
                    question_tokens,
                    passage_tokens,
                    num_answer,
                    qattn,
                    passage_num_mens,
                    passage_num_entidxs,
                    passage_num_vals,
                )

                if min_max_or_num is not "num" and relevant_number_entidxs:
                    minmax_node = program_node.children[0]
                    minmax_node.supervision["num_entidxs"] = relevant_number_entidxs
                    question_answer[constants.execution_supervised] = True
                    questions_w_numground += 1

                question_answer[constants.program_supervision] = program_node.to_dict()

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and program supervised: {questions_w_qtypes}")
    print(f"Num of QA with attention supervised: {questions_w_attn}")
    print(f"Num of QA with num-grounding supervised: {questions_w_numground}")
    print(f"Qtype dist: {qtype_dist}")

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

    print()
    new_train_dataset = preprocess_HowManyYardsWasThe_ques(train_dataset)
    print()
    new_dev_dataset = preprocess_HowManyYardsWasThe_ques(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written HowManyYards datasets")
