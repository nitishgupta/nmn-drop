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


def hmyw_program_qattn(tokenized_queslower: str):
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

    for i in range(4, qlen - 1):   # avoid "how many yards was" and '?' with -1
        if i == 4 and question_tokens[i] == "the":
            continue
        if question_tokens[i] not in ["longest", "shortest"]:
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

    return find_or_filter, min_max_or_num, filter_qattn, find_qattn


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


def get_num_minmax_filterfind_node(min_or_max: MinMaxNum, filter: bool, find_qattn, filter_qattn, question_tokens):
    min_max_str = "min" if min_or_max == MinMaxNum.min else "max"
    min_max_node = Node(predicate="select_{}_num".format(min_max_str))
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn, question_tokens)
    else:
        node = get_find_node(find_qattn, question_tokens)
    min_max_node.add_child(node)

    select_num_node = Node(predicate="select_num")
    select_num_node.add_child(min_max_node)

    return select_num_node


def get_num_filterfind_node(filter: bool, find_qattn, filter_qattn, question_tokens):
    if filter:
        node = get_filter_find_node(find_qattn, filter_qattn, question_tokens)
    else:
        node = get_find_node(find_qattn, question_tokens)
    select_num_node = Node(predicate="select_num")
    select_num_node.add_child(node)
    return select_num_node


def node_from_findfilter_maxminnum(find_or_filter: str, min_max_or_num: MinMaxNum,
                                   find_qattn, filter_qattn, question_tokens) -> Node:
    is_filter = True if find_or_filter == "filter" else False
    is_min_max = True if min_max_or_num in [MinMaxNum.min, MinMaxNum.max] else False
    min_or_max: MinMaxNum = None if not is_min_max else min_max_or_num

    if min_or_max is not None:
        select_num_node = get_num_minmax_filterfind_node(min_or_max, is_filter, find_qattn, filter_qattn,
                                                         question_tokens)
    else:
        select_num_node = get_num_filterfind_node(is_filter, find_qattn, filter_qattn, question_tokens)

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
            tokenizedques_lower = " ".join(ques_lower_tokens)
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
                (find_or_filter, min_max_or_num, filter_qattn, find_qattn) = hmyw_program_qattn(tokenizedques_lower)

                program_node: Node = node_from_findfilter_maxminnum(find_or_filter, min_max_or_num,
                                                                    find_qattn, filter_qattn, question_tokens)
                qtype = str(min_max_or_num) + "_" + find_or_filter
                qtype_dist[qtype] += 1

                # print(question)
                # print(qtype)
                # print(f"{find_qattn}  {filter_qattn}")
                # print()

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

                if min_max_or_num is not MinMaxNum.num and relevant_number_entidxs:
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
