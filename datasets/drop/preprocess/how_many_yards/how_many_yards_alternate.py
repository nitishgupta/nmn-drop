from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
import copy
from collections import defaultdict
import datasets.drop.constants as constants
import argparse
import re
from enum import Enum

from semqa.utils.qdmr_utils import Node, nested_expression_to_lisp, nested_expression_to_tree, lisp_to_nested_expression
from semqa.domain_languages.drop_language_v2 import DropLanguageV2, get_empty_language_object

drop_language: DropLanguageV2 = get_empty_language_object()


class MinMaxNum(Enum):
    min = 1
    max = 2
    num = 3


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


THRESHOLD = 20

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])


NUM_FIND_QUESTION_REGEX_PATTERNS = [
    "how many yards was \S* field goal \?",
    "how many yards was \S* \S* field goal \?",
    "how many yards was \S* td \S* \?",
    "how many yards was \S* \S* td \S* \?",
    "how many yards was \S* touchdown \S* \?",
    "how many yards was \S* \S* touchdown \S* \?",
    "how many yards was \S* touchdown \?",
    "how many yards was \S* \S* touchdown \?",

    "how many yards was \S* field goal of the game \?",
    "how many yards was \S* \S* field goal of the game \?",
    "how many yards was \S* td \S* of the game \?",
    "how many yards was \S* \S* td \S* of the game \?",
    "how many yards was \S* touchdown \S* of the game \?",
    "how many yards was \S* \S* touchdown \S* of the game \?",
    "how many yards was \S* touchdown of the game \?",
    "how many yards was \S* \S* touchdown of the game \?",
]

NUM_FIND_REGEX = re.compile("|".join(NUM_FIND_QUESTION_REGEX_PATTERNS))

NUM_MAXMIN_FIND_QUESTION_REGEX_PATTERNS = [
    "how many yards was \S* longest field goal \?",
    "how many yards was \S* \S* longest field goal \?",
    "how many yards was \S* \S* \S* longest field goal \?",
    "how many yards was \S* shortest field goal \?",
    "how many yards was \S* \S* shortest field goal \?",
    "how many yards was \S* \S* \S* shortest field goal \?",
    "how many yards was \S* longest td \S* \?",
    "how many yards was \S* \S* longest td \S* \?",
    "how many yards was \S* \S* \S* longest td \S* \?",
    "how many yards was \S* shortest td \S* \?",
    "how many yards was \S* \S* shortest td \S* \?",
    "how many yards was \S* \S* \S* shortest td \S* \?",
    "how many yards was \S* longest touchdown \S* \?",
    "how many yards was \S* \S* longest touchdown \S* \?",
    "how many yards was \S* \S* \S* longest touchdown \S* \?",
    "how many yards was \S* shortest touchdown \S* \?",
    "how many yards was \S* \S* shortest touchdown \S* \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* \?",
    "how many yards was \S* longest touchdown \?",
    "how many yards was \S* \S* longest touchdown \?",
    "how many yards was \S* \S* \S* longest touchdown \?",
    "how many yards was \S* shortest touchdown \?",
    "how many yards was \S* \S* shortest touchdown \?",
    "how many yards was \S* \S* \S* shortest touchdown \?",

    "how many yards was \S* longest field goal of the game \?",
    "how many yards was \S* \S* longest field goal of the game \?",
    "how many yards was \S* \S* \S* longest field goal of the game \?",
    "how many yards was \S* shortest field goal of the game \?",
    "how many yards was \S* \S* shortest field goal of the game \?",
    "how many yards was \S* \S* \S* shortest field goal of the game \?",
    "how many yards was \S* longest td \S* of the game \?",
    "how many yards was \S* \S* longest td \S* of the game \?",
    "how many yards was \S* \S* \S* longest td \S* of the game \?",
    "how many yards was \S* shortest td \S* of the game \?",
    "how many yards was \S* \S* shortest td \S* of the game \?",
    "how many yards was \S* \S* \S* shortest td \S* of the game \?",
    "how many yards was \S* longest touchdown \S* of the game \?",
    "how many yards was \S* \S* longest touchdown \S* of the game \?",
    "how many yards was \S* \S* \S* longest touchdown \S* of the game \?",
    "how many yards was \S* shortest touchdown \S* of the game \?",
    "how many yards was \S* \S* shortest touchdown \S* of the game \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* of the game \?",
    "how many yards was \S* longest touchdown of the game \?",
    "how many yards was \S* \S* longest touchdown of the game \?",
    "how many yards was \S* \S* \S* longest touchdown of the game \?",
    "how many yards was \S* shortest touchdown of the game \?",
    "how many yards was \S* \S* shortest touchdown of the game \?",
    "how many yards was \S* \S* \S* shortest touchdown of the game \?",
]

NUM_MINMAX_FIND_REGEX = re.compile("|".join(NUM_MAXMIN_FIND_QUESTION_REGEX_PATTERNS))


NUM_MAXMIN_FILTER_FIND_QUESTION_REGEX_PATTERNS = [
    "how many yards was \S* longest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* \S* longest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* longest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* shortest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* \S* shortest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* shortest field goal \S* \S* \S* quarter \?",
    "how many yards was \S* longest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* longest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* longest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* shortest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* shortest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* shortest td \S* \S* \S* \S* quarter \?",
    "how many yards was \S* longest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* longest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* longest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* shortest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* shortest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* \S* \S* \S* quarter \?",
    "how many yards was \S* longest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* \S* longest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* longest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* shortest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* \S* shortest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* \S* \S* quarter \?",
    "how many yards was \S* longest field goal \S* \S* \S* half \?",
    "how many yards was \S* \S* longest field goal \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* longest field goal \S* \S* \S* half \?",
    "how many yards was \S* shortest field goal \S* \S* \S* half \?",
    "how many yards was \S* \S* shortest field goal \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* shortest field goal \S* \S* \S* half \?",
    "how many yards was \S* longest td \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* longest td \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* longest td \S* \S* \S* \S* half \?",
    "how many yards was \S* shortest td \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* shortest td \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* shortest td \S* \S* \S* \S* half \?",
    "how many yards was \S* longest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* longest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* longest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* shortest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* shortest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* \S* \S* \S* half \?",
    "how many yards was \S* longest touchdown \S* \S* \S* half \?",
    "how many yards was \S* \S* longest touchdown \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* longest touchdown \S* \S* \S* half \?",
    "how many yards was \S* shortest touchdown \S* \S* \S* half \?",
    "how many yards was \S* \S* shortest touchdown \S* \S* \S* half \?",
    "how many yards was \S* \S* \S* shortest touchdown \S* \S* \S* half \?",
]

NUM_MINMAX_FILTER_FIND_REGEX = re.compile("|".join(NUM_MAXMIN_FILTER_FIND_QUESTION_REGEX_PATTERNS))

def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def is_num_find_question(tokenized_question_lower: str):
    match_result = NUM_FIND_REGEX.fullmatch(tokenized_question_lower)
    if (match_result is not None and "longest" not in tokenized_question_lower and
            "shortest" not in tokenized_question_lower):
        return True
    else:
        return False


def is_num_minmax_find_question(tokenized_question_lower: str):
    match_result = NUM_MINMAX_FIND_REGEX.fullmatch(tokenized_question_lower)
    if match_result is not None:
        return True
    else:
        return False


def is_num_minmax_filter_find_question(tokenized_question_lower: str):
    match_result = NUM_MINMAX_FILTER_FIND_REGEX.fullmatch(tokenized_question_lower)
    if match_result is not None:
        return True
    else:
        return False


def convert_start_end_to_attention_vector(length, start, end):
    """Convert start/end of a span into a binary vector. start/end is inclusive/exclusive."""
    attention_vector = [0.0] * length
    attention_vector[start:end] = [1.0] * (end - start)
    return attention_vector


def get_question_attention(question_tokens: List[str], find_or_filter: str, min_max_or_num: MinMaxNum):
    qlen = len(question_tokens)

    if find_or_filter == "find" and min_max_or_num == MinMaxNum.num:
        find_start = 4  # exclude "how many yards was"
        find_end = qlen - 1  # exclusive, don't attend to ?
        find_attention_vector = convert_start_end_to_attention_vector(qlen, find_start,
                                                                      find_end)
        filter_attention_vector = None

    elif find_or_filter == "find" and min_max_or_num in [MinMaxNum.min, MinMaxNum.max]:
        # Two spans of find -- after "how many yards was" until longest/shortest and after until the end
        if min_max_or_num == MinMaxNum.max:
            longest_shortest_idx = question_tokens.index("longest")
        elif min_max_or_num == MinMaxNum.min:
            longest_shortest_idx = question_tokens.index("shortest")
        else:
            raise NotImplementedError
        find_start1 = 4
        find_end1 = longest_shortest_idx    # exclusive, excluding longest/shortest
        find_start2 = longest_shortest_idx + 1
        find_end2 = qlen - 1    # exclusive, don't attend to ?
        if question_tokens[find_start1:find_end1] != ["the"]:
            find_attention_vector1 = convert_start_end_to_attention_vector(qlen, find_start1, find_end1)
        else:
            find_attention_vector1 = [0.0] * qlen
        find_attention_vector2 = convert_start_end_to_attention_vector(qlen, find_start2, find_end2)
        find_attention_vector = [x+y for x, y in zip(find_attention_vector1, find_attention_vector2)]
        filter_attention_vector = None

    elif find_or_filter == "filter" and min_max_or_num in [MinMaxNum.min, MinMaxNum.max]:
        # Two spans of find -- after "how many yards was" until longest/shortest & after that until \S* \S* \S* half ?
        if min_max_or_num == MinMaxNum.max:
            longest_shortest_idx = question_tokens.index("longest")
        elif min_max_or_num == MinMaxNum.min:
            longest_shortest_idx = question_tokens.index("shortest")
        else:
            raise NotImplementedError
        filter_start = qlen - 5  # inclusive, the last 4 tokens excluding "?"
        filter_end = qlen - 1  # exclusive, last token before "?"
        find_start1 = 4
        find_end1 = longest_shortest_idx  # exclusive, excluding longest/shortest
        find_start2 = longest_shortest_idx + 1
        find_end2 = qlen - 5  # exclusive, don't attend to ?
        if question_tokens[find_start1:find_end1] != ["the"]:
            find_attention_vector1 = convert_start_end_to_attention_vector(qlen, find_start1, find_end1)
        else:
            find_attention_vector1 = [0.0] * qlen
        find_attention_vector2 = convert_start_end_to_attention_vector(qlen, find_start2, find_end2)
        find_attention_vector = [x + y for x, y in zip(find_attention_vector1, find_attention_vector2)]
        filter_attention_vector = convert_start_end_to_attention_vector(qlen, filter_start, filter_end)
    else:
        raise NotImplementedError

    return find_attention_vector, filter_attention_vector


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
        relevant_number_entidxs = list(set(relevant_number_entidxs))
        number_values = set()
        for entidx in relevant_number_entidxs:
            number_values.add(passage_num_vals[entidx])
        number_values = list(number_values)
        if num_answer not in number_values:  # It's now a list
            relevant_number_entidxs = None
            number_values = None
        else:
            if "longest" in question_tokens and num_answer != max(number_values):
                relevant_number_entidxs = None
                number_values = None
            if "shortest" in question_tokens and num_answer != min(number_values):
                relevant_number_entidxs = None
                number_values = None

    else:
        relevant_number_entidxs = None
        number_values = None

    return relevant_number_entidxs, number_values


def preprocess_HowManyYardsWasThe_ques(dataset):
    """ This function prunes questions that start with "How many yards was".
        Mostly, longest, shortest style questions. We can also prune for these; look at longestshortest_ques.py

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
    questions_w_numground = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)

    for passage_id, passage_info in dataset.items():

        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_entidxs = passage_info[constants.passage_num_entidx]
        passage_num_vals = passage_info[constants.passage_num_normalized_values]
        passage_tokens = passage_info[constants.passage_tokens]

        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            answer = question_answer[constants.answer]

            total_ques += 1

            original_question = question_answer[constants.question]
            question_tokens = question_answer[constants.question_tokens]
            tokenized_question = " ".join(question_tokens)
            ques_lower_tokens = [t.lower() for t in question_tokens]
            question_lower = original_question.lower()

            # Keep questions that contain "how many yards was"
            if how_many_yards_was in question_lower:
                if "second longest" in question_lower or "second shortest" in question_lower:
                    continue

                find_or_filter = None
                min_max_or_num = None

                if "longest" in question_lower:
                    min_max_or_num = MinMaxNum.max
                elif "shortest" in question_lower:
                    min_max_or_num = MinMaxNum.min
                else:
                    min_max_or_num = MinMaxNum.num

                if is_num_find_question(tokenized_question.lower()):
                    find_or_filter = "find"
                    min_max_or_num = MinMaxNum.num
                if is_num_minmax_find_question(tokenized_question.lower()):
                    find_or_filter = "find"
                    if "longest" in tokenized_question:
                        min_max_or_num = MinMaxNum.max
                    elif "shortest" in tokenized_question:
                        min_max_or_num = MinMaxNum.min
                if is_num_minmax_filter_find_question(tokenized_question.lower()):
                    find_or_filter = "filter"
                    if "longest" in tokenized_question:
                        min_max_or_num = MinMaxNum.max
                    elif "shortest" in tokenized_question:
                        min_max_or_num = MinMaxNum.min

                if find_or_filter is None or min_max_or_num is None:
                    continue

                find_qattn, filter_qattn = get_question_attention(question_tokens=ques_lower_tokens,
                                                                  find_or_filter=find_or_filter,
                                                                  min_max_or_num=min_max_or_num)

                if find_qattn is None:
                    continue

                program_node: Node = node_from_findfilter_maxminnum(find_or_filter, min_max_or_num,
                                                                    find_qattn, filter_qattn)
                qtype = longest_shortest_or_num + "_" + find_or_filter
                qtype_dist[qtype] += 1

                # NUM SUPERVISION
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

                if longest_shortest_or_num is not "num" and relevant_number_entidxs:
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
    qattn = True
    numbergrounding = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput dir: {output_dir}")
    print(f"\nQuestion Attention annotation: {qattn}")
    print(f"\nNumber Grounding annotation: {numbergrounding}")

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

    # print("Written HowManyYards datasets")
