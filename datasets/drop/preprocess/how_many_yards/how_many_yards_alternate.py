from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
import copy
from collections import defaultdict
import datasets.drop.constants as constants
import argparse
import re

from semqa.utils.qdmr_utils import Node, nested_expression_to_lisp, nested_expression_to_tree, lisp_to_nested_expression
from semqa.domain_languages.drop_language_v2 import DropLanguageV2, get_empty_language_object

drop_language: DropLanguageV2 = get_empty_language_object()

def qtype_to_node(qtype) -> Node:
    num_find = ['select_num', 'select_passage']

    num_max_find = ['select_num', ['select_max_num', 'select_passage']]
    num_min_find = ['select_num', ['select_min_num', 'select_passage']]

    num_max_filter_find = ['select_num', ['select_max_num', ['filter_passage', 'select_passage']]]
    num_min_filter_find = ['select_num', ['select_min_num', ['filter_passage', 'select_passage']]]

    qtype_to_nested = {
        constants.NUM_find_qtype: num_find,
        constants.MAX_find_qtype: num_max_find,
        constants.MAX_filter_find_qtype: num_max_filter_find,
        constants.MIN_find_qtype: num_min_find,
        constants.MIN_filter_find_qtype: num_min_filter_find,
    }

    nested_expr = qtype_to_nested.get(qtype, None)
    node = None
    if nested_expr:
        node: Node = nested_expression_to_tree(nested_expr)

    return node


def add_supervision(qtype, node, find_qattn, number_entidxs, question_tokens):
    assert qtype in [constants.MAX_find_qtype, constants.MIN_find_qtype]
    assert len(find_qattn) == len(question_tokens)
    # ['select_num', ['select_max_num', 'select_passage']]
    string_arg = " ".join([t for i, t in enumerate(question_tokens) if find_qattn[i] == 1])
    select_node: Node = node.children[0].children[0]
    select_node.string_arg = string_arg
    select_node.supervision["question_attention_supervision"] = find_qattn

    minmax_node = node.children[0]
    minmax_node.supervision["num_entidxs"] = number_entidxs


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



def get_question_attention(question_tokens: List[str], qtype: str):
    qlen = len(question_tokens)

    if qtype in [constants.NUM_find_qtype]:
        find_start = 4  # exclude "how many yards was"
        find_end = qlen - 1  # exclusive, don't attend to ?
        find_attention_vector = convert_start_end_to_attention_vector(qlen, find_start,
                                                                      find_end)
        filter_attention_vector = None

    elif qtype in [constants.MAX_find_qtype, constants.MIN_find_qtype]:
        # Two spans of find -- after "how many yards was" until longest/shortest and after until the end
        if qtype == constants.MAX_find_qtype:
            longest_shortest_idx = question_tokens.index("longest")
        elif qtype == constants.MIN_find_qtype:
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

    elif qtype in [constants.MAX_filter_find_qtype, constants.MIN_filter_find_qtype]:
        # Two spans of find -- after "how many yards was" until longest/shortest & after that until \S* \S* \S* half ?
        if qtype == constants.MAX_filter_find_qtype:
            longest_shortest_idx = question_tokens.index("longest")
        elif qtype == constants.MIN_filter_find_qtype:
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
    counter = 1

    num_programsupervised_ques = 0

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

                qtype = None
                if is_num_find_question(tokenized_question.lower()):
                    qtype = constants.NUM_find_qtype
                if is_num_minmax_find_question(tokenized_question.lower()):
                    if "longest" in tokenized_question:
                        qtype = constants.MAX_find_qtype
                    elif "shortest" in tokenized_question:
                        qtype = constants.MIN_find_qtype
                if is_num_minmax_filter_find_question(tokenized_question.lower()):
                    if "longest" in tokenized_question:
                        qtype = constants.MAX_filter_find_qtype
                    elif "shortest" in tokenized_question:
                        qtype = constants.MIN_filter_find_qtype

                if qtype is not None and qtype in [constants.MAX_find_qtype, constants.MIN_find_qtype]:
                    find_qattn, filter_qattn = get_question_attention(question_tokens=ques_lower_tokens, qtype=qtype)

                    # NUM SUPERVISION
                    node: Node = qtype_to_node(qtype)
                    if node is None:
                        print(qtype)
                        continue

                    num_answer_str = answer["number"]
                    num_answer = float(num_answer_str) if num_answer_str else None

                    relevant_number_entidxs, number_values = get_number_distribution_supervision(
                        question_tokens,
                        passage_tokens,
                        num_answer,
                        find_qattn,
                        passage_num_mens,
                        passage_num_entidxs,
                        passage_num_vals,
                    )
                    if relevant_number_entidxs is not None:
                        add_supervision(qtype, node, find_qattn, relevant_number_entidxs, question_tokens)

                        program_supervision = node.to_dict()
                        question_answer[constants.program_supervision] = program_supervision
                        question_answer[constants.execution_supervised] = True

                        questions_w_numground += 1
                        num_programsupervised_ques += 1
                        qtype_dist[qtype] += 1

                        new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with num-grounding supervised: {questions_w_numground}")
    print(f"Number of program supervised questions: {num_programsupervised_ques}")
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
