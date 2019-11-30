from typing import List, Dict, Tuple
import json
import os
from collections import defaultdict
import itertools

import datasets.drop.constants as constants
import argparse

WHO_RELOCATE_NGRAMS = ["which player scored", "who kicked the", "who threw the", "who scored the", "who caught the"]

EVENTS = [
    "td",
    "touchdown",
    "td pass",
    "touchdown pass",
    "rushing td",
    "rushing touchdown",
    "defensive td",
    "defensive touchdown",
    "field goal"
]

RELOCATE_QUERIES = [
    "which player scored",
    "who kicked",
    "who threw",
    "who scored",
    "who caught",
]

MIN_MAX_MODIFIERS = [
    "longest",
    "shortest"
]

FIND_MODIFIERS = [
    "first",
    "last",
    "final"
]

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


def convert_start_end_to_attention_vector(length, start, end):
    """Convert start/end of a span into a binary vector. start/end is inclusive/exclusive."""
    attention_vector = [0.0] * length
    attention_vector[start:end] = [1.0] * (end - start)
    return attention_vector


def generate_relocate_find_questions():
    relocate_find_questions = []
    for relocate_query, find_modifier, event in itertools.product(RELOCATE_QUERIES, FIND_MODIFIERS, EVENTS):
        question1 = relocate_query + " the " + find_modifier + " " + event + " ?"
        question2 = relocate_query + " the " + find_modifier + " " + event + " of the game ?"
        relocate_find_questions.append(question1)
        relocate_find_questions.append(question2)

    print(f"Num of Relocate-Find questions in repo: {len(relocate_find_questions)}")
    return relocate_find_questions


def get_relocate_find_attention(question_tokens: List[str]):
    """Get the relocate and find attention for relocate-find questions.
    These are of the kind: "relocate_query + " the " + find_modifier + " " + event + " of the game ?"
    Everything before the first "the" is relocate-attn
    Everything after the first "the" until "?" or "of the game ?" is find-attn
    """
    question_len = len(question_tokens)
    first_the_idx = question_tokens.index("the")
    relocate_start = 0
    relocate_end = first_the_idx  # exclusive,

    find_start = first_the_idx + 1  # inclusive
    if "of the game" in " ".join(question_tokens):
        find_end = question_len - 4
    else:
        find_end = question_len - 1

    relocate_attention = convert_start_end_to_attention_vector(question_len, relocate_start, relocate_end)
    find_attention = convert_start_end_to_attention_vector(question_len, find_start, find_end)

    return relocate_attention, find_attention


def generate_relocate_minmax_find_questions():
    relocate_minmax_find_questions = []
    for relocate_query, mixmax_modifier, event in itertools.product(RELOCATE_QUERIES, MIN_MAX_MODIFIERS, EVENTS):
        question1 = relocate_query + " the " + mixmax_modifier + " " + event + " ?"
        question2 = relocate_query + " the " + mixmax_modifier + " " + event + " of the game ?"
        relocate_minmax_find_questions.append(question1)
        relocate_minmax_find_questions.append(question2)

    print(f"Num of Relocate-MinMax-Find questions in repo: {len(relocate_minmax_find_questions)}")
    return relocate_minmax_find_questions


def get_relocate_minmax_find_attention(question_tokens: List[str]):
    """Get the relocate and find attention for relocate-find questions.
    These are of the kind: "relocate_query + " the " + mixmax_modifier + " " + event + " of the game ?"
    Everything before the first "the" is relocate-attn
    Everything after the first "the mixmax_modifier" until "?" or "of the game ?" is find-attn
    """
    question_len = len(question_tokens)
    first_the_idx = question_tokens.index("the")
    relocate_start = 0
    relocate_end = first_the_idx  # exclusive,

    find_start = first_the_idx + 2  # inclusive, skipping "the minmax_modifier"
    if "of the game" in " ".join(question_tokens):
        find_end = question_len - 4
    else:
        find_end = question_len - 1

    relocate_attention = convert_start_end_to_attention_vector(question_len, relocate_start, relocate_end)
    find_attention = convert_start_end_to_attention_vector(question_len, find_start, find_end)
    return relocate_attention, find_attention


def generate_relocate_minmax_filter_find_questions():
    relocate_minmax_filter_find_questions = []
    for relocate_query, mixmax_modifier, event, filter_modifier in itertools.product(
            RELOCATE_QUERIES, MIN_MAX_MODIFIERS, EVENTS, FILTER_MODIFIERS):
        question1 = relocate_query + " the " + mixmax_modifier + " " + event + " of " + filter_modifier + " ?"
        question2 = relocate_query + " the " + mixmax_modifier + " " + event + " in " + filter_modifier + " ?"
        question3 = relocate_query + " the " + mixmax_modifier + " " + event + " during " + filter_modifier + " ?"
        relocate_minmax_filter_find_questions.append(question1)
        relocate_minmax_filter_find_questions.append(question2)
        relocate_minmax_filter_find_questions.append(question3)

    print(f"Num of Relocate-MinMax-Filter-Find questions in repo: {len(relocate_minmax_filter_find_questions)}")
    return relocate_minmax_filter_find_questions


def get_relocate_minmax_filter_find_attention(question_tokens: List[str]):
    """Get the relocate and find attention for relocate-find questions.
    These are : "relocate_query + " the " + mixmax_modifier + " " + event + " of/in/during " + filter_modifier + " ?"
    relocate-attn: Everything before the first "the" is
    find-attn:     Everything after the first "the mixmax_modifier" until the first "of/in/during"
    filter-attn:   Everything after the first "of/in/during"

    Since the only instance of "of"/"in"/"during" would be as a filler in these questions, the above rules should work.
    """
    question_len = len(question_tokens)
    first_the_idx = question_tokens.index("the")
    relocate_start = 0
    relocate_end = first_the_idx  # exclusive,
    relocate_attention = convert_start_end_to_attention_vector(question_len, relocate_start, relocate_end)

    preposition_idx = None
    for prep in ["of", "in", "during"]:
        if prep in question_tokens:
            preposition_idx = question_tokens.index(prep)
    assert preposition_idx is not None

    find_start = first_the_idx + 2  # inclusive, skipping "the minmax_modifier"
    find_end = preposition_idx  # exclusive
    find_attention = convert_start_end_to_attention_vector(question_len, find_start, find_end)

    filter_start = preposition_idx + 1  # inclusive, start after the preposition
    filter_end = question_len - 1  # exclusive, skip the "?" at the end
    filter_attention = convert_start_end_to_attention_vector(question_len, filter_start, filter_end)

    return relocate_attention, filter_attention, find_attention


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def preprocess_Relocate_ques_wattn(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    relocate_find_questions = generate_relocate_find_questions()
    relocate_minmax_find_questions = generate_relocate_minmax_find_questions()
    relocate_minmax_filter_find_questions = generate_relocate_minmax_filter_find_questions()

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)
    program_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]
            question_tokens = tokenized_ques.lower().split(" ")
            if any(span in question_lower for span in WHO_RELOCATE_NGRAMS):

                if tokenized_ques.lower() in relocate_find_questions:
                    qtype = constants.RELOC_find_qtype
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    relocate_qattn, find_qattn = get_relocate_find_attention(question_tokens)
                    question_answer[constants.ques_attention_supervision] = [relocate_qattn, find_qattn]
                    question_answer[constants.qattn_supervised] = True
                    program_dist[qtype] += 1

                if tokenized_ques.lower() in relocate_minmax_find_questions:
                    if "longest" in question_tokens:
                        qtype = constants.RELOC_maxfind_qtype
                    elif "shortest" in question_tokens:
                        qtype = constants.RELOC_minfind_qtype
                    else:
                        raise NotImplementedError
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    relocate_qattn, find_qattn = get_relocate_minmax_find_attention(question_tokens)
                    question_answer[constants.ques_attention_supervision] = [relocate_qattn, find_qattn]
                    question_answer[constants.qattn_supervised] = True
                    program_dist[qtype] += 1

                if tokenized_ques.lower() in relocate_minmax_filter_find_questions:
                    if "longest" in question_tokens:
                        qtype = constants.RELOC_maxfilterfind_qtype
                    elif "shortest" in question_tokens:
                        qtype = constants.RELOC_minfilterfind_qtype
                    else:
                        raise NotImplementedError
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    relocate_qattn, filter_qattn, find_qattn = get_relocate_minmax_filter_find_attention(
                        question_tokens)
                    question_answer[constants.ques_attention_supervision] = [relocate_qattn, filter_qattn, find_qattn]
                    question_answer[constants.qattn_supervised] = True
                    program_dist[qtype] += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")

    print(f"\nProgram Dist: {program_dist}")

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
