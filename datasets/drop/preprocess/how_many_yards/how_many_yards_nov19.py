from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
import copy
from collections import defaultdict
import datasets.drop.constants as constants
import argparse
import re

""" This script is used to augment date-comparison-data by flipping events in the questions """
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
    tokenized_question,
    tokenized_passage,
    num_answer,
    attention,
    passage_num_mens,
    passage_num_entidxs,
    passage_num_vals,
):
    WINDOW = 10
    passage_tokens = tokenized_passage.split(" ")
    question_tokens = tokenized_question.split(" ")

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
        number_grounding = [0] * len(passage_num_vals)
        number_values = set()
        for entidx in relevant_number_entidxs:
            number_grounding[entidx] = 1
            number_values.add(passage_num_vals[entidx])
        number_grounding = [number_grounding]
        number_values = [list(number_values)]
        if num_answer not in number_values[0]:  # It's now a list
            number_grounding = None
            number_values = None
        else:
            if "longest" in question_tokens and num_answer != max(number_values[0]):
                number_grounding = None
                number_values = None
            if "shortest" in question_tokens and num_answer != min(number_values[0]):
                number_grounding = None
                number_values = None

    else:
        number_grounding = None
        number_values = None


    # if number_grounding is not None:
    #     print(tokenized_question)
    #     print(attended_tokens)
    #     print(f"Answer: {num_answer}")
    #     print(tokenized_passage)
    #     print(passage_num_vals)
    #     print(f"Annotation: {number_values}")
    #     print()

    return number_grounding, number_values


def preprocess_HowManyYardsWasThe_ques(dataset, ques_attn: bool, number_supervision: bool):
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
    questions_w_attn = 0
    questions_w_numground = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)
    counter = 1


    num_programsupervised_ques = 0

    for passage_id, passage_info in dataset.items():

        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_entidxs = passage_info[constants.passage_num_entidx]
        passage_num_vals = passage_info[constants.passage_num_normalized_values]
        tokenized_passage = passage_info[constants.tokenized_passage]

        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            answer = question_answer[constants.answer]

            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            tokenized_question = question_answer[constants.tokenized_question]
            ques_lower_tokens = tokenized_question.lower().split(" ")
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

                if qtype is not None:
                    num_programsupervised_ques += 1
                    qtype_dist[qtype] += 1
                    find_qattn, filter_qattn = get_question_attention(question_tokens=ques_lower_tokens, qtype=qtype)

                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True

                    question_answer[constants.qattn_supervised] = True
                    if filter_qattn is not None:
                        question_answer[constants.ques_attention_supervision] = [filter_qattn, find_qattn]
                    else:
                        question_answer[constants.ques_attention_supervision] = [find_qattn]
                    questions_w_attn += 1

                    if number_supervision is True:
                        num_answer_str = answer["number"]
                        num_answer = float(num_answer_str) if num_answer_str else None

                        qattn = copy.deepcopy(find_qattn)
                        # if filter_qattn is not None:
                        #     qattn = [x + y for (x, y) in zip(qattn, filter_qattn)]

                        number_grounding, number_values = get_number_distribution_supervision(
                            tokenized_question,
                            tokenized_passage,
                            num_answer,
                            qattn,
                            passage_num_mens,
                            passage_num_entidxs,
                            passage_num_vals,
                        )
                        if number_grounding is not None:
                            question_answer[constants.exection_supervised] = True
                            question_answer[constants.qspan_numgrounding_supervision] = number_grounding
                            question_answer[constants.qspan_numvalue_supervision] = number_values
                            questions_w_numground += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with attention supervised: {questions_w_attn}")
    print(f"Num of QA with num-grounding supervised: {questions_w_numground}")
    print(f"Number of program supervised questions: {num_programsupervised_ques}")
    print(f"Qtype dist: {qtype_dist}")

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--qattn", action="store_true", default=False)
    parser.add_argument("--numground", action="store_true", default=False)
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir
    output_dir = args.output_dir
    qattn = args.qattn
    numbergrounding = args.numground

    if numbergrounding is True:
        assert qattn is True

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
    new_train_dataset = preprocess_HowManyYardsWasThe_ques(
        train_dataset, ques_attn=qattn, number_supervision=numbergrounding
    )
    print()
    new_dev_dataset = preprocess_HowManyYardsWasThe_ques(
        dev_dataset, ques_attn=qattn, number_supervision=numbergrounding
    )

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written HowManyYards datasets")
