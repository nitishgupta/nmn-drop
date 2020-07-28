from typing import List, Dict, Tuple

import os
import json
import copy
import string
from enum import Enum
from collections import defaultdict
import random
import argparse

from allennlp.data.tokenizers import SpacyTokenizer
import datasets.drop.constants as constants
from semqa.utils.qdmr_utils import node_from_dict, Node

random.seed(100)
spacy_tokenizer = SpacyTokenizer()
IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])

""" This script is used to augment date-comparison-data by flipping events in the questions """

FIRST = "first"
SECOND = "second"

FIRST_operator_tokens = ["first", "earlier"]
SECOND_operator_tokens = ["later", "last", "second"]


def tokenize(text: str) -> List[str]:
    tokens = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def find_valid_spans(passage_tokens: List[str], answer_texts: List[str]) -> List[Tuple[int, int]]:

    # debug = False
    # if 'T. J. Houshmandzadeh' in answer_texts:
    #     debug = True
    normalized_tokens = [token.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
    # if debug:
    #     print('\n')
    #     print(normalized_tokens)
    #     print()

    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        # answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        answer_text_tokens = answer_text.split()
        answer_tokens = [token.lower().strip(STRIPPED_CHARACTERS) for token in answer_text_tokens]
        # if debug:
        #     print(answer_tokens)

        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue

        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans



class AnswerEventOrder(Enum):
    FIRST = 1
    SECOND = 1


def get_answer_event_order(ans_tokens: List[str],
                           event1_tokens: List[str],
                           event2_tokens: List[str]) -> AnswerEventOrder:
    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = AnswerEventOrder.FIRST if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) \
        else AnswerEventOrder.SECOND
    return ans_event


class OperatorType(Enum):
    FIRST = 1
    SECOND = 2


def getQuestionComparisonOperator(question: str) -> OperatorType:
    question_tokens = question.split(" ")
    # Correct if Attn1 is first event

    for t in ["first", "earlier", "forst", "firts"]:
        if t in question_tokens:
            return OperatorType.FIRST

    for t in ["later", "last", "second"]:
        if t in question_tokens:
            return OperatorType.SECOND

    return OperatorType.SECOND


def quesEvents(qstr):
    """ Returns (start, end) span tuples for event1 and event2 in the question """
    or_split = qstr.split(" or ")
    if len(or_split) != 2:
        return None

    tokens = qstr.split(" ")

    or_idx = tokens.index("or")
    # Last token is ? which we don't want to attend to
    event2 = tokens[or_idx + 1 : len(tokens) - 1]
    event2_span = (or_idx + 1, len(tokens) - 1)

    # Gets first index of the item
    try:
        comma_idx = tokens.index(",")
    except:
        comma_idx = 100000
    try:
        colon_idx = tokens.index(":")
    except:
        colon_idx = 100000

    try:
        hyphen_idx = tokens.index("-")
    except:
        hyphen_idx = 100000

    split_idx = min(comma_idx, colon_idx, hyphen_idx)

    if split_idx == 100000 or (or_idx - split_idx <= 1):
        # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
        if "first" in tokens:
            split_idx = tokens.index("first")
        elif "second" in tokens:
            split_idx = tokens.index("second")
        elif "last" in tokens:
            split_idx = tokens.index("last")
        elif "later" in tokens:
            split_idx = tokens.index("later")
        else:
            split_idx = -1

    # assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"
    if split_idx == -1:
        return None

    if tokens[or_idx - 1] == ",":
        event1 = tokens[split_idx + 1 : or_idx - 1]
        event1_span = (split_idx + 1, or_idx - 1)
    else:
        event1 = tokens[split_idx + 1 : or_idx]
        event1_span = (split_idx + 1, or_idx)

    return event1_span, event2_span


def getEventOrderSwitchQuestion(question_answer):
    """ Creating the question by switching the events in the question.
        After this we don't know the exact answer_as_question_span; hence if ans_as_passage_span is []; return None

        For validation data we can keep all the validated_answers as the answer string doesn't change
    """

    question_tokens = question_answer[constants.question_tokens]
    question_tokens = [t for t in question_tokens if t not in ['', ' ']]
    question_tokenized_text = " ".join(question_tokens)
    answer_as_passage_spans = question_answer[constants.answer_passage_spans]

    if not answer_as_passage_spans:
        return None

    event_spans = quesEvents(question_tokenized_text)
    if event_spans is None:
        return None

    # These span ends are exclusive
    event1_span, event2_span = event_spans

    new_question_answer = copy.deepcopy(question_answer)

    pretext = question_tokens[0 : event1_span[0]]
    event2_text = question_tokens[event2_span[0] : event2_span[1]]
    mid_text = question_tokens[event1_span[1] : event2_span[0]]
    event1_text = question_tokens[event1_span[0] : event1_span[1]]
    end_text = question_tokens[event2_span[1] :]

    new_question_tokens = pretext + event2_text + mid_text + event1_text + end_text
    new_question_text = " ".join(new_question_tokens)

    new_question_answer[constants.question] = new_question_text
    new_question_answer[constants.question_tokens] = new_question_tokens

    # Since we don't know the following, we will keep them blank
    new_question_answer[constants.answer_question_spans] = []

    # Reversing the order of supervision
    if constants.program_supervision in question_answer:
        program_node: Node = node_from_dict(new_question_answer[constants.program_supervision])
        date_compare_node = program_node.children[0]
        find1_node, find2_node = date_compare_node.children[0], date_compare_node.children[1]
        if "date1_entidxs" in date_compare_node.supervision and "date2_entidxs" in date_compare_node.supervision:
            date1 = date_compare_node.supervision["date1_entidxs"]
            date2 = date_compare_node.supervision["date2_entidxs"]
            date_compare_node.supervision["date1_entidxs"] = date2
            date_compare_node.supervision["date2_entidxs"] = date1
        find1_sup = find1_node.supervision
        find2_sup = find2_node.supervision
        find1_node.supervision = find2_sup
        find2_node.supervision = find1_sup

    new_question_answer["augmented_data"] = True
    new_question_answer[constants.query_id] = question_answer[constants.query_id] + "-dc-event-switch"

    return new_question_answer


def getQuestionOperatorSwitchQA_wo_QSA(
    question_answer,
    question_tokens,
    passage_tokens,
    passage_token_charidxs,
    passage_text,
):
    """ For a question; "What happened first, A or B" -- generate a question "What happened second, A or B"

        The supervision (question-attention and date-grounding) should work as it is.
    """

    new_question_answer = copy.deepcopy(question_answer)

    question_tokens = [t for t in question_tokens if t not in ['', ' ']]
    question_tokenized_text = " ".join(question_tokens)

    event_spans = quesEvents(question_tokenized_text)
    if event_spans is None:
        return None
    event1_span, event2_span = event_spans

    # Get correct answer to find which event in question is correct
    answer_as_passage_spans = question_answer[constants.answer_passage_spans]
    if answer_as_passage_spans:
        passage_ans_span = answer_as_passage_spans[0]
        answer_tokens = passage_tokens[passage_ans_span[0]: passage_ans_span[1] + 1]
    else:
        answer_text = new_question_answer[constants.answer]["spans"][0]
        answer_tokens = tokenize(answer_text)

    event1_tokens = question_tokens[event1_span[0] : event1_span[1]]
    event2_tokens = question_tokens[event2_span[0] : event2_span[1]]
    # First or Second
    answer_event: AnswerEventOrder = get_answer_event_order(answer_tokens, event1_tokens, event2_tokens)

    # If event1 of question is original_answer, then the other is the answer for our new question
    if answer_event == AnswerEventOrder.FIRST:
        new_ans_tokens = event2_tokens
    else:
        new_ans_tokens = event1_tokens

    if new_ans_tokens[0] == "the":
        new_ans_tokens = new_ans_tokens[1:]

    # Find this question answer span in the passage
    new_ans_as_passage_spans = find_valid_spans(passage_tokens, [" ".join(new_ans_tokens)])
    if not new_ans_as_passage_spans:
        return None

    # Only consider the first grounding
    new_ans_as_passage_span = new_ans_as_passage_spans[0]
    # To find the un-tokenized surface of this answer
    new_ans_start_char_offset, new_ans_end_charoffset = (
        passage_token_charidxs[new_ans_as_passage_span[0]],
        passage_token_charidxs[new_ans_as_passage_span[1] + 1],
    )

    answer_passage_text = passage_text[new_ans_start_char_offset:new_ans_end_charoffset]

    new_question_answer[constants.answer]["spans"] = [answer_passage_text]
    new_question_answer[constants.answer_question_spans] = []
    new_question_answer[constants.answer_passage_spans] = [new_ans_as_passage_span]
    if "validated_answers" in new_question_answer:
        new_question_answer.pop("validated_answers")

    # Make the new question -
    original_ques_operator: OperatorType = getQuestionComparisonOperator(question_tokenized_text)
    # not using the FIRST_operator_tokens list since there is noise in original question
    if original_ques_operator == OperatorType.FIRST:
        tokens_to_replace = ["first", "earlier", "forst", "firts"]
        new_operator_token = random.choice(SECOND_operator_tokens)
    else:
        tokens_to_replace = ["later", "last", "second"]
        new_operator_token = random.choice(FIRST_operator_tokens)

    # Replace the
    new_question_text = question_tokenized_text
    for t in tokens_to_replace:
        new_question_text = new_question_text.replace(t, new_operator_token)

    # Original question doesn't contain first, last kind replacable words. Hence skipping
    if question_tokenized_text == new_question_text:
        return None

    new_question_answer[constants.question] = new_question_text
    new_question_answer[constants.question_tokens] = new_question_text.split(" ")

    new_question_answer["augmented_data"] = True
    new_question_answer[constants.query_id] = question_answer[constants.query_id] + "-dc-qop-switch"
    return new_question_answer


# def strongSupervisionFlagAndQType(question_answer_pairs: List[Dict]):
#     num_strongly_supervised_qas = 0
#     supervision_distribution = defaultdict(int)
#
#     for question_answer in question_answer_pairs:
#         question_answer[constants.qtype] = constants.DATECOMP_QTYPE
#         question_answer[constants.program_supervised] = True
#
#         if (
#             question_answer[constants.program_supervised]
#             and question_answer[constants.qattn_supervised]
#             and question_answer[constants.execution_supervised]
#         ):
#             question_answer[constants.strongly_supervised] = True
#         else:
#             question_answer[constants.strongly_supervised] = False
#
#         supervision_distribution[constants.program_supervised] += (
#             1 if question_answer[constants.program_supervised] else 0
#         )
#         supervision_distribution[constants.qattn_supervised] += 1 if question_answer[constants.qattn_supervised] else 0
#         supervision_distribution[constants.execution_supervised] += (
#             1 if question_answer[constants.execution_supervised] else 0
#         )
#         supervision_distribution[constants.strongly_supervised] += (
#             1 if question_answer[constants.strongly_supervised] else 0
#         )
#
#     return question_answer_pairs, supervision_distribution


def augmentDateComparisonData(dataset):
    """ Given a dataset containing date-comparison questions, we augment the data in the following manner:
        - Adding new questions. For eg: "What happened first, eventA or eventB ?"
            1. QuesEvent switched questions: "What happened first, eventB or eventA" - with same answer. 100% success
            2. QuesOperator switched question: "What happened second, eventA or eventB" -- for this we randomly sample
               a token for the operator from the {FIRST, SECOND}_operator_tokens list. Heuristically find which event
               in the question is the answer, and now try to ground the other event in the passage as the answer.
               If grounding is not found then this augmentation fails. Hence <100% success rate for this step.
            3. QuesOperator switched question for the QuesEvent switched question from step 1.

        - QuestionEvent attention annotation -- Two 0/1 binary vectors marking the two events mentioned in the question
          These will act as a supervision for question attention in the model.
          Add a constants.question_attention field that contains a two-tuple for attention vectorss

        - Strong supervision flag
        Additionally, for each instance we also add a boolean-field constants.strongly_annotated that indicates whether
        the question-answer has strong annotations or not.

        - QuestionType
        Add a strong-field constants.qtype to indicate that all these questions are date_comparison questions
        The value is constants.DATECOMP_QTYPE
    """

    new_dataset = {}
    original_operator_dist = defaultdict(float)
    augment_operator_dist = defaultdict(float)
    total_strongly_supervised_qas = 0
    num_qa_original = 0
    num_qa_augment = 0

    supervision_distribution = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        passage = passage_info[constants.passage]
        passage_tokens = passage_info[constants.passage_tokens]
        passage_tokenized_text = " ".join(passage_tokens)
        passage_token_charidxs = passage_info[constants.passage_charidxs]

        num_qa_original += len(passage_info[constants.qa_pairs])
        for question_answer in passage_info[constants.qa_pairs]:
            question = question_answer[constants.question]
            question_tokens = question_answer[constants.question_tokens]
            question_tokenized_text = " ".join(question_tokens)

            ques_operator: OperatorType = getQuestionComparisonOperator(question_tokenized_text)
            original_operator_dist[ques_operator] += 1

            new_qa_pairs.append(question_answer)

            # Given: "first A or B"; generate QA which is "first B or A"
            event_switch_question_answer = getEventOrderSwitchQuestion(question_answer)

            if event_switch_question_answer is not None:
                augment_operator_dist[ques_operator] += 1
                new_qa_pairs.append(event_switch_question_answer)

            # "first A or B" --> "second A or B"
            qoperator_switch_question_answer = getQuestionOperatorSwitchQA_wo_QSA(
                question_answer,
                question_tokens,
                passage_tokens,
                passage_token_charidxs,
                passage)

            if qoperator_switch_question_answer is not None:
                ques_operator = getQuestionComparisonOperator(
                    " ".join(qoperator_switch_question_answer[constants.question_tokens])
                )
                augment_operator_dist[ques_operator] += 1
                new_qa_pairs.append(qoperator_switch_question_answer)

            if event_switch_question_answer is not None:
                # "first B or A" --> "second B or A"
                event_sw_qoperator_sw_question_answer = getQuestionOperatorSwitchQA_wo_QSA(
                    event_switch_question_answer,
                    event_switch_question_answer[constants.question_tokens],
                    passage_tokens,
                    passage_token_charidxs,
                    passage)
                if event_sw_qoperator_sw_question_answer is not None:
                    ques_operator = getQuestionComparisonOperator(
                        " ".join(event_sw_qoperator_sw_question_answer[constants.question_tokens])
                    )
                    augment_operator_dist[ques_operator] += 1
                    new_qa_pairs.append(event_sw_qoperator_sw_question_answer)

        q_ids = set([qa[constants.query_id] for qa in new_qa_pairs])
        assert len(q_ids) == len(new_qa_pairs)

        num_qa_augment += len(new_qa_pairs)
        passage_info[constants.qa_pairs] = new_qa_pairs
        new_dataset[passage_id] = passage_info

    print()
    print(f"Num of original question: {num_qa_original}")
    print(original_operator_dist)
    print(f"Num of question after augmentation: {num_qa_augment}")
    print(augment_operator_dist)
    print(f"Supervision Distribution: {supervision_distribution}")

    return new_dataset


if __name__ == "__main__":
    print("\nData augmentation for date-comp questions")

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

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    new_train_dataset = augmentDateComparisonData(train_dataset)
    new_dev_dataset = augmentDateComparisonData(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written augmented datasets")
