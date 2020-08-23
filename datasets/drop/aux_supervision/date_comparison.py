from typing import List, Tuple
import json
from nltk.corpus import stopwords
import os
import datasets.drop.constants as constants
from semqa.domain_languages.drop_language import Date
from semqa.utils import qdmr_utils
import argparse

from allennlp.data.tokenizers import SpacyTokenizer


""" This script is used to augment drop_data with auxiliary execution supervision for date-comparison questions """

THRESHOLD = 20

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ",", "when", "was", "did"])

FIRST = "first"
SECOND = "second"

DATE_COMPARISON_TRIGRAMS = [
    "which happened",
    "which event",
    "what happened first",
    "what happened second",
    "what happened later",
    "what happened last",
    "what event happened",
    "what event came",
]


spacy_tokenizer = SpacyTokenizer()


def tokenize(text: str) -> List[str]:
    tokens = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def get_program_supervision(operator: str) -> qdmr_utils.Node:
    assert operator in [FIRST, SECOND], "Operator: {}".format(operator)
    date_lt = "(find_passageSpanAnswer (compare_date_lesser_than find_PassageAttention find_PassageAttention))"
    date_gt = "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
    lisp_program = date_lt if operator == FIRST else date_gt
    node: qdmr_utils.Node = qdmr_utils.nested_expression_to_tree(qdmr_utils.lisp_to_nested_expression(lisp_program))
    return node


def is_date_comparison(question: str, nested_expr_tuple: Tuple):
    if nested_expr_tuple in [('select_passagespan_answer', ('compare_date_lt', 'select_passage', 'select_passage')),
                             ('select_passagespan_answer', ('compare_date_gt', 'select_passage', 'select_passage'))]:
        return True
    else:
        return False
    # question_lower = question.lower()
    # if any(span in question_lower for span in DATE_COMPARISON_TRIGRAMS):
    #     return True
    # else:
    #     return False


def getQuestionComparisonOperator(question_tokens: List[str], nested_expr_tuple: Tuple) -> str:
    if nested_expr_tuple == ('select_passagespan_answer', ('compare_date_lt', 'select_passage', 'select_passage')):
        return FIRST
    elif nested_expr_tuple == ('select_passagespan_answer', ('compare_date_gt', 'select_passage', 'select_passage')):
        return SECOND
    else:
        print(nested_expr_tuple)
        raise NotImplementedError
    # Correct if Attn1 is first event
    # lesser_tokens = ["first", "earlier", "forst", "firts"]
    # greater_tokens = ["later", "last", "second"]
    #
    # for t in lesser_tokens:
    #     if t in question_tokens:
    #         return FIRST
    #
    # for t in greater_tokens:
    #     if t in question_tokens:
    #         return SECOND
    #
    # return SECOND


def getDateTokenIdxs(
    p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]], p_date_entidxs: List[int]
) -> Tuple[List[int], List[int]]:
    """List of date token idxs, and list of their date-ent-idxs."""
    # List of token idxs that are dates
    passage_date_tokens = []
    passage_datetoken_entidxs = []
    for date_men, date_idx in zip(p_date_mens, p_date_entidxs):
        (s, e) = date_men[1]
        passage_date_tokens.extend([x for x in range(s, e + 1)])
        passage_datetoken_entidxs.extend([date_idx for _ in range(s, e + 1)])

    assert len(passage_date_tokens) == len(passage_datetoken_entidxs)
    return passage_date_tokens, passage_datetoken_entidxs


# def quesEvents(qstr) -> Tuple[Tuple[int, int], Tuple[int, int]]:
#     """ Returns (start, end) span tuples for event1 and event2 in the question. end is exclusive"""
#     or_split = qstr.split(" or ")
#     if len(or_split) != 2:
#         return None
#
#     tokens = qstr.split(" ")
#
#     or_idx = tokens.index("or")
#     # Last token is ? which we don't want to attend to
#     event2_span = (or_idx + 1, len(tokens) - 1)
#
#     # Gets first index of the item
#     try:
#         comma_idx = tokens.index(",")
#     except:
#         comma_idx = 100000
#     try:
#         colon_idx = tokens.index(":")
#     except:
#         colon_idx = 100000
#
#     try:
#         hyphen_idx = tokens.index("-")
#     except:
#         hyphen_idx = 100000
#
#     split_idx = min(comma_idx, colon_idx, hyphen_idx)
#
#     if split_idx == 100000 or (or_idx - split_idx <= 1):
#         # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
#         if "first" in tokens:
#             split_idx = tokens.index("first")
#         elif "second" in tokens:
#             split_idx = tokens.index("second")
#         elif "last" in tokens:
#             split_idx = tokens.index("last")
#         elif "later" in tokens:
#             split_idx = tokens.index("later")
#         else:
#             split_idx = -1
#
#     assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"
#
#     event1_span = (split_idx + 1, or_idx)
#
#     return event1_span, event2_span


def find_answer_event(answer_span: str, event1_tokens: List[str], event2_tokens: List[str]) -> str:
    ans_tokens = set(answer_span.split(" "))
    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = FIRST if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) else SECOND
    return ans_event


def difference_in_successive_terms(l: List[int]):
    sum_of_differences = 0
    for i in range(len(l) - 1):
        sum_of_differences += l[i + 1] - l[i]
    return sum_of_differences


def matchEventToPassage(event_tokens: List[str], passage_tokens: List[str]) -> List[int]:
    """ Match a given event's tokens (from ques) to relevant tokens in the passage. """
    important_event_tokens = [t.lower() for t in event_tokens if t.lower() not in STOP_WORDS]

    # These are (auto) sorted in increasing order -- these imp. event tokens found in the passage
    relevant_passage_tokenidxs = []
    for (idx, passage_token) in enumerate(passage_tokens):
        if passage_token.lower() in important_event_tokens:
            relevant_passage_tokenidxs.append(idx)

    # Since event tokens can match spuriously at many locations, the idea is to find a list of important tokens in the
    # passage that are close to each other, aka tighest span
    num_imp_event_tokens = len(important_event_tokens)
    best_diff = 100000
    best_start_point = 0
    for i in range(0, len(relevant_passage_tokenidxs) - num_imp_event_tokens + 1):
        # Starting at i, taking the next n=num_imp_event_tokens passage tokens
        passage_token_span = relevant_passage_tokenidxs[i : i + num_imp_event_tokens]
        # This is tightness in these tokens
        sum_of_token_diffs = difference_in_successive_terms(passage_token_span)
        # If best tightness, then i is a good starting point
        if sum_of_token_diffs < best_diff:
            best_start_point = i
            best_diff = sum_of_token_diffs

    # These tokens now are passage-tokens where the event tokens ground
    pruned_relevant_passage_tokenidxs = relevant_passage_tokenidxs[best_start_point : best_start_point + num_imp_event_tokens]

    return pruned_relevant_passage_tokenidxs


def dateInNeighborhood(
    passage_tokenidxs: List[int], passage_date_tokenidxs: List[int], passage_datetoken_entidxs: List[int], threshold=20
):
    """ Given a list of relevant-passage-tokens, and list of date-tokens in the passage, figure out -
        if there's a date in the neighborhood of the relevant tokens
        For each passage-token, first find the min-distance to a date token. Then find the min-amongst that.
        If this distance crosses a threshold, then a date is not considered in the neighborhood of the passage-tokens
    """

    distance_to_dates = []
    closest_date_entidxs = []
    for tokenidx in passage_tokenidxs:
        min_distance_to_date = 10000
        closest_dateidx = -1
        for date_tokenidx, date_idx in zip(passage_date_tokenidxs, passage_datetoken_entidxs):
            dis = abs(date_tokenidx - tokenidx)
            if dis < min_distance_to_date:
                min_distance_to_date = dis
                closest_dateidx = date_idx
        distance_to_dates.append(min_distance_to_date)
        closest_date_entidxs.append(closest_dateidx)

    if len(distance_to_dates) == 0:
        return False, -1

    avg_distance_to_dates = float(sum(distance_to_dates)) / len(distance_to_dates)
    # Mode
    closest_date_entidx = max(set(closest_date_entidxs), key=closest_date_entidxs.count)

    if avg_distance_to_dates > threshold:
        return False, closest_date_entidx
    else:
        return True, closest_date_entidx


def date_comparison_aux_supervision(dataset):
    """ Prunes questions where a date is not present near both events in the question.
        1. Find events in the question
        2. Find whether a nearby date exists or not; if yes, which date.

        Additionally provides a weak-supervison for which date grounding of the ques-events. We don't keep all date
        supervision from previous step. Don't trust the annotation if:
        1. Dates for both events are the same.
        2. Our date prediction for the events don't match the question's answer. For example, if the questions asks for
           the earlier event, but according to our annotation the answer-event happened later, then the date annotation
           must be wrong.

        This annotation is stored as:
        1. constants.datecomp_ques_event_date_groundings -- a one hot-vector the size of num_passage_dates
        2. constants.datecomp_ques_event_date_values -- A two-tuple, each containing (day, month, year)
        In cases where we don't store the date annotation, the grounding vector is left to zeros, and date values to -1.
        The model should use this fact to figure out a mask in case no annotation is provided
    """

    total_ques = 0
    num_date_compare_ques = 0

    numexamaples_w_dates_annotated = 0

    for passage_id, passage_info in dataset.items():
        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = passage_info[constants.passage_date_mens]
        p_date_entidxs = passage_info[constants.passage_date_entidx]
        p_date_values: List[Tuple] = passage_info[constants.passage_date_normalized_values]

        passage_date_tokenidxs, passage_datetoken_entidxs = getDateTokenIdxs(p_date_mens, p_date_entidxs)

        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            # question_tokenized_text = question_answer[constants.tokenized_question]
            question_tokens: List[str] = question_answer[constants.question_tokens]
            question_tokenized_text = " ".join(question_tokens)
            answer_annotation = question_answer[constants.answer]

            program_supervision = question_answer[constants.program_supervision]
            if program_supervision is None:
                # Only add auxiliary supervision
                continue

            program_node: qdmr_utils.Node = qdmr_utils.node_from_dict(program_supervision)
            nested_expr_tuple = qdmr_utils.convert_nestedexpr_to_tuple(program_node.get_nested_expression())
            if not is_date_comparison(question=question_tokenized_text, nested_expr_tuple=nested_expr_tuple):
                continue

            num_date_compare_ques += 1

            if question_answer[constants.answer_type] != constants.SPAN_TYPE:
                continue

            answer_span: str = answer_annotation["spans"][0]

            ''' old way of finding events from questions. instead we're using arguments from the program. '''
            # # Two (start, end) tuples for the two events mentioned in the question (end exclusive)
            # ques_event_spans: Tuple[Tuple[int, int], Tuple[int, int]] = quesEvents(question_tokenized_text)
            # # If events are not found, we cannot find dates etc. therefore don't add this question
            # if ques_event_spans is None:
            #     continue
            # # For questions with identified events, we'll try to ground dates
            # event1_span, event2_span = ques_event_spans
            # event1_tokens = question_tokens[event1_span[0]:event1_span[1]]
            # event2_tokens = question_tokens[event2_span[0]:event2_span[1]]

            select1_node = program_node.children[0].children[0]
            select2_node = program_node.children[0].children[1]
            select1_string_arg = select1_node.string_arg
            select2_string_arg = select2_node.string_arg

            event1_tokens = tokenize(select1_string_arg)
            event2_tokens = tokenize(select2_string_arg)

            # List of tokenidxs in passage that is a rough grounding for event 1/2
            event1_passage_tokenidxs: List[int] = matchEventToPassage(event1_tokens, passage_tokens)
            event2_passage_tokenidxs: List[int] = matchEventToPassage(event2_tokens, passage_tokens)

            date_near_event1, event1_date_idx = dateInNeighborhood(
                event1_passage_tokenidxs, passage_date_tokenidxs, passage_datetoken_entidxs, threshold=THRESHOLD
            )
            date_near_event2, event2_date_idx = dateInNeighborhood(
                event2_passage_tokenidxs, passage_date_tokenidxs, passage_datetoken_entidxs, threshold=THRESHOLD
            )

            if not date_near_event1 or not date_near_event2:
                continue

            # Returns FIRST if less-than or SECOND if greater-than
            question_operator = getQuestionComparisonOperator(question_tokens, nested_expr_tuple)
            answer_event = find_answer_event(answer_span, event1_tokens, event2_tokens)

            # First find if the date groundings are relevant, checks:
            # 1. Date grounding shouldn't be -1
            # 2. Shouldn't be equal
            # 3. The date grounding should be consistent with answer annotation
            keep_dates = True
            if not (event1_date_idx == -1 or event2_date_idx == -1 or (event1_date_idx == event2_date_idx)):
                answer_event_date = event1_date_idx if answer_event == FIRST else event2_date_idx
                other_event_date = event2_date_idx if answer_event == FIRST else event1_date_idx

                answer_event_date = Date(
                    year=p_date_values[answer_event_date][2],
                    month=p_date_values[answer_event_date][1],
                    day=p_date_values[answer_event_date][0],
                )

                other_event_date = Date(
                    year=p_date_values[other_event_date][2],
                    month=p_date_values[other_event_date][1],
                    day=p_date_values[other_event_date][0],
                )

                # If the questions asks for what happened first, and our date grounding says that answer happened later,
                # means our date annotation is wrong.
                if question_operator == FIRST:
                    if answer_event_date > other_event_date:
                        keep_dates = False
                if question_operator == SECOND:
                    if answer_event_date < other_event_date:
                        keep_dates = False
            else:
                keep_dates = False

            # Compiling gold - program
            date_compare_node = program_node.children[0]
            assert date_compare_node.predicate in ["compare_date_lt", "compare_date_gt"]
            if keep_dates:
                date_compare_node.supervision["date1_entidxs"] = [event1_date_idx]
                date_compare_node.supervision["date2_entidxs"] = [event2_date_idx]
                question_answer[constants.execution_supervised] = True
                numexamaples_w_dates_annotated += 1

            question_answer[constants.program_supervision] = program_node.to_dict()

    print(f"Total num questions:{total_ques}  num of date_compare_ques:{num_date_compare_ques}")
    print(f"Num of QA with annotated dates: {numexamaples_w_dates_annotated}")

    return dataset


if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir

    train_json_path = os.path.join(input_dir, train_json)
    dev_json_path = os.path.join(input_dir, dev_json)

    train_dataset = qdmr_utils.read_drop_dataset(train_json_path)
    dev_dataset = qdmr_utils.read_drop_dataset(dev_json_path)

    new_train_dataset = date_comparison_aux_supervision(train_dataset)

    new_dev_dataset = date_comparison_aux_supervision(dev_dataset)

    with open(train_json_path, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(dev_json_path, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written datasets w/ date-compare aux-supervision")
