from typing import List, Dict, Tuple
import json
import nltk
from nltk.corpus import stopwords
import copy
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants
from semqa.domain_languages.drop.drop_language import Date
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])

FIRST="first"
SECOND="second"


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def getQuestionComparisonOperator(question_tokens: List[str]) -> str:
    # Correct if Attn1 is first event
    lesser_tokens = ['first', 'earlier', 'forst', 'firts']
    greater_tokens = ['later', 'last', 'second']

    for t in lesser_tokens:
        if t in question_tokens:
            return FIRST

    for t in greater_tokens:
        if t in question_tokens:
            return SECOND

    return SECOND


def getDateTokenIdxs(p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
                     p_date_entidxs: List[int]) -> List[int]:
    # List of token idxs that are dates
    passage_date_tokens = []
    passage_datetoken_entidxs = []
    for date_men, date_idx in zip(p_date_mens, p_date_entidxs):
        (s, e) = date_men[1]
        passage_date_tokens.extend([x for x in range(s, e+1)])
        passage_datetoken_entidxs.extend([date_idx for x in range(s, e + 1)])

    assert len(passage_date_tokens) == len(passage_datetoken_entidxs)
    return passage_date_tokens, passage_datetoken_entidxs


def quesEvents(qstr) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ Returns (start, end) span tuples for event1 and event2 in the question """
    or_split = qstr.split(' or ')
    if len(or_split) != 2:
        return None

    tokens = qstr.split(' ')

    or_idx = tokens.index('or')
    # Last token is ? which we don't want to attend to
    event2 = tokens[or_idx + 1 : len(tokens) - 1]
    event2_span = (or_idx + 1, len(tokens) - 1)

    # Gets first index of the item
    try:
        comma_idx = tokens.index(',')
    except:
        comma_idx = 100000
    try:
        colon_idx = tokens.index(':')
    except:
        colon_idx = 100000

    try:
        hyphen_idx = tokens.index('-')
    except:
        hyphen_idx = 100000

    split_idx = min(comma_idx, colon_idx, hyphen_idx)

    if split_idx == 100000 or (or_idx - split_idx <= 1):
        # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
        if 'first' in tokens:
            split_idx = tokens.index('first')
        elif 'second' in tokens:
            split_idx = tokens.index('second')
        elif 'last' in tokens:
            split_idx = tokens.index('last')
        elif 'later' in tokens:
            split_idx = tokens.index('later')
        else:
            split_idx = -1

    assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"

    event1 = tokens[split_idx + 1: or_idx]
    event1_span = (split_idx + 1, or_idx)

    return event1_span, event2_span


def find_answer_event(answer_span: str, event1_tokens: List[str], event2_tokens: List[str]) -> str:
    ans_tokens = set(answer_span.split(' '))
    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = FIRST if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) else SECOND
    return ans_event


def difference_in_successive_terms(l: List[int]):
    sum_of_differences = 0
    for i in range(len(l) - 1):
        sum_of_differences += l[i+1] - l[i]
    return sum_of_differences


def matchEventToPassage(event_tokens: List[str], passage_tokens: List[str]) -> List[int]:
    """ Match a given event's tokens and find relevant tokens in the passage. """
    relevant_event_tokens = [t.lower() for t in event_tokens if t.lower() not in STOP_WORDS]

    relevant_passage_tokenidxs = []
    for (idx, passage_token) in enumerate(passage_tokens):
        if passage_token.lower() in relevant_event_tokens:
            relevant_passage_tokenidxs.append(idx)

    # Since event tokens can match spuriously at many locations -
    # Here we try to find the tightest span, i.e. a span that is the same length as event_span and contains close-tokens
    len_event_span = len(relevant_event_tokens)
    best_diff = 100000
    best_start_point = 0
    for i in range(0, len(relevant_passage_tokenidxs) - len_event_span + 1):
        passage_token_span = relevant_passage_tokenidxs[i:i+len_event_span]
        sum_of_token_diffs = difference_in_successive_terms(passage_token_span)
        if sum_of_token_diffs < best_diff:
            best_start_point = i
            best_diff = sum_of_token_diffs

    pruned_relevant_passage_tokenidxs = relevant_passage_tokenidxs[best_start_point:best_start_point + len_event_span]

    """
    passage_str = ""
    for idx, token in enumerate(passage_tokens):
        passage_str += f"{token}|{idx} "
    print(f"{passage_str}")
    print(f"{relevant_event_tokens}\n{relevant_passage_tokenidxs}\n{pruned_relevant_passage_tokenidxs}\n")
    """

    return pruned_relevant_passage_tokenidxs

def dateInNeighborhood(passage_tokenidxs: List[int], passage_date_tokenidxs: List[int],
                       passage_datetoken_entidxs: List[int], threshold = 20):
    """ Given a list of relevant-passage-tokens, and list of date-tokens in the passage, figure out -
        if there's a date in the neighborhood of the relevant tokens
        For each passage-token, first find the min-distance to a date token. Then find the min-amongst that.
        If this distance crosses a threshold, then a date is not considered in the neighborhood of the passage-tokens
    """

    distance_to_dates = []
    closest_date_entidxs  = []
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

    avg_distance_to_dates = float(sum(distance_to_dates))/len(distance_to_dates)
    # Mode
    closest_date_entidx = max(set(closest_date_entidxs), key=closest_date_entidxs.count)

    if avg_distance_to_dates > threshold:
        return False, closest_date_entidx
    else:
        return True, closest_date_entidx


def pruneDateQuestions(dataset, weakdate: bool = False):
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

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)

    numexamaples_w_dates_annotated = 0

    for passage_id, passage_info in dataset.items():
        passage = passage_info[constants.passage]
        passage_tokens: List[str] = passage.split(' ')
        p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = passage_info[constants.passage_date_mens]
        p_date_entidxs = passage_info[constants.passage_date_entidx]
        p_date_values: List[Tuple] = passage_info[constants.passage_date_normalized_values]

        passage_date_tokenidxs, passage_datetoken_entidxs = getDateTokenIdxs(p_date_mens, p_date_entidxs)

        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1
            question_tokenized_text = question_answer[constants.question]
            question_tokens: List[str] = question_tokenized_text.split(' ')
            answer_annotation = question_answer[constants.answer]
            answer_span: str = answer_annotation["spans"][0]

            event_spans = quesEvents(question_tokenized_text)
            if event_spans is None:
                continue
            event1_span, event2_span = event_spans

            event1_tokens = question_tokens[event1_span[0]: event1_span[1]]
            event2_tokens = question_tokens[event2_span[0]: event2_span[1]]

            # List of tokenidxs in passage that is a rough grounding for event 1/2
            event1_passage_tokenidxs: List[int] = matchEventToPassage(event1_tokens, passage_tokens)
            event2_passage_tokenidxs: List[int] = matchEventToPassage(event2_tokens, passage_tokens)

            date_near_event1, event1_date_idx = dateInNeighborhood(event1_passage_tokenidxs, passage_date_tokenidxs,
                                                                   passage_datetoken_entidxs, threshold=THRESHOLD)
            date_near_event2, event2_date_idx = dateInNeighborhood(event2_passage_tokenidxs, passage_date_tokenidxs,
                                                                   passage_datetoken_entidxs, threshold=THRESHOLD)


            question_operator = getQuestionComparisonOperator(question_tokens)
            answer_event = find_answer_event(answer_span, event1_tokens, event2_tokens)

            # Adding a tuple of zero vectors and empty_values to later store the date grounding of the two ques-events
            event1_date_grounding = [0] * len(p_date_values)
            event2_date_grounding = [0] * len(p_date_values)
            event1_date_value = [-1, -1, -1]
            event2_date_value = [-1, -1, -1]

            if weakdate:
                # First find if the date groundings are relevant, checks:
                # 1. Date grounding shouldn't be -1
                # 2. Shouldn't be equal
                # 3. The date grounding should be consistent with answer annotation
                keep_dates = True
                if not (event1_date_idx == -1 or event2_date_idx == -1 or (event1_date_idx == event2_date_idx)):
                    answer_event_date = event1_date_idx if answer_event == FIRST else event2_date_idx
                    other_event_date = event2_date_idx if answer_event == FIRST else event1_date_idx

                    answer_event_date = Date(year=p_date_values[answer_event_date][2],
                                             month=p_date_values[answer_event_date][1],
                                             day=p_date_values[answer_event_date][0])
                    other_event_date = Date(year=p_date_values[other_event_date][2],
                                            month=p_date_values[other_event_date][1],
                                            day=p_date_values[other_event_date][0])

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

                if keep_dates:
                    numexamaples_w_dates_annotated += 1
                    event1_date_grounding[event1_date_idx] = 1
                    event1_date_value = p_date_values[event1_date_idx]
                    event2_date_grounding[event2_date_idx] = 1
                    event2_date_value = p_date_values[event2_date_idx]

            question_answer[constants.datecomp_ques_event_date_groundings] = [event1_date_grounding,
                                                                              event2_date_grounding]
            question_answer[constants.datecomp_ques_event_date_values] = [event1_date_value,
                                                                          event2_date_value]

            if date_near_event1 and date_near_event2:
                new_qa_pairs.append(question_answer)
            # else:
            # print(question_tokenized_text)
            # print(passage)
            # print(p_date_values)
            # print(f"{date_near_event1}  {event1_date_idx}")
            # print(f"{date_near_event2}  {event2_date_idx}")
            # print()
        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with annotated dates: {numexamaples_w_dates_annotated}")

    return new_dataset


if __name__=='__main__':
    # input_dir = "date"
    # output_dir = "date_prune_weakdate"
    #
    # trnfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_train.json"
    # devfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_dev.json"
    #
    # out_trfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_train.json"
    # out_devfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_dev.json"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_trnfp')
    parser.add_argument('--input_devfp')
    parser.add_argument('--output_trnfp')
    parser.add_argument('--output_devfp')
    parser.add_argument('--no_weakdate', action='store_true', default=False)
    args = parser.parse_args()

    weakdate = not args.no_weakdate

    input_trnfp = args.input_trnfp
    input_devfp = args.input_devfp
    output_trnfp = args.output_trnfp
    output_devfp = args.output_devfp

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = pruneDateQuestions(train_dataset, weakdate=weakdate)
    new_dev_dataset = pruneDateQuestions(dev_dataset, weakdate=weakdate)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written augmented datasets")

