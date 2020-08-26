from typing import List, Tuple, Dict, Union
import os
import json
import argparse
from enum import Enum
from collections import defaultdict
from nltk.corpus import stopwords
from datasets.drop import constants

from semqa.utils import qdmr_utils

NUMBER_COMPARISON = ["were there more", "were there fewer", "which age group", "which group"]

GT_OPERATOR = "MORE"
LT_OPERATOR = "LESS"

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])

greater_than_tokens = ["larger", "more", "largest", "bigger", "higher", "highest", "most", "greater"]
lesser_than_tokens = ["smaller", "fewer", "lowest", "smallest", "less", "least", "fewest", "lower"]

FIRST = "first"
SECOND = "second"


def number_comparison_filter(question: str) -> bool:
    """ Figures whether the question is a number comparison or not. """
    question_lower = question.lower()
    football_ques_spans = ["first half", "second half", "quarter", "touchdown", "field goals"]
    relevant = True
    if any(span in question_lower for span in NUMBER_COMPARISON):
        or_split = question_lower.split(" or ")
        if len(or_split) != 2:
            relevant = False
        comma_split = question_lower.split(",")
        if len(comma_split) > 2:
            relevant = False
        # were there more / fewer -- remove these difficult football questions
        if any(span in question_lower for span in football_ques_spans):
            relevant = False
    else:
        relevant = False
    return relevant


class OperatorType(Enum):
    LT_OPERATOR = 1
    GT_OPERATOR = 2


class AnswerEvent(Enum):
    FIRST = 1
    SECOND = 2


def getQuestionComparisonOperator(question_tokens: List[str]) -> Union[OperatorType, None]:
    """ Figure out which kind of comparison is needed; return GT_OPERATOR or LT_OPERATOR"""
    for t in lesser_than_tokens:
        if t in question_tokens:
            return OperatorType.LT_OPERATOR
    for t in greater_than_tokens:
        if t in question_tokens:
            return OperatorType.GT_OPERATOR
    return None


def get_program_supervision(operator: OperatorType) -> qdmr_utils.Node:
    assert operator is not None
    num_lt = "(select_passagespan_answer (compare_num_lt select_passage select_passage))"
    num_gt = "(select_passagespan_answer (compare_num_gt select_passage select_passage))"
    lisp_program = num_lt if operator == OperatorType.LT_OPERATOR else num_gt
    node: qdmr_utils.Node = qdmr_utils.nested_expression_to_tree(qdmr_utils.lisp_to_nested_expression(lisp_program))
    return node


def find_answer_event(answer_span: str, event1_tokens: List[str], event2_tokens: List[str]) -> AnswerEvent:
    """ Figure which of the event is the answer, FIRST or SECOND.
    This is doneby fuzzy matching answer string with the two events' tokens
    """
    ans_tokens = set(answer_span.split(" "))
    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = AnswerEvent.FIRST if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) else \
        AnswerEvent.SECOND
    return ans_event


def getNumTokenIdxs(p_num_mens: List[Tuple[str, int, float]],
                    passage_num_entidx: List[int]) -> Tuple[List[int], List[int]]:
    """ Lists telling which tokens are numbers, and num_ent_idx they ground to. """
    passage_num_tokens = []
    passage_numtoken_entidxs = []
    for num_men, num_idx in zip(p_num_mens, passage_num_entidx):
        num_token_idx = num_men[1]
        passage_num_tokens.append(num_token_idx)
        passage_numtoken_entidxs.append(num_idx)

    assert len(passage_num_tokens) == len(passage_numtoken_entidxs)
    return passage_num_tokens, passage_numtoken_entidxs


def questionAttns(qstr: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ For the "which group" questions output the two-relevant question attentions """
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
        if "more" in tokens:
            split_idx = tokens.index("more")
        elif "fewer" in tokens:
            split_idx = tokens.index("fewer")
        elif "last" in tokens:
            split_idx = tokens.index("last")
        elif "later" in tokens:
            split_idx = tokens.index("later")
        elif "larger" in tokens:
            split_idx = tokens.index("larger")
        else:
            split_idx = -1

    if split_idx == -1:
        print(f"Cannot split -- {qstr} {split_idx} {or_idx}")
        return None
    event1_span = (split_idx + 1, or_idx)
    return event1_span, event2_span


def difference_in_successive_terms(l: List[int]):
    sum_of_differences = 0
    for i in range(len(l) - 1):
        sum_of_differences += l[i + 1] - l[i]
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
        passage_token_span = relevant_passage_tokenidxs[i : i + len_event_span]
        sum_of_token_diffs = difference_in_successive_terms(passage_token_span)
        if sum_of_token_diffs < best_diff:
            best_start_point = i
            best_diff = sum_of_token_diffs
    pruned_relevant_passage_tokenidxs = relevant_passage_tokenidxs[best_start_point : best_start_point + len_event_span]
    return pruned_relevant_passage_tokenidxs


def numInNeighborhood(relevant_passage_tokenidxs: List[int], passage_num_tokenidxs: List[int],
                      passage_numtoken_entidxs: List[int], threshold: int):
    """ Given a list of relevant-passage-tokens, and list of date-tokens in the passage, figure out -
        if there's a date in the neighborhood of the relevant tokens
        For each passage-token, first find the min-distance to a date token. Then find the min-amongst that.
        If this distance crosses a threshold, then a date is not considered in the neighborhood of the passage-tokens
    """
    distance_to_nums = []
    closest_num_entidxs = []
    for tokenidx in relevant_passage_tokenidxs:
        min_distance_to_num = 10000
        closest_numidx = -1
        for num_tokenidx, num_idx in zip(passage_num_tokenidxs, passage_numtoken_entidxs):
            # Since some spans are made up of numbers themselves (age groups) don't consider those numbers
            if num_tokenidx in relevant_passage_tokenidxs:
                continue
            # Only keeping numbers that appear before the span as in these questions it seems to better ground
            if (num_tokenidx - tokenidx) > 0:
                dis = 1000
            else:
                dis = abs(num_tokenidx - tokenidx)

            if dis == 0:
                dis = 1000

            if dis < min_distance_to_num:
                min_distance_to_num = dis
                closest_numidx = num_idx
        distance_to_nums.append(min_distance_to_num)
        closest_num_entidxs.append(closest_numidx)

    if len(distance_to_nums) == 0:
        return False, -1

    avg_distance_to_dates = float(sum(distance_to_nums)) / len(distance_to_nums)
    # Mode
    closest_num_entidx = max(set(closest_num_entidxs), key=closest_num_entidxs.count)

    if avg_distance_to_dates > threshold:
        return False, closest_num_entidx
    else:
        return True, closest_num_entidx



def pruneNumCompQuestions(dataset, THRESHOLD: int = 10) -> Dict:
    """ Prune dataset to only contain questions that qualify after certain NUM comparison question tests.
        Currently only keeping questions with a single passage SpanType answer.
    """
    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)

    numexamaples_w_num_annotated = 0

    qoperator_dict = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        passage_num_mens: List[Tuple[str, int, float]] = passage_info[constants.passage_num_mens]
        passage_num_idxs = passage_info[constants.passage_num_entidx]
        passage_num_values: List[float] = passage_info[constants.passage_num_normalized_values]
        (passage_num_tokenidxs, passage_numtoken_entidxs) = getNumTokenIdxs(passage_num_mens, passage_num_idxs)

        qa_pairs = passage_info[constants.qa_pairs]
        relevant_qa_pairs = []

        new_qa_pairs = []
        for question_answer in qa_pairs:
            total_ques += 1

            # question_tokenized_text = question_answer[constants.tokenized_question]
            question_tokens: List[str] = question_answer[constants.question_tokens]
            question_tokenized_text = " ".join(question_tokens)
            answer_annotation = question_answer[constants.answer]

            # Number Comparison questions we care about
            if not number_comparison_filter(question_tokenized_text):
                continue
            # Only SPAN type questions
            if question_answer[constants.answer_type] != constants.SPAN_TYPE:
                continue
            qoperator: Union[OperatorType, None] = getQuestionComparisonOperator(question_tokens)
            if qoperator is None:
                continue
            else:
                qoperator_dict[qoperator] += 1

            ques_event_spans: Tuple[Tuple[int, int], Tuple[int, int]] = questionAttns(" ".join(question_tokens))
            answer_span: str = answer_annotation["spans"][0]

            if ques_event_spans is None:
                continue

            # For questions with identified events, we'll try to ground dates
            event1_span, event2_span = ques_event_spans
            event1_tokens = question_tokens[event1_span[0]:event1_span[1]]
            event2_tokens = question_tokens[event2_span[0]:event2_span[1]]

            """
            # NUMBER GROUNDING SUPERVISION
            # List of tokenidxs in passage that is a rough grounding for event 1/2
            event1_passage_tokenidxs: List[int] = matchEventToPassage(event1_tokens, passage_tokens)
            event2_passage_tokenidxs: List[int] = matchEventToPassage(event2_tokens, passage_tokens)

            num_near_event1, event1_num_idx = numInNeighborhood(
                event1_passage_tokenidxs, passage_num_tokenidxs, passage_numtoken_entidxs, threshold=THRESHOLD
            )

            num_near_event2, event2_num_idx = numInNeighborhood(
                event2_passage_tokenidxs, passage_num_tokenidxs, passage_numtoken_entidxs, threshold=THRESHOLD
            )
            if num_near_event1:
                value1 = passage_num_values[event1_num_idx]
            else:
                value1 = -1000000
            if num_near_event2:
                value2 = passage_num_values[event2_num_idx]
            else:
                value2 = -1000000

            execution_supervision = True
            if value1 == -1000000 or value2 == -1000000:
                execution_supervision = False
            else:
                # First or Second
                answer_event: AnswerEvent = find_answer_event(answer_span, event1_tokens, event2_tokens)
                # Need to check if the values are coherent with the operator and the answer
                grounded_answer_num_value = value1 if answer_event == AnswerEvent.FIRST else value2
                grounded_other_num_value = value2 if answer_event == AnswerEvent.FIRST else value1

                if qoperator == OperatorType.LT_OPERATOR:
                    if grounded_answer_num_value >= grounded_other_num_value:
                        execution_supervision = False
                if qoperator == OperatorType.GT_OPERATOR:
                    if grounded_answer_num_value <= grounded_other_num_value:
                        execution_supervision = False
            """

            # Compiling gold - program
            program_node: qdmr_utils.Node = get_program_supervision(operator=qoperator)
            num_compare_node = program_node.children[0]
            find1_node, find2_node = num_compare_node.children[0], num_compare_node.children[1]
            find1_node.string_arg = " ".join(event1_tokens)
            find2_node.string_arg = " ".join(event2_tokens)
            event1_token_idxs = list(range(event1_span[0], event1_span[1]))
            event1_attn = [1 if i in event1_token_idxs else 0 for i in range(len(question_tokens))]
            event2_token_idxs = list(range(event2_span[0], event2_span[1]))
            event2_attn = [1 if i in event2_token_idxs else 0 for i in range(len(question_tokens))]
            find1_node.supervision["question_attention_supervision"] = event1_attn
            find2_node.supervision["question_attention_supervision"] = event2_attn

            """ Avoiding number-grounding supervision; this tends to be noisy and hurts performance
            question_answer[constants.execution_supervised] = False
            if execution_supervision is True:
                num_compare_node.supervision["num1_entidxs"] = [event1_num_idx]
                num_compare_node.supervision["num2_entidxs"] = [event2_num_idx]
                question_answer[constants.execution_supervised] = True
                numexamaples_w_num_annotated += 1
            """

            question_answer[constants.program_supervision] = program_node.to_dict()

            new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with annotated numbers: {numexamaples_w_num_annotated}")

    return new_dataset


if __name__ == "__main__":
    print("\nPruning and heuristic annotation for num-comp questions")

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

    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = qdmr_utils.read_drop_dataset(input_trnfp)
    dev_dataset = qdmr_utils.read_drop_dataset(input_devfp)

    new_train_dataset = pruneNumCompQuestions(train_dataset)

    new_dev_dataset = pruneNumCompQuestions(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written augmented datasets")
