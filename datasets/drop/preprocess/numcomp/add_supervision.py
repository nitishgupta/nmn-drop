import os
import json
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from nltk.corpus import stopwords
from datasets.drop import constants

NUMBER_COMPARISON = ["were there more", "were there fewer", "which age group", "which group"]

GT_OPERATOR = "MORE"
LT_OPERATOR = "LESS"

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])

greater_than_tokens = ["larger", "more", "largest", "bigger", "higher", "highest", "most", "greater"]
lesser_than_tokens = ["smaller", "fewer", "lowest", "smallest", "less", "least", "fewest", "lower"]

FIRST = "first"
SECOND = "second"


def getQuestionComparisonOperator(question_tokens: List[str]) -> str:
    for t in lesser_than_tokens:
        if t in question_tokens:
            return LT_OPERATOR

    for t in greater_than_tokens:
        if t in question_tokens:
            return GT_OPERATOR

    return None


def find_answer_event(answer_span: str, event1_tokens: List[str], event2_tokens: List[str]) -> str:
    ans_tokens = set(answer_span.split(" "))
    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = FIRST if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) else SECOND
    return ans_event


def getNumTokenIdxs(p_num_mens: List[Tuple[str, int, int]], passage_num_entidx: List[int]) -> List[int]:
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

    event1 = tokens[split_idx + 1 : or_idx]
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
    """
    passage_str = ""
    for idx, token in enumerate(passage_tokens):
        passage_str += f"{token}|{idx} "
    print(f"{passage_str}")
    print(f"{relevant_event_tokens}\n{relevant_passage_tokenidxs}\n{pruned_relevant_passage_tokenidxs}\n")
    """
    return pruned_relevant_passage_tokenidxs


def numInNeighborhood(
    relevant_passage_tokenidxs: List[int],
    passage_num_tokenidxs: List[int],
    passage_numtoken_entidxs: List[int],
    threshold,
):
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
    closest_date_entidx = max(set(closest_num_entidxs), key=closest_num_entidxs.count)

    if avg_distance_to_dates > threshold:
        return False, closest_date_entidx
    else:
        return True, closest_date_entidx


def addSupervision(input_json: str, output_json: str, output_txt: str, THRESHOLD=10) -> None:
    """ Questions of number comparison from ["were there more/fewer", "which age group", "which group"]

        For each question, try to find the following supervision:
            1. Question attention -- two attention values for the spans for which the numbers will be compared
                constants.ques_attention_supervision -- holds a 2-tuple of attentions
            2. Number supervision -- for the two spans, a passage_num_grounding and their values. These satisfy the
                coherency between the question operator and the values we ground to. Similar to depr_prune_date_comparison.py

                Fields added:
                constants.numcomp_qspan_num_groundings - 2-tuple containing number grounding
                constants.numcomp_qspan_num_values - 2-tuple containing number values

            3. QTYPE - constants.qtype is set to NUMCOMP_QTYPE

            4. Strongly Supervised - If 1. and 2. are found, then we set the strongly_supervised flag to True
                constants.strongly_supervised = True
    """

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, "r") as f:
        dataset = json.load(f)

    txtfile = open(output_txt, "w")

    new_dataset = {}

    num_original_questions = 0
    qoperator_dict = defaultdict(float)
    qoperator_undefined = 0
    num_grounding_unsuccessful = 0
    supervision_distribution = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        passage = passage_info[constants.tokenized_passage]
        passage_tokens = passage.split(" ")
        qa_pairs = passage_info[constants.qa_pairs]
        num_original_questions += len(qa_pairs)

        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_idxs = passage_info[constants.passage_num_entidx]
        passage_num_values = passage_info[constants.passage_num_normalized_values]

        (passage_num_tokenidxs, passage_numtoken_entidxs) = getNumTokenIdxs(passage_num_mens, passage_num_idxs)

        new_qa_pairs = []
        for qa_pair in qa_pairs:
            question = qa_pair[constants.tokenized_question]
            question_tokens = question.split(" ")
            answer_span: str = qa_pair[constants.answer]["spans"][0]
            qlen = len(question_tokens)

            qoperator = getQuestionComparisonOperator(question_tokens)
            if qoperator is None:
                qoperator_undefined += 1
                program_supervision = False
            else:
                program_supervision = True
                qoperator_dict[qoperator] += 1

            """ QUESTION ATTENTION SUPERVISION """
            # Exclusive ends
            attention1 = [0.0] * qlen
            attention2 = [0.0] * qlen
            ques_spans = questionAttns(question)
            if ques_spans and program_supervision:
                # These span ends are exclusive
                span1, span2 = ques_spans
                for i in range(span1[0], span1[1]):
                    attention1[i] = 1.0
                for i in range(span2[0], span2[1]):
                    attention2[i] = 1.0

            annotated_qtokens = (
                question_tokens[0 : span1[0]]
                + ["[["]
                + question_tokens[span1[0] : span1[1]]
                + ["]]"]
                + question_tokens[span1[1] : span2[0]]
                + ["[["]
                + question_tokens[span2[0] : span2[1]]
                + ["]]"]
                + question_tokens[span2[1] :]
            )
            annotated_qtxt = " ".join(annotated_qtokens)

            if sum(attention1) > 0 and sum(attention2) > 0:
                qattn_supervision = True
            else:
                qattn_supervision = False

            """ Keeping reverse question attentions """
            qa_pair[constants.ques_attention_supervision] = (attention2, attention1)

            """ NUMBER GROUNDING SUPERVISION """
            if program_supervision and qattn_supervision:
                span1, span2 = ques_spans
                span1_tokens = question_tokens[span1[0] : span1[1]]
                span2_tokens = question_tokens[span2[0] : span2[1]]

                # List of tokenidxs in passage that is a rough grounding for event 1/2
                event1_passage_tokenidxs: List[int] = matchEventToPassage(span1_tokens, passage_tokens)
                event2_passage_tokenidxs: List[int] = matchEventToPassage(span2_tokens, passage_tokens)

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
                    num_grounding_unsuccessful += 1
                    execution_supervision = False

                else:
                    # First or Second
                    answer_event = find_answer_event(answer_span, span1_tokens, span2_tokens)
                    # Need to check if the values are coherent with the operator and the answer
                    grounded_answer_num_value = value1 if answer_event == FIRST else value2
                    grounded_other_num_value = value2 if answer_event == FIRST else value1

                    if qoperator == LT_OPERATOR:
                        if grounded_answer_num_value >= grounded_other_num_value:
                            num_grounding_unsuccessful += 1
                            execution_supervision = False
                    if qoperator == GT_OPERATOR:
                        if grounded_answer_num_value <= grounded_other_num_value:
                            num_grounding_unsuccessful += 1
                            execution_supervision = False

                span1_num_grounding = [0] * len(passage_num_values)
                span2_num_grounding = [0] * len(passage_num_values)

                if execution_supervision is True:
                    span2_num_grounding[event2_num_idx] = 1
                    span1_num_grounding[event1_num_idx] = 1

                    txtfile.write(f"{annotated_qtxt} -- {qoperator}\n")
                    txtfile.write(f"{passage}\n")
                    event1_tokens = [passage_tokens[x] for x in event1_passage_tokenidxs]
                    event2_tokens = [passage_tokens[x] for x in event2_passage_tokenidxs]
                    txtfile.write(f"{event1_tokens} - {value1}\n")
                    txtfile.write(f"{event2_tokens} {value2}\n\n")

                # else:
                #     txtfile.write(f"{annotated_qtxt} -- {qoperator}\n")
                #     txtfile.write(f"{passage}\n")
                #     event1_tokens = [passage_tokens[x] for x in event1_passage_tokenidxs]
                #     event2_tokens = [passage_tokens[x] for x in event2_passage_tokenidxs]
                #     txtfile.write(f"{event1_tokens}  {event2_tokens}")
                #     txtfile.write(f"{value1}  {value2}\n\n")

                """ Storing reversed supervision since it helps a little """
                qa_pair[constants.qspan_numgrounding_supervision] = (span2_num_grounding, span1_num_grounding)
                qa_pair[constants.qspan_numvalue_supervision] = (value2, value1)
            else:
                execution_supervision = False
                span1_num_grounding = [0] * len(passage_num_values)
                span2_num_grounding = [0] * len(passage_num_values)
                qa_pair[constants.qspan_numgrounding_supervision] = (span2_num_grounding, span1_num_grounding)
                qa_pair[constants.qspan_numvalue_supervision] = (-1, -1)

            """ QTYPE SUPERVISION """
            if program_supervision is True:
                qa_pair[constants.qtype] = constants.NUMCOMP_QTYPE
            qa_pair[constants.program_supervised] = program_supervision
            qa_pair[constants.qattn_supervised] = qattn_supervision
            qa_pair[constants.exection_supervised] = execution_supervision

            if program_supervision and qattn_supervision and execution_supervision:
                strongly_supervised = True
            else:
                strongly_supervised = False

            qa_pair[constants.strongly_supervised] = strongly_supervised

            supervision_distribution["program_supervision"] += 1 if program_supervision else 0
            supervision_distribution["qattn_supervision"] += 1 if qattn_supervision else 0
            supervision_distribution["execution_supervision"] += 1 if execution_supervision else 0
            supervision_distribution["strongly_supervised"] += 1 if strongly_supervised else 0

            new_qa_pairs.append(qa_pair)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info

    with open(output_json, "w") as outf:
        json.dump(new_dataset, outf, indent=4)

    txtfile.close()

    print(f"Number of input passages: {len(dataset)}\nNumber of input QA pairs: {num_original_questions}")
    print(f"Number of output passages: {len(new_dataset)}")
    print(f"Num grounding unsuccess: {num_grounding_unsuccessful}")
    print(f"{qoperator_dict}")
    print(supervision_distribution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"
    train_txt = "train.txt"
    dev_txt = "dev.txt"

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)

    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)
    output_trn_txt = os.path.join(output_dir, train_txt)
    output_dev_txt = os.path.join(output_dir, dev_txt)

    # args.input_json --- is the raw json from the DROP dataset
    addSupervision(input_json=input_trnfp, output_json=output_trnfp, output_txt=output_trn_txt)

    # args.input_json --- is the raw json from the DROP dataset
    addSupervision(input_json=input_devfp, output_json=output_devfp, output_txt=output_dev_txt)
