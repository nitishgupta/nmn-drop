from typing import List, Dict, Tuple
import json
import random
import os
from collections import defaultdict
import datasets.drop.constants as constants
from utils import util
import argparse

random.seed(100)

COUNT_NGRAMS = [
    "how many field goals did",
    "how many field goals were",
    "how many interceptions did",
    "how many passes",
    "how many rushing",
    "how many touchdown passes did",
    "how many touchdowns did the",
    "how many touchdowns were scored",
]


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def contains(small, big):
    starting_positions = []
    for i in range(len(big) - len(small) + 1):
        start = True
        for j in range(len(small)):
            if big[i + j] != small[j]:
                start = False
                break
        if start:
            starting_positions.append(i)
    return starting_positions


def preprocess_HowManyYardsCount_ques(dataset):
    """ Here we make synthetic data for counting questions.
        Idea is to generate semi-gold passage attentions and count-answer to train the count module.

        Each question we generate will be UNK (irrelevant), containing:
            - qytpe and program-supervision -- (count findPassageAttention_FAKE)
                This findPassageAttention_FAKE will not take question-attention as input, in fact the gold passage-attn
                as a side-arg.
            - passage_attention & count value
                We will generate semi-gold passage-attentions and count-values. These passage-attentions will be
                used in the program above.

        We generate these questions for passages that contain count-questions
    """

    new_dataset = {}
    total_ques = 0
    num_passages = len(dataset)

    num_of_gen_ques = 0
    count_distribution = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]

            tokenized_passage = passage_info[constants.tokenized_passage]
            passage_tokens = tokenized_passage.split(" ")

            if any(span in question_lower for span in COUNT_NGRAMS):
                attention, count, mask = make_count_instance(passage_tokens)
                if mask == 0:
                    continue

                answer = question_answer[constants.answer]
                answer["spans"] = []
                answer["number"] = str(count)

                question_answer[constants.answer_passage_spans] = []
                question_answer[constants.answer_question_spans] = []

                count_distribution[count] += 1

                query_id = question_answer[constants.query_id]
                query_id += "-synthetic-count"
                question_answer[constants.query_id] = query_id

                question_answer[constants.question] = "synthetic count"
                question_answer[constants.tokenized_question] = "synthetic count"
                question_answer[constants.cleaned_question] = "synthetic count"
                question_answer[constants.question_charidxs] = [0, 10]
                question_answer[constants.answer_type] = constants.NUM_TYPE

                question_answer[constants.qtype] = constants.SYN_COUNT_qtype
                question_answer[constants.program_supervised] = True

                question_answer[constants.pattn_supervised] = True
                question_answer[constants.passage_attn_supervision] = attention

                # Adding this so that the instance remains strongly supervised
                question_answer[constants.qattn_supervised] = True
                question_answer[constants.ques_attention_supervision] = [
                    [1.0, 1.0]
                ]  # Single attention vector of size=1

                # The final output of the program is enough to train, so no aux loss / execution supervision is needed
                # Still label as execution_supervised as it requires passing the pattn as side-arg
                question_answer[constants.exection_supervised] = True

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            num_of_gen_ques += len(new_qa_pairs)

    for k, v in count_distribution.items():
        count_distribution[k] = util.round_all((float(v) / num_of_gen_ques) * 100, 3)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages:{num_passages_after_prune}  Questions:{num_of_gen_ques}")
    print(f"CountDist: {count_distribution}")

    return new_dataset


def make_count_instance(passage_tokens: List[str]):
    """ output an attention, count_answer, mask. Mask is when we don;t find relevant spans """

    # We would like to count these spans
    relevant_spans = ["TD pass", "TD run", "touchdown pass", "field goal", "touchdown run"]
    num_relevant_spans = len(relevant_spans)

    attention = [0.0] * len(passage_tokens)

    # With 10% prob select no span
    count_zero_prob = random.random()
    if count_zero_prob < 0.1:
        return (attention, 0, 1)

    # Choose a particular type of span from relevant ones and find it's starting positions
    tries = 0
    starting_positions_in_passage = []
    while len(starting_positions_in_passage) == 0 and tries < 5:
        choosen_span = random.randint(0, num_relevant_spans - 1)
        span_tokens = relevant_spans[choosen_span].split(" ")
        starting_positions_in_passage = contains(span_tokens, passage_tokens)
        tries += 1

    # even after 5 tries, span to count not found. Return masked attention
    if len(starting_positions_in_passage) == 0:
        return attention, 0, 0

    # # TO save from infinite loop
    # count_zero_prob = random.random()
    # if count_zero_prob < 0.1:
    #     return attention, 0

    if len(starting_positions_in_passage) == 1:
        count = len(starting_positions_in_passage)
        starting_position = starting_positions_in_passage[0]
        attention[starting_position] = 1.0
        attention[starting_position + 1] = 1.0

    else:
        num_of_spans_found = len(starting_positions_in_passage)
        # Choose a subset of the starting_positions
        random.shuffle(starting_positions_in_passage)
        num_spans = random.randint(2, num_of_spans_found)
        num_spans = min(num_spans, 9)

        count = num_spans

        spread_len = random.randint(1, 3)

        chosen_starting_positions = starting_positions_in_passage[0:num_spans]
        for starting_position in chosen_starting_positions:
            attention[starting_position] = 1.0
            attention[starting_position + 1] = 1.0
            for i in range(1, spread_len + 1):
                prev_idx = starting_position - i
                if prev_idx >= 0:
                    attention[prev_idx] = 0.5
                next_idx = starting_position + 1 + i
                if next_idx < len(passage_tokens):
                    attention[next_idx] = 0.5

    # Adding noise and normalizing
    attention = [x + abs(random.gauss(0, 0.001)) for x in attention]
    attention_sum = sum(attention)
    attention = [float(x) / attention_sum for x in attention]

    return attention, count, 1


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

    new_train_dataset = preprocess_HowManyYardsCount_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsCount_ques(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=2)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=2)

    print("Written count dataset")
