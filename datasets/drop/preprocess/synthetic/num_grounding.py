from typing import List, Dict, Tuple
import json
import random
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

random.seed(100)

COUNT_NGRAMS = ["how many field goals did", "how many field goals were",
                "how many interceptions did", "how many passes", "how many rushing",
                "how many touchdown passes did", "how many touchdowns did the",
                "how many touchdowns were scored"]


RELEVANT_TOKENS = ['TD', 'pass', 'run', 'field', 'goal', 'touchdown']


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset

def generateNumGroundingQues(dataset):
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

    BACKWARD_WINDOW_SIZE = 5
    FORWARD_WINDOW_SIZE = 10

    for passage_id, passage_info in dataset.items():

        tokenized_passage = passage_info[constants.tokenized_passage]
        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_values = passage_info[constants.passage_num_normalized_values]

        # Keeping passage to a maximum of 400 tokens to avoid conflicts later
        passage_tokens = tokenized_passage.split(' ')

        new_qa_pairs = []

        token_number_idx_pairs = []

        # The num corresponding to the first relevant-token will be the answer to fake question, keeping track
        min_token_idx = 1000000
        answer_number_value = -1
        # Only keeping values that are within 400 tokens so we don't have to prune in the reader.
        for (str_val, num_token_idx, num_value) in passage_num_mens:
            if num_token_idx >= 400:
                continue
            starting_limit = max(0, num_token_idx - BACKWARD_WINDOW_SIZE)      # Inclusive
            ending_limit = min(len(passage_tokens), 400, num_token_idx + FORWARD_WINDOW_SIZE)   # Exclusive
            for i in range(starting_limit, ending_limit):
                if passage_tokens[i] in RELEVANT_TOKENS and num_value in passage_num_values:
                    token_number_idx_pairs.append((i, num_token_idx))
                    if num_token_idx < min_token_idx:
                        min_token_idx = num_token_idx
                        answer_number_value = num_value

        if answer_number_value == -1:
            continue


        # Now we have token_number_idx_pairs: [ (tokenidx, numberidx) ]
        # Sort according to tokenidx, and make a fake question with
        #   1. passage-attention on first tokenidx and corresponding number as answer

        token_number_idx_pairs = sorted(token_number_idx_pairs, key=lambda x: x[0])

        answer_token_idx = token_number_idx_pairs[0][0]
        attention = [0.0] * len(passage_tokens)
        attention[answer_token_idx] = 1.0


        question_answer = passage_info[constants.qa_pairs][0]

        answer = question_answer[constants.answer]
        answer["spans"] = []
        answer["number"] = str(answer_number_value)

        question_answer[constants.answer_passage_spans] = []
        question_answer[constants.answer_question_spans] = []

        query_id = question_answer[constants.query_id]
        query_id += "-synthetic-numgrounding"
        question_answer[constants.query_id] = query_id

        question_answer[constants.question] = "synthetic number"
        question_answer[constants.tokenized_question] = "synthetic number"
        question_answer[constants.cleaned_question] = "synthetic number"
        question_answer[constants.question_charidxs] = [0, 10]
        question_answer[constants.answer_type] = constants.NUM_TYPE

        question_answer[constants.SYN_NUMGROUND_METADATA] = token_number_idx_pairs

        question_answer[constants.qtype] = constants.SYN_NUMGROUND_qtype
        question_answer[constants.program_supervised] = True

        question_answer[constants.pattn_supervised] = True
        question_answer[constants.passage_attn_supervision] = attention

        # Adding this so that the instance remains strongly supervised
        question_answer[constants.qattn_supervised] = True
        question_answer[constants.ques_attention_supervision] = [[1.0, 1.0]]     # Single attention vector of size=1

        # The final output of the program is enough to train, so no aux loss / execution supervision is needed
        # Still label as execution_supervised as it requires passing the pattn as side-arg
        question_answer[constants.exection_supervised] = True

        new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            num_of_gen_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages:{num_passages_after_prune}  Questions:{num_of_gen_ques}")

    return new_dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

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

    new_train_dataset = generateNumGroundingQues(train_dataset)

    new_dev_dataset = generateNumGroundingQues(dev_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=2)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=2)

    print("Written count dataset")

