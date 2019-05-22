from typing import List
import json
from nltk.corpus import stopwords
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

COUNT_TRIGRAMS = ["how many field goals did", "how many field goals were",
                  "how many interceptions did", "how many passes", "how many rushing",
                  "how many touchdown passes did", "how many touchdowns did the",
                  "how many touchdowns were scored"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def filter_questionattention(tokenized_queslower: str):
    """ Here we'll annotate questions with one/two attentions depending on if the program type is
        1. find(QuestionAttention)
        2. filter(QuestionAttention, find(QuestionAttention))
    """
    question_lower = tokenized_queslower
    question_tokens: List[str] = question_lower.split(' ')
    qlen = len(question_tokens)

    if "how many field goals were" in question_lower:
        # Non-filter question
        if question_lower in ["how many field goals were kicked ?", "how many field goals were kicked in the game ?",
                              "how many field goals were made ?", "how many field goals were made in the game ?",
                              "how many field goals were scored ?", "how many field goals were scored in the game ?",
                              "how many field goals were in the game ?",
                              "how many field goals were made in this game ?",
                              "how many field goals were in this game?"]:
            qtype = constants.COUNT_find_qtype
            question_attention_find = [2, 3]       # Inclusive
            question_attention_filter = None

        else:
            # QAttn1 (filter) is everything after "how many f gs were" until ?. QAttn2 (find) is F Gs, i.e. [2, 3]
            qtype = constants.COUNT_filter_find_qtype
            question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?
            question_attention_find = [2, 3]

    elif "how many field goals did" in question_lower:
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 3]
        question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?

    elif "how many interceptions did" in question_lower:
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 2]
        question_attention_filter = [4, qlen - 2]  # skipping first 4 tokens and ?

    elif "how many passes" in question_lower:
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 2]
        question_attention_filter = [4, qlen - 2]  # skipping first 4 tokens and ?

    elif "how many rushing" in question_lower:
        # Most questions are How many rushing touchdowns/yards were / did
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 3]
        question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?

    elif "how many touchdown passes did" in question_lower:
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 3]
        question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?

    elif "how many touchdowns did the" in question_lower:
        qtype = constants.COUNT_filter_find_qtype
        question_attention_find = [2, 2]
        question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?

    elif "how many touchdowns were scored" in question_lower:
        if question_lower in ["how many touchdowns were scored in the game ?", "how many touchdowns were scored ?",
                              "how many touchdowns were scored in total ?"]:
            qtype = constants.COUNT_find_qtype
            question_attention_find = [2, 2]  # Inclusive
            question_attention_filter = None
        else:
            qtype = constants.COUNT_filter_find_qtype
            question_attention_find = [2, 2]
            question_attention_filter = [5, qlen - 2]  # skipping first 5 tokens and ?

    return qtype, question_attention_filter, question_attention_find


def convert_span_to_attention(qlen, span):
    # span is inclusive on both ends
    qattn = [0.0] * qlen
    for x in range(span[0], span[1] + 1):
        qattn[x] = 1.0

    return qattn


def preprocess_HowManyYardsCount_ques(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_attn = 0
    num_passages = len(dataset)
    qtype_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]
            tokens = tokenized_ques.split(' ')
            qlen = len(tokens)
            if any(span in question_lower for span in COUNT_TRIGRAMS):
                (qtype,
                 question_attention_filter_span,
                 question_attention_find_span) = filter_questionattention(tokenized_queslower=tokenized_ques.lower())

                if question_attention_filter_span is not None:
                    filter_qattn = convert_span_to_attention(qlen, question_attention_filter_span)
                else:
                    filter_qattn = None

                find_qattn = convert_span_to_attention(qlen, question_attention_find_span)

                question_answer[constants.qtype] = qtype
                question_answer[constants.program_supervised] = True
                qtype_dist[qtype] += 1

                # Also adding qattn -- everything apart from the first two tokens
                if filter_qattn is not None:
                    question_answer[constants.ques_attention_supervision] = [filter_qattn, find_qattn]
                else:
                    question_answer[constants.ques_attention_supervision] = [find_qattn]
                question_answer[constants.qattn_supervised] = True
                questions_w_attn += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Ques with attn: {questions_w_attn}")
    print(f"QType distribution: {qtype_dist}")


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

    new_train_dataset = preprocess_HowManyYardsCount_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsCount_ques(dev_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")

