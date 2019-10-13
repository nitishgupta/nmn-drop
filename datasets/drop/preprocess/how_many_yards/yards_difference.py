from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def get_question_type_and_attention(question_tokens: List[str]):
    numbertype_to_qtype_mapping = {
        ("num", "num"): constants.DIFF_NUMNUM_qtype,
        ("num", "min"): constants.DIFF_NUMMIN_qtype,
        ("num", "max"): constants.DIFF_NUMMAX_qtype,
        ("min", "num"): constants.DIFF_MINNUM_qtype,
        ("min", "max"): constants.DIFF_MINMAX_qtype,
        ("min", "min"): constants.DIFF_MINMIN_qtype,
        ("max", "num"): constants.DIFF_MAXNUM_qtype,
        ("max", "min"): constants.DIFF_MAXMIN_qtype,
        ("max", "max"): constants.DIFF_MAXMAX_qtype,
    }

    split_point = -1
    if "compared" in question_tokens:
        split_point = question_tokens.index("compared")
    elif "than" in question_tokens:
        split_point = question_tokens.index("than")
    elif "and" in question_tokens:
        split_point = question_tokens.index("and")
    elif "over" in question_tokens:
        split_point = question_tokens.index("over")
    else:
        pass

    if split_point == -1:
        return None

    first_half_tokens = question_tokens[0:split_point]
    second_half_tokens = question_tokens[split_point:]

    if "longest" in first_half_tokens:
        first_number = "max"
    elif "shortest" in first_half_tokens:
        first_number = "min"
    else:
        first_number = "num"

    if "longest" in second_half_tokens:
        second_number = "max"
    elif "shortest" in second_half_tokens:
        second_number = "min"
    else:
        second_number = "num"

    qtype = numbertype_to_qtype_mapping[(first_number, second_number)]

    return qtype


def preprocess_HowManyYardsDifference_ques(dataset):
    """ This function prunes questions that start with "How many yards longer was" and "How many yards difference"

        Also add qtype for program supervision
    """

    difference_ngrams = ["how many yards difference", "how many yards longer was"]

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_qtypes = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)

    missed_questions = 0

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.tokenized_question]
            question_lower = original_question.lower()

            if any(span in question_lower for span in difference_ngrams):
                # qtype = get_question_type_and_attention(question_lower.split(" "))
                # if qtype is not None:
                #     question_answer[constants.qtype] = qtype
                #     question_answer[constants.program_supervised] = True
                #     qtype_dist[qtype] += 1
                #     questions_w_qtypes += 1
                # else:
                #     missed_questions += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and program supervised: {questions_w_qtypes}")
    print(f"Qtype dist: {qtype_dist}")
    print(f"Missed questions: {missed_questions}")

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

    new_train_dataset = preprocess_HowManyYardsDifference_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsDifference_ques(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written HowManyYards datasets")

""" DATASET CREATED THIS WAY

input_dir = "./resources/data/drop_old/analysis/ngram/num/how_many_yards_was_the/"
output_dir = "./resources/data/drop_old/num/how_many_yards_was_the"

"""
