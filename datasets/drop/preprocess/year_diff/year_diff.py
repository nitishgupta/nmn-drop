import argparse
from collections import defaultdict
import json
from semqa.utils.qdmr_utils import lisp_to_nested_expression, nested_expression_to_tree, Node
import os

import datasets.drop.constants as constants

YEAR_DIFF_NGRAMS = [
    "how many years after the",
    "how many years did it",
    "how many years did the",
    "how many years passed between",
    "how many years was",
]

# "how many years passed between" -- two events
# "how many years after the" - year-diff between 2 events

# "how many years was" - single event if doesn't contain "from" or "between" (high precision filter)
# "how many years did it" - single event if it doesn't contain the word "from"
# "how many years did the" - single event


def is_single_event_yeardiff_question(question_lower: str):
    single_event_ques: bool = None
    if "how many years was" in question_lower:
        if "how many years was it between" in question_lower or "how many years was it from" in question_lower:
            single_event_ques = False
        else:
            single_event_ques = True

    # If "from" doesn't exist then surely single_event; o/w don't know
    if "how many years did it" in question_lower:
        if "from" not in question_lower:
            single_event_ques = True

    if "how many years did the" in question_lower:
        single_event_ques = True

    if "how many years passed between" in question_lower or "how many years after the" in question_lower:
        single_event_ques = False

    return single_event_ques


def program_supervision_node(is_single_event_ques: bool) -> Node:
    year_two_diff_lisp = "(year_difference_two_events select_passage select_passage)"
    year_one_diff_lisp = "(year_difference_single_event select_passage)"
    year_two_diff: Node = nested_expression_to_tree(lisp_to_nested_expression(year_two_diff_lisp))
    year_one_diff: Node = nested_expression_to_tree(lisp_to_nested_expression(year_one_diff_lisp))

    return year_one_diff if is_single_event_ques else year_two_diff


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def prune_YearDiffQues(dataset):
    """ Extract all questions that contain any one of the YEAR_DIFF_TRIGRAMS """

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)

    qtype_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            question = question_answer[constants.question]
            question_lower = question.lower()

            if any(span in question_lower for span in YEAR_DIFF_NGRAMS):
                single_event_ques: bool = is_single_event_yeardiff_question(question_lower)
                program_node = program_supervision_node(single_event_ques)
                question_answer[constants.program_supervision] = program_node.to_dict()

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Qtype dict: {qtype_dist}")

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

    print(f"Output dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = prune_YearDiffQues(train_dataset)

    new_dev_dataset = prune_YearDiffQues(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written augmented datasets")
