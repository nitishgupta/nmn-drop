from typing import List, Dict, Tuple
import json
import copy
import string
from collections import defaultdict
import datasets.drop.constants as constants
import random
import argparse

random.seed(100)

IGNORED_TOKENS = {'a', 'an', 'the'}
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])

FIRST = "first"
SECOND = "second"

""" This script is used to remove the weak/strong supervision for question-attention, event-date-groundings, etc.
    from all but 'x' number of passages.
"""

def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def getQuestionComparisonOperator(question: str) -> str:
    question_tokens = question.split(' ')
    # Correct if Attn1 is first event

    for t in ['first', 'earlier', 'forst', 'firts']:
        if t in question_tokens:
            return FIRST

    for t in ['later', 'last', 'second']:
        if t in question_tokens:
            return SECOND

    return SECOND


def removeDateCompPassageWeakAnnotations(dataset, annotation_for_numpassages):
    """ Given a dataset containing date-comparison questions that are heuristically strongly annotated
        and the number of passages that need to remain strongly annotated, we remove the strong annotations for other
        passages. These annotations include: question-attention, event-date-groundings, etc.
        Fields from which annotation is removed:
        1. constants.datecomp_ques_event_date_groundings, constants.datecomp_ques_event_date_values
        2. constants.datecomp_ques_event_attentions
        3. constants.strongly_annotated - is set to False for all questions

    """

    total_num_qa = 0

    total_num_passages = len(dataset)
    if annotation_for_numpassages == -1:
        print("Keeping annotation for all passages")
        return dataset

    passage_idxs = list(dataset.keys())
    random.shuffle(passage_idxs)
    choosen_passage_idxs = passage_idxs[0:annotation_for_numpassages]

    operator_dist_annotated = defaultdict(float)
    operator_dist_unannotated = defaultdict(float)
    total_strongly_supervised_qas = 0

    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])

        if passage_idx in choosen_passage_idxs:
            for qa in passage_info[constants.qa_pairs]:
                question = qa[constants.question]
                operator = getQuestionComparisonOperator(question)
                strongly_supervised = qa[constants.strongly_supervised]
                total_strongly_supervised_qas += 1 if strongly_supervised else 0
                if strongly_supervised:
                    operator_dist_annotated[operator] += 1
                else:
                    operator_dist_unannotated[operator] += 1
        else:
            # Removing the strong annotations for all QAs in this passage
            for question_answer in passage_info[constants.qa_pairs]:
                question = question_answer[constants.question]
                operator = getQuestionComparisonOperator(question)
                operator_dist_unannotated[operator] += 1

                # Removing QuestionEvent date grounding
                (qevent1_date_grounding,
                 qevent2_date_grounding) = question_answer[constants.datecomp_ques_event_date_groundings]

                qevent1_date_grounding = [0.0 for _ in qevent1_date_grounding]
                qevent2_date_grounding = [0.0 for _ in qevent2_date_grounding]

                qevent1_date_value = [-1, -1, -1]
                qevent2_date_value = [-1, -1, -1]

                question_answer[constants.datecomp_ques_event_date_groundings] = (qevent1_date_grounding,
                                                                                  qevent2_date_grounding)
                question_answer[constants.datecomp_ques_event_date_values] = (qevent1_date_value,
                                                                              qevent2_date_value)

                # Removing QuestionEvent attention grounding
                (qevent1_qattn,
                 qevent2_qattn) = question_answer[constants.ques_attention_supervision]

                qevent1_qattn = [0.0 for _ in qevent1_qattn]
                qevent2_qattn = [0.0 for _ in qevent2_qattn]

                question_answer[constants.ques_attention_supervision] = (qevent1_qattn, qevent2_qattn)

                question_answer[constants.strongly_supervised] = False

    print()
    print(f"TotalNumPassages: {total_num_passages}  Passages remaining annotated: {annotation_for_numpassages}")
    print(f"Num of original question: {total_num_qa}")
    print(f"Annotated Questions operator dist: {operator_dist_annotated}")
    print(f"Un-annotated Questions operator dist: {operator_dist_unannotated}")
    print(f"Strong Annotated Questions: {total_strongly_supervised_qas}")

    return dataset


if __name__=='__main__':
    # input_dir = "date_prune_weakdate"
    # trnfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_train.json"
    # devfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_dev.json"
    #
    # output_dir = "date_prune_weakdate_augment"
    # out_trfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_train.json"
    # out_devfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_dev.json"
    print("Removing strong annotation for DateComp QAs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_trnfp')
    parser.add_argument('--input_devfp')
    parser.add_argument('--output_trnfp')
    parser.add_argument('--output_devfp')
    parser.add_argument('--annotation_for_numpassages', type=int, required=True)
    args = parser.parse_args()

    annotation_for_numpassages = args.annotation_for_numpassages

    input_trnfp = args.input_trnfp
    input_devfp = args.input_devfp
    output_trnfp = args.output_trnfp
    output_devfp = args.output_devfp


    input_dir = "date_prune_augment"
    input_trnfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_train.json"
    input_devfp = f"/srv/local/data/nitishg/data/drop/{input_dir}/drop_dataset_dev.json"

    output_dir = "date_prune_augment_500"
    output_trnfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_train.json"
    output_devfp = f"/srv/local/data/nitishg/data/drop/{output_dir}/drop_dataset_dev.json"

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = removeDateCompPassageWeakAnnotations(train_dataset,
                                                             annotation_for_numpassages=annotation_for_numpassages)
    new_dev_dataset = removeDateCompPassageWeakAnnotations(dev_dataset,
                                                           annotation_for_numpassages=annotation_for_numpassages)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written datasets after annotation-removal")

