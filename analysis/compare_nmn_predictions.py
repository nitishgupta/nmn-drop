from typing import List, Tuple, Dict, Union, Any

from semqa.utils.qdmr_utils import convert_nestedexpr_to_tuple

import json
import argparse


class NMNPredictionInstance:
    """ Class to hold a single NMN prediction written in json format.
        Typically these outputs are written by the "drop_parser_jsonl_predictor"
    """
    def __init__(self, pred_dict):
        self.question: str = pred_dict["question"]
        self.question_id: str = pred_dict["query_id"]
        self.gold_logical_form: str = pred_dict["gold_logical_form"]
        self.predicted_ans: str = pred_dict["predicted_ans"]
        self.top_logical_form: str = pred_dict["logical_form"]
        self.top_nested_expr: List = pred_dict["nested_expression"]
        self.top_logical_form_prob: float = pred_dict["logical_form_prob"]
        self.f1_score: float = pred_dict["f1"]
        self.exact_match: float = pred_dict["em"]
        self.correct: bool = True if self.f1_score > 0.6 else False


def read_prediction_file(jsonl_file) -> List[NMNPredictionInstance]:
    """ Input json-lines written typically by the "drop_parser_jsonl_predictor". """
    with open(jsonl_file, "r") as f:
        return [NMNPredictionInstance(json.loads(line)) for line in f.readlines()]


def avg_f1(instances: List[NMNPredictionInstance]):
    """ Avg F1 score for the predictions. """
    total = sum([instance.f1_score for instance in instances])
    return float(total)/float(len(instances))


def get_correct_qids(instances: List[NMNPredictionInstance], filtered_qids=None) -> List[str]:
    """ Get list of QIDs with correc predictions """
    if filtered_qids is None:
        qids = [instance.question_id for instance in instances if instance.correct]
    else:
        qids = [instance.question_id for instance in instances if instance.correct and
                instance.question_id in filtered_qids]
    return qids


def filter_qids_w_logicalforms(instances: List[NMNPredictionInstance], logical_forms: List[str]):
    filtered_qids = []
    for instance in instances:
        if instance.top_logical_form in logical_forms:
            filtered_qids.append(instance.question_id)
    return filtered_qids


def get_correct_w_module(instances: List[NMNPredictionInstance], module_name:str):
    """ Compute the number of instances w/ the given module in their predicted program and correct among them. """
    correct_and_module = 0
    w_module = 0
    for instance in instances:
        if module_name in instance.top_logical_form:
            w_module += 1
            if instance.correct:
                correct_and_module += 1
    return w_module, correct_and_module


def compute_parser_accuracy(instances: List[NMNPredictionInstance]):
    """ Compute parser accuracy for instances with gold-program supervision
        Only program-template are compared (for now).
    """
    total_w_gold = 0
    correct = 0
    for instance in instances:
        gold_logical_form = instance.gold_logical_form
        pred_logical_form = instance.top_logical_form
        if gold_logical_form:
            total_w_gold += 1
            correct += int(gold_logical_form == pred_logical_form)
    return float(correct)/total_w_gold, total_w_gold


def print_about_model(pred_instances: List[NMNPredictionInstance], modules_of_interest: List[str]):
    print(f"Total instances: {len(pred_instances)}")
    print(f"F1: {avg_f1(pred_instances)}")
    correct_qids = get_correct_qids(pred_instances)
    correct_qids = set(correct_qids)
    print(f"Correct instances: {len(correct_qids)}")
    parser_acc, total_w_gold_programs = compute_parser_accuracy(pred_instances)
    print(f"Parser accuracy: {parser_acc}   Total w/ gold: {total_w_gold_programs}")

    for module_name in modules_of_interest:
        w_module, correct_w_module = get_correct_w_module(pred_instances, module_name=module_name)
        print("Module: {} \t T:{}  C: {}".format(module_name, w_module, correct_w_module))


def model_comparison(pred_instances_1: List[NMNPredictionInstance], pred_instances_2: List[NMNPredictionInstance]):
    correct_qids1 = set(get_correct_qids(pred_instances_1))
    correct_qids2 = set(get_correct_qids(pred_instances_2))

    common_correct = correct_qids1.intersection(correct_qids2)

    print(f"Correct intersection between model-1 and model-2")
    print(len(common_correct))

    # print(f"Correct in model-1 but not in model-2")
    # diff_qids_set = correct_qids1.difference(correct_qids2)
    # diff_qids = [instance.question_id for instance in pred_instances_1 if instance.question_id in diff_qids_set]
    # print(diff_qids)

    qids1_w_select = filter_qids_w_logicalforms(pred_instances_1,
                                                logical_forms=["(select_passagespan_answer select_passage)"])
    qids1_w_select = set(qids1_w_select)
    qids1_w_select_correct = set(get_correct_qids(pred_instances_1, qids1_w_select))

    qids2_w_select = filter_qids_w_logicalforms(pred_instances_2,
                                                logical_forms=["(select_passagespan_answer select_passage)"])
    qids2_w_select = set(qids2_w_select)
    qids2_w_select_correct = set(get_correct_qids(pred_instances_2, qids2_w_select))

    common_w_select = qids1_w_select.intersection(qids2_w_select)
    common_w_select_correct = qids1_w_select_correct.intersection(qids2_w_select_correct)

    print(f"Model 1 with select-prog: {len(qids1_w_select)}  Correct: {len(qids1_w_select_correct)}")
    print(f"Model 2 with select-prog: {len(qids2_w_select)}  Correct: {len(qids2_w_select_correct)}")
    print(f"Common: {len(common_w_select)}  Common-correct: {len(common_w_select_correct)}")









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl_1")
    parser.add_argument("--pred_jsonl_2")
    args = parser.parse_args()

    pred_instances_1: List[NMNPredictionInstance] = read_prediction_file(args.pred_jsonl_1)
    pred_instances_2: List[NMNPredictionInstance] = read_prediction_file(args.pred_jsonl_2)

    modules_of_interest = ["year_difference_single_event",
                           "year_difference_two_events",
                           "compare_num_gt", "compare_num_lt",
                           "compare_date_gt", "compare_date_gt",
                           "select_min_num", "select_max_num",
                           "aggregate_count",
                           "select_num", "filter_passage", "project_passage",
                           "passagenumber_difference", "passagenumber_addition"]

    print()
    print("Model 1 : {}".format(args.pred_jsonl_1))
    print_about_model(pred_instances_1, modules_of_interest)
    print()

    print("Model 2 {}".format(args.pred_jsonl_2))
    print_about_model(pred_instances_2, modules_of_interest)

    print()

    print("Model prediction comparison")
    model_comparison(pred_instances_1, pred_instances_2)




