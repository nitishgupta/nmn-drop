from typing import List, Tuple, Dict, Union, Any
import json
import argparse
from collections import defaultdict

from utils.util import round_all
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    filter_qids_w_logicalforms


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
        if w_module > 0:
            perc_correct_w_module = (float(correct_w_module)/float(w_module))*100.0
        else:
            perc_correct_w_module = 0.0
        perc_correct_w_module = round_all(perc_correct_w_module, 1)
        print("Module: {} \t T:{}  C: {} Perc: {} %".format(module_name, w_module, correct_w_module,
                                                            perc_correct_w_module))


def logicalform_distribution(pred_instances_1: List[NMNPredictionInstance],
                             pred_instances_2: List[NMNPredictionInstance]):
    lf2qids = defaultdict(list)
    qid2lf = defaultdict(str)
    lf2nmn1_correct = defaultdict(list)
    lf2nmn1_correctqids = defaultdict(set)
    for instance in pred_instances_1:
        qid = instance.query_id
        lf = instance.top_logical_form
        lf2qids[lf].append(qid)
        qid2lf[qid] = lf
        nmn1_correct: bool = instance.correct
        if nmn1_correct:
            lf2nmn1_correctqids[lf].add(qid)
        lf2nmn1_correct[lf].append(nmn1_correct)

    lf2nmn2_correct = defaultdict(list)
    lf2nmn2_correctqids = defaultdict(set)
    for mtmsn_ins in pred_instances_2:
        qid = mtmsn_ins.query_id
        lf = qid2lf[qid]
        mtmsn_correct = mtmsn_ins.correct
        lf2nmn2_correct[lf].append(mtmsn_correct)
        if mtmsn_correct:
            lf2nmn2_correctqids[lf].add(qid)

    sorted_lf = sorted(lf2qids.keys(), key=lambda x: len(lf2qids[x]), reverse=True)
    for lf in sorted_lf:
        print(f"{lf}")
        total_nmn1_correct = sum(lf2nmn1_correct[lf])
        total_nmn2_correct = sum(lf2nmn2_correct[lf])
        nmn2_correct_qids = lf2nmn2_correctqids[lf]
        nmn_correct_qids = lf2nmn1_correctqids[lf]
        common_correct_qids = nmn2_correct_qids.intersection(nmn_correct_qids)
        correct_in_nmn2_not_nmn1 = nmn2_correct_qids.difference(nmn_correct_qids)
        all_qids = set(lf2qids[lf])
        incorrect_in_both = all_qids.difference(nmn2_correct_qids.union(nmn_correct_qids))
        print(f"Total: {len(lf2qids[lf])}  NMN-1:{total_nmn1_correct}  NMN-2:{total_nmn2_correct} "
              f"Common correct: {len(common_correct_qids)}")
        print(f"Correct in NMN-2 not in NMN-1: \n{correct_in_nmn2_not_nmn1}")
        print(f"Incorrect in both: \n{incorrect_in_both}")
        print()



def model_comparison(pred_instances_1: List[NMNPredictionInstance], pred_instances_2: List[NMNPredictionInstance]):
    correct_qids1 = set(get_correct_qids(pred_instances_1))
    correct_qids2 = set(get_correct_qids(pred_instances_2))

    common_correct = correct_qids1.intersection(correct_qids2)

    print(f"Correct intersection between model-1 and model-2")
    print(len(common_correct))

    correct_in_1_not_2 = correct_qids1.difference(correct_qids2)
    correct_in_2_not_1 = correct_qids2.difference(correct_qids1)

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

    print(f"\nCorrect in NMN-1 but in NMN-2: {len(correct_in_1_not_2)}\n{correct_in_1_not_2}")
    print(f"\nCorrect in NMN-2 but in NMN-1: {len(correct_in_2_not_1)}\n{correct_in_2_not_1}")

    print("\nLogical form comparison\n")
    logicalform_distribution(pred_instances_1, pred_instances_2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmn_jsonl1")
    parser.add_argument("--nmn_jsonl2")
    args = parser.parse_args()

    nmn_jsonl1 = args.nmn_jsonl1
    nmn_jsonl2 = args.nmn_jsonl2

    pred_instances_1: List[NMNPredictionInstance] = read_nmn_prediction_file(nmn_jsonl1)
    pred_instances_2: List[NMNPredictionInstance] = read_nmn_prediction_file(nmn_jsonl2)

    modules_of_interest = ["year_difference_single_event",
                           "year_difference_two_events",
                           "compare_num_gt", "compare_num_lt",
                           "compare_date_gt", "compare_date_lt",
                           "select_min_num", "select_max_num",
                           "aggregate_count",
                           "select_num", "filter_passage", "project_passage",
                           "passagenumber_difference", "passagenumber_addition"]

    print()
    print("Model 1 : {}".format(nmn_jsonl1))
    print_about_model(pred_instances_1, modules_of_interest)
    print()

    print("Model 2 {}".format(nmn_jsonl2))
    print_about_model(pred_instances_2, modules_of_interest)

    print()

    print("Model prediction comparison")
    model_comparison(pred_instances_1, pred_instances_2)
