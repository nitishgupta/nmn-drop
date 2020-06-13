from typing import List, Dict, Tuple

import os
import json
import argparse
from collections import defaultdict
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    filter_qids_w_logicalforms


def read_mtmsn_predictions(json_file) -> List[NMNPredictionInstance]:
    with open(json_file, 'r') as f:
        mtmsn_preds = json.load(f)

    pred_instances = []
    for qid, preds in mtmsn_preds.items():
        pred = preds[0]         # Take first dictionary only
        pred_dict = {
            "query_id": qid,
            "predicted_ans": pred["text"],
            "top_logical_form": pred["type"],
            "f1": pred["f1"],
            "em": pred["em"],
        }
        pred_instances.append(NMNPredictionInstance(pred_dict))
    return pred_instances


def logicalform_distribution(nmn_instances: List[NMNPredictionInstance], mtmsn_instances: List[NMNPredictionInstance]):
    lf2qids = defaultdict(list)
    qid2lf = defaultdict(str)
    lf2nmn_correct = defaultdict(list)
    lf2nmn_correctqids = defaultdict(set)
    for instance in nmn_instances:
        qid = instance.query_id
        lf = instance.top_logical_form
        lf2qids[lf].append(qid)
        qid2lf[qid] = lf
        nmn_correct: bool = instance.correct
        if nmn_correct:
            lf2nmn_correctqids[lf].add(qid)
        lf2nmn_correct[lf].append(nmn_correct)

    lf2mtmsn_correct = defaultdict(list)
    lf2mtmsn_correctqids = defaultdict(set)
    for mtmsn_ins in mtmsn_instances:
        qid = mtmsn_ins.query_id
        lf = qid2lf[qid]
        mtmsn_correct = mtmsn_ins.correct
        lf2mtmsn_correct[lf].append(mtmsn_correct)
        if mtmsn_correct:
            lf2mtmsn_correctqids[lf].add(qid)

    sorted_lf = sorted(lf2qids.keys(), key=lambda x: len(lf2qids[x]), reverse=True)
    for lf in sorted_lf:
        print(f"{lf}")
        total_nmn_correct = sum(lf2nmn_correct[lf])
        total_mtmsn_correct = sum(lf2mtmsn_correct[lf])
        print(f"Total: {len(lf2qids[lf])}  NMN:{total_nmn_correct}  MTMSN:{total_mtmsn_correct} "
              f"Common correct: {len(lf2nmn_correctqids[lf].intersection(lf2mtmsn_correctqids[lf]))}")
        print()


def compute_mtmsn_type_distribution(qids: List[str], predictions: List[NMNPredictionInstance]):
    qids = set(qids)
    type_dist = {}
    type2qids = {}
    for pred in predictions:
        if pred.query_id in qids:
            type_dist[pred.top_logical_form] = type_dist.get(pred.top_logical_form, 0) + 1
            if pred.top_logical_form not in type2qids:
                type2qids[pred.top_logical_form] = []
            type2qids[pred.top_logical_form].append(pred.query_id)
    return type_dist, type2qids


def model_stats(pred_instances: List[NMNPredictionInstance]):
    print()
    print(f"Total instances: {len(pred_instances)}")
    print(f"F1: {avg_f1(pred_instances)}")
    correct_qids = get_correct_qids(pred_instances)
    correct_qids = set(correct_qids)
    print(f"Correct instances: {len(correct_qids)}")
    print()



def model_comparison(nmn_predictions: List[NMNPredictionInstance], mtmsn_predictions: List[NMNPredictionInstance]):
    correct_nmn_qids = set(get_correct_qids(nmn_predictions))
    correct_mtmsn_qids = set(get_correct_qids(mtmsn_predictions))

    print("Correct in NMN: {}   MTMSN:{}".format(len(correct_nmn_qids), len(correct_mtmsn_qids)))

    common_correct = correct_nmn_qids.intersection(correct_mtmsn_qids)

    print(f"Correct intersection between model-1 and model-2")
    print(len(common_correct))

    print(f"Correct in MTMSN but not in NMN")
    correct_in_mtmsn_qids = correct_mtmsn_qids.difference(correct_nmn_qids)
    diff_qids = [instance.query_id for instance in nmn_predictions if instance.query_id in correct_in_mtmsn_qids]
    print(diff_qids)

    print(f"Correct in NMN but not in MTMSN")
    correct_in_nmn_qids = correct_nmn_qids.difference(correct_mtmsn_qids)
    diff_qids = [instance.query_id for instance in nmn_predictions if instance.query_id in correct_in_nmn_qids]
    print(diff_qids)

    correct_in_mtmsn_typedist, correct_in_mtmsn_type2qids = compute_mtmsn_type_distribution(list(correct_in_mtmsn_qids),
                                                                                            mtmsn_predictions)
    # print(correct_in_mtmsn_typedist)
    # for k, v in correct_in_mtmsn_type2qids.items():
    #     print(k)
    #     print(v)

    logicalform_distribution(nmn_predictions, mtmsn_predictions)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_preds_jsonl")
    parser.add_argument("--mtmsn_preds_json")
    # parser.add_argument("--analysis_output_dir", default=None)
    args = parser.parse_args()

    # This is usuallya JSON-L file
    nmn_preds: List[NMNPredictionInstance] = read_nmn_prediction_file(args.our_preds_jsonl)

    # This is a JSON file
    mtmsn_preds: List[NMNPredictionInstance] = read_mtmsn_predictions(args.mtmsn_preds_json)

    model_stats(nmn_preds)
    model_stats(mtmsn_preds)

    model_comparison(nmn_preds, mtmsn_preds)
