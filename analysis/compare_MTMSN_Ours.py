from typing import List, Dict, Tuple

import os
import json
import argparse
from collections import defaultdict


class NMNPredictionInstance:
    """ Class to hold a single NMN prediction written in json format.
        Typically these outputs are written by the "drop_parser_jsonl_predictor"
    """
    def __init__(self, pred_dict):
        self.question = pred_dict.get("question", "")
        self.question_id = pred_dict.get("query_id", "")
        self.gold_logical_form = pred_dict.get("gold_logical_form", "")
        self.predicted_ans = pred_dict.get("predicted_ans", "")
        self.top_logical_form = pred_dict.get("logical_form", "")
        self.top_nested_expr = pred_dict.get("nested_expression", [])
        self.top_logical_form_prob = pred_dict.get("logical_form_prob", 0.0)
        self.f1_score = pred_dict.get("f1", 0.0)
        self.exact_match = pred_dict.get("em", 0.0)
        self.correct = True if self.f1_score > 0.6 else False


def read_nmn_prediction(jsonl_file) -> List[NMNPredictionInstance]:
    """ Input json-lines written typically by the "drop_parser_jsonl_predictor". """
    with open(jsonl_file, "r") as f:
        return [NMNPredictionInstance(json.loads(line)) for line in f.readlines()]


def avg_f1(instances: List[NMNPredictionInstance]):
    """ Avg F1 score for the predictions. """
    total = sum([instance.f1_score for instance in instances])
    return float(total)/float(len(instances))


def read_mtmsn_predictions(json_file) -> List[NMNPredictionInstance]:
    with open(json_file, 'r') as f:
        mtmsn_preds = json.load(f)

    pred_instances = []
    for qid, preds in mtmsn_preds.items():
        pred = preds[0]         # Take first dictionary only
        pred_dict = {
            "query_id": qid,
            "predicted_ans": pred["text"],
            "logical_form": pred["type"],
            "f1": pred["f1"],
            "em": pred["em"],
        }
        pred_instances.append(NMNPredictionInstance(pred_dict))
    return pred_instances


def logicalform_distribution(nmn_instances: List[NMNPredictionInstance], mtmsn_instances: List[NMNPredictionInstance]):
    lf2qids = defaultdict(list)
    lf2nmn_correct = defaultdict(list)
    for instance in nmn_instances:
        lf = instance.top_logical_form
        qid = instance.question_id
        nmn_correct = instance.correct
        lf2qids[lf].append(qid)
        lf2nmn_correct[lf].append(nmn_correct)

    lf2mtmsn_correct = defaultdict(list)
    for mtmsn_ins in mtmsn_instances:
        qid = mtmsn_ins.question_id
        for lf, qids in lf2qids.items():
            if qid in qids:
                lf2mtmsn_correct[lf].append(mtmsn_ins.correct)

    sorted_lf = sorted(lf2qids.keys(), key=lambda x: len(lf2qids[x]), reverse=True)
    for lf in sorted_lf:
        print(f"{lf}")
        total_nmn_correct = sum(lf2nmn_correct[lf])
        total_mtmsn_correct = sum(lf2mtmsn_correct[lf])
        print(f"Total: {len(lf2qids[lf])}  NMN:{total_nmn_correct}  MTMSN:{total_mtmsn_correct}")
        print()


def compute_mtmsn_type_distribution(qids: List[str], predictions: List[NMNPredictionInstance]):
    qids = set(qids)
    type_dist = {}
    type2qids = {}
    for pred in predictions:
        if pred.question_id in qids:
            type_dist[pred.top_logical_form] = type_dist.get(pred.top_logical_form, 0) + 1
            if pred.top_logical_form not in type2qids:
                type2qids[pred.top_logical_form] = []
            type2qids[pred.top_logical_form].append(pred.question_id)
    return type_dist, type2qids



def get_correct_qids(instances: List[NMNPredictionInstance]) -> List[str]:
    """ Get list of QIDs with correc predictions """
    qids = [instance.question_id for instance in instances if instance.correct]
    return qids


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
    diff_qids = [instance.question_id for instance in nmn_predictions if instance.question_id in correct_in_mtmsn_qids]
    print(diff_qids)

    print(f"Correct in NMN but not in MTMSN")
    correct_in_nmn_qids = correct_nmn_qids.difference(correct_mtmsn_qids)
    diff_qids = [instance.question_id for instance in nmn_predictions if instance.question_id in correct_in_nmn_qids]
    print(diff_qids)

    correct_in_mtmsn_typedist, correct_in_mtmsn_type2qids = compute_mtmsn_type_distribution(correct_in_mtmsn_qids,
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
    nmn_preds: List[NMNPredictionInstance] = read_nmn_prediction(args.our_preds_jsonl)

    # This is a JSON file
    mtmsn_preds: List[NMNPredictionInstance] = read_mtmsn_predictions(args.mtmsn_preds_json)

    model_stats(nmn_preds)
    model_stats(mtmsn_preds)

    model_comparison(nmn_preds, mtmsn_preds)
