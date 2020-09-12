from typing import List, Dict, Tuple, Any, Union

import json


class NMNPredictionInstance:
    """ Class to hold a single NMN prediction written in json format.
        Typically these outputs are written by the "drop_parser_jsonl_predictor"
    """
    def __init__(self, pred_dict):
        self.question: str = pred_dict.get("question", "")
        self.query_id: str = pred_dict["query_id"]
        self.gold_logical_form = pred_dict.get("gold_logical_form", "")
        self.predicted_ans = pred_dict.get("predicted_ans", "")
        self.top_logical_form = pred_dict.get("top_logical_form", "")
        self.top_nested_expr: List = pred_dict.get("top_nested_expr", "")
        self.top_logical_form_prob: float = pred_dict.get("top_logical_form_prob", "")
        self.program_execution: List[Dict] = pred_dict.get("program_execution", None)
        self.gold_answers: float = pred_dict.get("gold_answers", [])
        self.f1_score: float = pred_dict.get("f1", 0.0)
        self.exact_match: float = pred_dict.get("em", 0.0)
        self.correct: bool = True if self.f1_score > 0.5 else False


def read_nmn_prediction_file(jsonl_file) -> List[NMNPredictionInstance]:
    """ Input json-lines written typically by the "drop_parser_jsonl_predictor". """
    with open(jsonl_file, "r") as f:
        return [NMNPredictionInstance(json.loads(line)) for line in f.readlines()]


def avg_f1(instances: List[NMNPredictionInstance]) -> float:
    """ Avg F1 score for the predictions. """
    if not instances:
        return 0.0
    total = sum([instance.f1_score for instance in instances])
    return float(total)/float(len(instances))

def avg_em(instances: List[NMNPredictionInstance]) -> float:
    """ Avg EM score for the predictions. """
    if not instances:
        return 0.0
    total = sum([instance.exact_match for instance in instances])
    return float(total)/float(len(instances))


def get_correct_qids(instances: List[NMNPredictionInstance], filtered_qids=None) -> List[str]:
    """ Get list of QIDs with correc predictions """
    if filtered_qids is None:
        qids = [instance.query_id for instance in instances if instance.correct]
    else:
        qids = [instance.query_id for instance in instances if instance.correct and
                instance.query_id in filtered_qids]
    return qids


def filter_qids_w_logicalforms(instances: List[NMNPredictionInstance], logical_forms: List[str]):
    filtered_qids = []
    for instance in instances:
        if instance.top_logical_form in logical_forms:
            filtered_qids.append(instance.query_id)
    return filtered_qids


def get_qid2nmninstance_map(instances: List[NMNPredictionInstance]) -> Dict[str, NMNPredictionInstance]:
    qid2nmninstance = {}
    for instance in instances:
        qid2nmninstance[instance.query_id] = instance
    return qid2nmninstance







