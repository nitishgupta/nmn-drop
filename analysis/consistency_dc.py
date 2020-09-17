from typing import List, Tuple, Dict, Union, Any
import json
import argparse
from collections import defaultdict, OrderedDict

from utils.util import round_all
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    avg_em


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




def compute_consistency(orig_preds_jsonl: str, aug_preds_jsonl: str = None):
    # orig_instances: List[NMNPredictionInstance] = read_nmn_prediction_file(orig_preds_jsonl)
    # aug_instances: List[NMNPredictionInstance] = read_nmn_prediction_file(aug_preds_jsonl)

    orig_instances: List[NMNPredictionInstance] = read_mtmsn_predictions(orig_preds_jsonl)
    aug_instances: List[NMNPredictionInstance] = read_mtmsn_predictions(aug_preds_jsonl)

    qid2allqids = defaultdict(list)
    qid2f1s_all = {}
    for instance in aug_instances:
        qid = instance.query_id
        if "dc-event-switch-dc-qop-switch" in qid:
            pruned_qid = qid.replace("-dc-event-switch-dc-qop-switch", "")
        elif "-dc-qop-switch" in qid:
            pruned_qid = qid.replace("-dc-qop-switch", "")
        elif "-dc-event-switch" in qid:
            pruned_qid = qid.replace("-dc-event-switch", "")
        else:
            continue

        if pruned_qid not in qid2allqids[pruned_qid]:
            qid2allqids[pruned_qid].append(pruned_qid)
        qid2allqids[pruned_qid].append(qid)

        qid2f1s_all[qid] = instance.f1_score

        # print(instance.question)
        # print(instance.gold_answers)
        # print(instance.predicted_ans)
        # print(instance.f1_score)
        # print()

    for instance in orig_instances:
        qid2f1s_all[instance.query_id] = instance.f1_score


    qid2f1s_dc = {}
    consistency_total = 0
    for qid, qids in qid2allqids.items():
        qid2f1s_dc[qid] = []
        for qd in qids:
            qid2f1s_dc[qid].append(qid2f1s_all[qd])
        consistent = False
        if all([x >= 0.5 for x in qid2f1s_dc[qid]]):
            consistent = True
            consistency_total += 1
        # print("{} {}".format(qid2f1s_dc[qid], consistent))



    print("Number of original questions: {}".format(len(qid2allqids)))
    print(consistency_total)
    # print(qid2f1s_dc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_preds_jsonl")
    parser.add_argument("--aug_preds_jsonl")
    args = parser.parse_args()

    # orig_preds_jsonl = "/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_10/predictions/iclr_qdmr-v4-noexc_test_predictions-Ex0-Rev.jsonl"
    #
    # aug_preds_jsonl = "/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_10/predictions/dc-aug-test_test_predictions.jsonl"

    # orig_preds_jsonl = "/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FGS-DCYD-ND-MM/predictions/iclr_qdmr-v4-noexc_test_predictions-Ex0-Rev.jsonl"
    #
    # aug_preds_jsonl = "/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FGS-DCYD-ND-MM/predictions/dc-aug-test_test_predictions.jsonl"

    orig_preds_jsonl = "/shared/nitishg/checkpoints/MTMSN/iclr21/iclr_qdmr-v2-noexc/S_42/predictions/iclr_qdmr-v2-noexc_test_preds.json"

    aug_preds_jsonl = "/shared/nitishg/checkpoints/MTMSN/iclr21/iclr_qdmr-v2-noexc/S_42/predictions/dc-aug-test_test_preds.json"

    compute_consistency(orig_preds_jsonl, aug_preds_jsonl)

