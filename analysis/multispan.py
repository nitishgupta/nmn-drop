from typing import List, Dict, Tuple
from semqa.utils.qdmr_utils import read_drop_dataset, convert_answer
from datasets.drop import constants

from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    filter_qids_w_logicalforms, get_qid2nmninstance_map


def is_ans_multispan(qa):
    answer_annotation = qa[constants.answer]
    ans_type, answerlist = convert_answer(answer_annotation)

    ans_multispan = False
    ans_span = False
    if ans_type == "spans":
        ans_span = True
        if len(answerlist) > 1:
            ans_multispan = True
    return ans_span, ans_multispan


def num_multispan_ans_in_dataset(drop_json):
    dataset = read_drop_dataset(drop_json)

    total, num_spanans, num_multispan = 0, 0, 0
    qids = []
    for pid, pinfo in dataset.items():
        for qa in pinfo[constants.qa_pairs]:
            total += 1
            ans_span, ans_multispan = is_ans_multispan(qa)
            num_spanans += int(ans_span)
            num_multispan += int(ans_multispan)

            if ans_multispan:
                qids.append(qa[constants.query_id])

    print("Total: {} SpanAns: {}  multi-span ans: {}".format(total, num_spanans, num_multispan))
    return qids


DROP_DATASET_JSON = "/shared/nitishg/data/drop-w-qdmr/qdmr-filter-post-v6/drop_dataset_train.json"

NMN_PRED_JSONL = "/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-filter-post-v6/drop_parser_bert/Qattn_true/EXCLOSS_true/" \
                 "aux_true/BIO_false_IO/SUPEPOCHS_0_HEM_0_BM_1/S_42_DeCont/predictions/qdmr-filter-post-v6_train_predictions.jsonl"


multispan_qids = num_multispan_ans_in_dataset(drop_json=DROP_DATASET_JSON)
# print(multispan_qids)

nmn_predictions: List[NMNPredictionInstance] = read_nmn_prediction_file(NMN_PRED_JSONL)
avgF1 = avg_f1(nmn_predictions)
print(f"Average F1: {avgF1}")

qid2nmninstance_map = get_qid2nmninstance_map(nmn_predictions)

multispan_instances = [qid2nmninstance_map[qid] for qid in multispan_qids]
multispan_question_avgf1 = avg_f1(multispan_instances)
print("Gold multispan avgF1: {}".format(multispan_question_avgf1))




