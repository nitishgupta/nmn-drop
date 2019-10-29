from typing import List, Dict, Tuple

import os
import json
import argparse


def make_correct_incorrect_dicts(preds):
    """Make separate sets of question ids for correctly and in-correctly answered questions.

    Args:
        instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )

    Returns:
        correct_qids: Set of qids answered correctly
        incorrect_qids: Set of qids answered incorrectly
    """
    correct_qids = set()
    incorrect_qids = set()

    for query_id, pred_dict in preds.items():
        if pred_dict["f1"] >= 0.7:
            correct_qids.add(query_id)
        else:
            incorrect_qids.add(query_id)

    return correct_qids, incorrect_qids


def make_qid2questiondict(our_model_preds):
    """Make question_id to question text map

    Args:
        our_model_preds: keys are query_id and the value is a dictionary containing the key "question"

    Returns:
        qid2qtext: Dict from qid 2 qtext
    """
    qid2qtext = {}
    for query_id, pred_dict in our_model_preds.items():
        question = pred_dict["question"]
        qid2qtext[query_id] = question
    return qid2qtext


def make_qid2logicalform(preds):
    """Make question_id to predicted logical_form map

    Args:
        preds: prediction dict with "type" key containing LF

    Returns:
        qid2lf: Dict from qid 2 logical form
    """
    qid2lf = {}
    for query_id, pred_dict in preds.items():
        logical_form = pred_dict["type"]
        qid2lf[query_id] = logical_form
    return qid2lf


def print_qtexts(qids, qid2qtext):
    for i, qid in enumerate(qids):
        print("{}: {}".format(i, qid2qtext[qid]))


def print_qtext_lf(qids, qid2qtext, qid2lf):
    for i, qid in enumerate(qids):
        print("{}: {} -- {}".format(i, qid2qtext[qid], qid2lf[qid]))


def print_qtext_bothlfs(qids, qid2qtext, qid2lf1, qid2lf2):
    for i, qid in enumerate(qids):
        print("qid: {}".format(qid))
        print("{}: {}\n{}\n{}\n".format(i, qid2qtext[qid], qid2lf1[qid], qid2lf2[qid]))


def diff_in_lfs(qids, qid2lf1, qid2lf2):
    num_diff_lf = 0
    for i, qid in enumerate(qids):
        lf1 = qid2lf1[qid]
        lf2 = qid2lf2[qid]

        if lf1 != lf2:
            num_diff_lf += 1

    return num_diff_lf


def overlap_in_correct_incorrect_questext(correct_qids, incorrect_qids, qid2ques):
    correct_qtexts = set([qid2ques[qid] for qid in correct_qids])
    incorrect_qtexts = set([qid2ques[qid] for qid in incorrect_qids])

    overlap = correct_qtexts.intersection(incorrect_qtexts)

    print("Unique correct questions: {}".format(len(correct_qtexts)))
    print("Unique incorrect questions: {}".format(len(incorrect_qtexts)))
    print("Unique question overlap: {}".format(len(overlap)))


def error_overlap_statistics(our_model_preds, base_model_preds, analysis_output_dir=None):
    """Perform analysis on predictions of two different models and see the level of overlap in their predictions.
    """
    num_instances = len(our_model_preds)

    qid2ques = make_qid2questiondict(our_model_preds)

    qid2lf_ours = make_qid2logicalform(our_model_preds)
    qid2lf_base = make_qid2logicalform(base_model_preds)

    correct_qids_ours, incorrect_qids_ours = make_correct_incorrect_dicts(our_model_preds)
    correct_qids_base, incorrect_qids_base = make_correct_incorrect_dicts(base_model_preds)

    perf1 = len(correct_qids_ours) / float(num_instances)
    perf2 = len(correct_qids_base) / float(num_instances)

    correct_overlap_qids = correct_qids_ours.intersection(correct_qids_base)

    correct1_incorrect2_qids = correct_qids_ours.intersection(incorrect_qids_base)
    incorrect1_correct2_qids = incorrect_qids_ours.intersection(correct_qids_base)


    # # For questions predicted correct by M1, and incorrect by M2 -- the number of ques with diff LFs
    # c1_nc2_lf_diff = diff_in_lfs(correct1_incorrect2_qids, qid2lf_ours, qid2lf_base)
    # nc1_c2_lf_diff = diff_in_lfs(incorrect1_correct2_qids, qid2lf1, qid2lf2)

    # print("Correct in Model 1, Incorrect in Model 2")
    # print_qtext_bothlfs(correct1_incorrect2_qids, qid2qtext2, qid2lf1, qid2lf2)

    # print("Incorrect in Model 1, Correct in Model 2")
    # print_qtext_bothlfs(incorrect1_correct2_qids, qid2ques, qid2lf1, qid2lf2)

    print("Model Ours")
    print(
        "Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(
            num_instances, len(correct_qids_ours), len(incorrect_qids_ours), perf1
        )
    )
    print()
    print("Model Base")
    print(
        "Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(
            num_instances, len(correct_qids_base), len(incorrect_qids_base), perf2
        )
    )

    print("Correct Overlap : {}".format(len(correct_overlap_qids)))
    print()

    print("Correct in M1; Incorrect in M2: {}".format(len(correct1_incorrect2_qids)))
    print()

    print("Incorrect in M1; Correct in M2: {}".format(len(incorrect1_correct2_qids)))
    print()


    if analysis_output_dir is None:
        return

    print("\nWriting analysis to files in {}".format(analysis_output_dir))
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir, exist_ok=True)

    with open(os.path.join(analysis_output_dir, "correct_ours_incorrect_base.txt"), "w") as f:
        for qid in correct1_incorrect2_qids:
            f.write("{} \t {}\n".format(qid, qid2ques[qid]))

    with open(os.path.join(analysis_output_dir, "incorrect_ours_correct_base.txt"), "w") as f:
        for qid in incorrect1_correct2_qids:
            f.write("{} \t {}\n".format(qid, qid2ques[qid]))

    # print("Num of ques w/ different LF predictions: {}".format(nc1_c2_lf_diff))
    #
    # print("Model1")
    # overlap_in_correct_incorrect_questext(correct_qids_ours, incorrect_qids_ours, qid2ques)
    #
    # print("Model2")
    # overlap_in_correct_incorrect_questext(correct_qids_base, incorrect_qids_base, qid2qtext2)


def mtmsn_get_first_prediction(mtmsn_preds):
    """MTMSN has the ability to predict multiple spans as the answer. This in the predictions is stored as
    {qid: List of dicts} with each dictionary having the key "text" as the predicted span.

    We store "f1" and "em" in each dictionary which is computed based on all predicted spans.
    This function only keeps the first dictionary for analysis and converts the mtmsn_preds into a {qid: Dict} map.
    """
    mtmsn_preds_onlyfirst = {}
    for qid, preds in mtmsn_preds.items():
        mtmsn_preds_onlyfirst[qid] = preds[0]
    return mtmsn_preds_onlyfirst



def convert_ourlist_to_dict(data: List[Dict]) -> Dict:
    output_dict = {}
    for d in data:
        query_id = d["query_id"]
        if query_id in output_dict:
            print("Duplicate : {}".format(query_id))
        output_dict[query_id] = d
    return output_dict


def read_json(input_json):
    with open(input_json, 'r') as f:
        data = json.load(f)
    return data


def read_jsonl(input_jsonl):
    data = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_preds_jsonl")
    parser.add_argument("--mtmsn_preds_json")
    parser.add_argument("--analysis_output_dir", default=None)
    args = parser.parse_args()

    # This is usuallya JSON-L file
    our_preds_list = read_jsonl(args.our_preds_jsonl)
    our_preds: Dict = convert_ourlist_to_dict(our_preds_list)

    # This is a JSON file
    mtmsn_preds: Dict = read_json(args.mtmsn_preds_json)
    mtmsn_preds_onlyfirst: Dict = mtmsn_get_first_prediction(mtmsn_preds)

    print("Number of predictions in our preds: {}".format(len(our_preds)))
    print("Number of predictions in mtmsn: {}".format(len(mtmsn_preds_onlyfirst)))

    """
    
    1. Read ours into { query_id : pred }
    2. Compare this with MTMSN. assert query ids are same. 
    3. Output intersection, a - b, and b - a
    """

    # If analysis_output_dir is not None: the function makes the directory and writes a bunch of files in it
    error_overlap_statistics(our_model_preds=our_preds, base_model_preds=mtmsn_preds_onlyfirst,
                             analysis_output_dir=args.analysis_output_dir)