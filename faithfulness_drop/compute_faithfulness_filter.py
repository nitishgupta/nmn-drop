from typing import List, Dict, Tuple
import json
import argparse
import numpy as np
from collections import defaultdict

import datasets.drop.constants as constants
from semqa.utils.qdmr_utils import node_from_dict, nested_expression_to_lisp, lisp_to_nested_expression
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    filter_qids_w_logicalforms
from utils.util import round_all


QTYPES = ['date_comparison', 'number_comparison', 'num_find_qtype', 'num_filterfind_qtype', 'min_find_qtype',
          'min_filterfind_qtype', 'max_find_qtype', 'max_filterfind_qtype', 'count_find_qtype',
          'count_filterfind_qtype', 'relocate_find_qtype', 'relocate_filterfind_qtype', 'relocate_maxfind_qtype',
          'relocate_maxfilterfind_qtype', 'relocate_minfind_qtype', 'relocate_minfilterfind_qtype',
          'yeardiff_find_qtype', 'yeardiff_find2_qtype']


def convert_to_nparray(attention_score):
    return np.array(attention_score, dtype=np.float32)


def get_module_outputs(module_output_dict: Dict[str, List[Dict]], module_name):
    """Get the relevant outputs from a module

    module_output_dict has a single key, module_name, and the values are a list of Output.as_json().
    """
    outputs = module_output_dict.get(module_name, None)
    if outputs is None:
        print("{} not in {}".format(module_name, module_output_dict.keys()))
        return None

    returned_attentions = []

    if module_name == "select_passage":
        select_pattn = None
        for output in outputs:
            if output['label'] == "passage_attn":
                select_pattn = output['values']
        returned_attentions.append(convert_to_nparray(select_pattn))

    elif module_name == "filter_passage":
        filter_pattn = None
        for output in outputs:
            if output['label'] == "filtered_pattn":
                filter_pattn = output['values']
        returned_attentions.append(convert_to_nparray(filter_pattn))

    elif module_name == "project_passage":
        project_pattn = None
        for output in outputs:
            if output['label'] == "project_pattn":
                project_pattn = convert_to_nparray(output['values'])
        returned_attentions.append(project_pattn)

    elif module_name in ["select_max_num", "select_min_num"]:
        number_input_pattn, minmax_pattn = None, None
        for output in outputs:
            if output['label'] == "passage_attn":
                minmax_pattn = convert_to_nparray(output['values'])
            if output['label'] == "number_input":
                number_input_pattn = convert_to_nparray(output['values'])
        returned_attentions.append(number_input_pattn)
        returned_attentions.append(minmax_pattn)

    elif module_name in ["year_difference_two_events", "year_difference_single_event",
                         "compare_date_lt", "compare_date_gt"]:
        date1_pattn, date2_pattn = None, None
        for output in outputs:
            if output['label'] == "date_1_passage_attn":
                date1_pattn = convert_to_nparray(output['values'])
            if output['label'] == "date_2_passage_attn":
                date2_pattn = convert_to_nparray(output['values'])
        returned_attentions.append(date1_pattn)
        returned_attentions.append(date2_pattn)

    elif module_name in ["compare_num_lt", "compare_num_gt"]:
        num1_pattn, num2_pattn = None, None
        for output in outputs:
            if output['label'] == "num_1_passage_attn":
                num1_pattn = convert_to_nparray(output['values'])
            if output['label'] == "num_2_passage_attn":
                num2_pattn = convert_to_nparray(output['values'])
        returned_attentions.append(num1_pattn)
        returned_attentions.append(num2_pattn)

    return returned_attentions


MODULEWISE_INTERPRETABILITY = defaultdict(float)
MODULEWISE_COUNT = defaultdict(float)

# This would contain all scores. This will be used for significance testing
MODULE_SCORE = []


def yeardiff_single_event(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (year_difference_single_event select_passage)
    # dates, select
    gold_select_spans = faithfulness_annotation["select"]
    gold_date_spans = faithfulness_annotation["dates"]

    # 'year_difference_single_event', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution
    [date1_passage_attn, date2_passage_attn] = get_module_outputs(program_execution[0], "year_difference_single_event")
    [select_passage_attn] = get_module_outputs(program_execution[1], "select_passage")

    merged_date_attn = np.maximum(date1_passage_attn, date2_passage_attn)

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    date_score = compute_interpretability_loss(merged_date_attn, gold_date_spans)

    final_score = select_score + date_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["date"] += date_score
    MODULEWISE_COUNT["date"] += 1

    return final_score


def yeardiff_two_event(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (year_difference_two_events select_passage select_passage)
    # dates, select
    gold_select_spans = faithfulness_annotation["select"]
    gold_date_spans = faithfulness_annotation["dates"]

    # 'year_difference_two_events', 'select_passage', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution
    [date1_passage_attn, date2_passage_attn] = get_module_outputs(program_execution[0], "year_difference_two_events")
    [select1_passage_attn] = get_module_outputs(program_execution[1], "select_passage")
    [select2_passage_attn] = get_module_outputs(program_execution[2], "select_passage")

    merged_select_attn = np.maximum(select1_passage_attn, select2_passage_attn)
    merged_date_attn = np.maximum(date1_passage_attn, date2_passage_attn)

    select_score = compute_interpretability_loss(merged_select_attn, gold_select_spans)
    date_score = compute_interpretability_loss(merged_date_attn, gold_date_spans)

    final_score = select_score + date_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["date"] += date_score
    MODULEWISE_COUNT["date"] += 1

    return final_score


def count_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (aggregate_count select_passage)
    # "count", "select"

    gold_select_spans = faithfulness_annotation["select"]

    # 'aggregate_count', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    # [select_num_attn] = get_module_outputs(program_execution[0], "select_num")
    [select_passage_attn] = get_module_outputs(program_execution[1], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)

    final_score = select_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    return final_score


def count_filter_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (aggregate_count (filter_passage select_passage))
    # "count", "select"

    gold_select_spans = faithfulness_annotation["select"]
    gold_filter_spans = faithfulness_annotation["filter"]

    # 'aggregate_count', 'filter_passage', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    [filter_passage_attn] = get_module_outputs(program_execution[1], "filter_passage")
    [select_passage_attn] = get_module_outputs(program_execution[2], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    filter_score = compute_interpretability_loss(filter_passage_attn, gold_filter_spans)

    final_score = select_score + filter_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["filter"] += filter_score
    MODULEWISE_COUNT["filter"] += 1

    return final_score



def project_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_passagespan_answer (project_passage select_passage))
    # "project", "select"

    gold_select_spans = faithfulness_annotation["select"]
    gold_project_spans = faithfulness_annotation["project"]

    # [dict_keys(['select_passagespan_answer']), dict_keys(['project_passage']), dict_keys(['select_passage'])]
    program_execution: List[Dict] = prediction_instance.program_execution

    [project_passage_attn] = get_module_outputs(program_execution[1], "project_passage")
    [select_passage_attn] = get_module_outputs(program_execution[2], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    project_score = compute_interpretability_loss(project_passage_attn, gold_project_spans)

    final_score = select_score + project_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["project"] += project_score
    MODULEWISE_COUNT["project"] += 1

    return final_score



def project_minmax_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_passagespan_answer (project_passage (select_max_num select_passage)))
    # "project", "input_numbers", "minmax" "select"

    gold_select_spans = faithfulness_annotation["select"]
    gold_project_spans = faithfulness_annotation["project"]
    gold_inputnum_spans = faithfulness_annotation["input_numbers"]
    gold_minmax_spans = faithfulness_annotation["minmax"]

    # ['select_passagespan_answer', 'project_passage', 'select_max/min_num', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    minmax_module_name = list(program_execution[2].keys())[0]
    [project_passage_attn] = get_module_outputs(program_execution[1], "project_passage")
    [number_input_pattn, minmax_pattn] = get_module_outputs(program_execution[2], minmax_module_name)
    [select_passage_attn] = get_module_outputs(program_execution[3], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    project_score = compute_interpretability_loss(project_passage_attn, gold_project_spans)
    input_num_score = compute_interpretability_loss(number_input_pattn, gold_inputnum_spans)
    minmax_score = compute_interpretability_loss(minmax_pattn, gold_minmax_spans)

    final_score = select_score + project_score + input_num_score + minmax_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["project"] += project_score
    MODULEWISE_COUNT["project"] += 1

    MODULEWISE_INTERPRETABILITY["number"] += input_num_score
    MODULEWISE_COUNT["number"] += 1

    MODULEWISE_INTERPRETABILITY["minmax"] += minmax_score
    MODULEWISE_COUNT["minmax"] += 1

    return final_score


def project_minmaxfilter_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))
    # "project", "input_numbers", "minmax" "filter", "select"

    gold_select_spans = faithfulness_annotation["select"]
    gold_filter_spans = faithfulness_annotation["filter"]
    gold_project_spans = faithfulness_annotation["project"]
    gold_inputnum_spans = faithfulness_annotation["input_numbers"]
    gold_minmax_spans = faithfulness_annotation["minmax"]

    # ['select_passagespan_answer', 'project_passage', 'select_max/min_num', 'filter_passage', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    minmax_module_name = list(program_execution[2].keys())[0]
    [project_passage_attn] = get_module_outputs(program_execution[1], "project_passage")
    [number_input_pattn, minmax_pattn] = get_module_outputs(program_execution[2], minmax_module_name)
    [filter_passage_attn] = get_module_outputs(program_execution[3], "filter_passage")
    [select_passage_attn] = get_module_outputs(program_execution[4], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    filter_score = compute_interpretability_loss(filter_passage_attn, gold_filter_spans)
    project_score = compute_interpretability_loss(project_passage_attn, gold_project_spans)
    input_num_score = compute_interpretability_loss(number_input_pattn, gold_inputnum_spans)
    minmax_score = compute_interpretability_loss(minmax_pattn, gold_minmax_spans)

    final_score = select_score + project_score + input_num_score + minmax_score + filter_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["filter"] += filter_score
    MODULEWISE_COUNT["filter"] += 1

    MODULEWISE_INTERPRETABILITY["project"] += project_score
    MODULEWISE_COUNT["project"] += 1

    MODULEWISE_INTERPRETABILITY["number"] += input_num_score
    MODULEWISE_COUNT["number"] += 1

    MODULEWISE_INTERPRETABILITY["minmax"] += minmax_score
    MODULEWISE_COUNT["minmax"] += 1

    return final_score


def num_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_num select_passage)
    # "select", numbers"

    gold_select_spans = faithfulness_annotation["select"]
    # gold_number_spans = faithfulness_annotation["numbers"]

    # 'select_num', 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    # [select_num_attn] = get_module_outputs(program_execution[0], "select_num")
    [select_passage_attn] = get_module_outputs(program_execution[1], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)

    final_score = select_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    return final_score


def num_filter_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_num (filter_passage select_passage))
    # "select", numbers", "filter"

    gold_select_spans = faithfulness_annotation["select"]
    gold_filter_spans = faithfulness_annotation["filter"]
    # gold_number_spans = faithfulness_annotation["numbers"]

    # 'select_num', 'filter_passage' 'select_passage'
    program_execution: List[Dict] = prediction_instance.program_execution

    # [select_num_attn] = get_module_outputs(program_execution[0], "select_num")
    [filter_passage_attn] = get_module_outputs(program_execution[1], "filter_passage")
    [select_passage_attn] = get_module_outputs(program_execution[2], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    filter_score = compute_interpretability_loss(filter_passage_attn, gold_filter_spans)

    final_score = select_score + filter_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["filter"] += filter_score
    MODULEWISE_COUNT["filter"] += 1

    return final_score


def num_minmax_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_num (select_min_num select_passage))
    # "select", "input_numbers", and "minmax"
    gold_select_spans = faithfulness_annotation["select"]
    gold_inputnum_spans = faithfulness_annotation["input_numbers"]
    gold_minmax_spans = faithfulness_annotation["minmax"]

    # [dict_keys(['select_num']), dict_keys(['select_max/min_num']), dict_keys(['select_passage'])]
    program_execution: List[Dict] = prediction_instance.program_execution

    # select_min_num or select_max_num
    minmax_module_name = list(program_execution[1].keys())[0]
    [number_input_pattn, minmax_pattn] = get_module_outputs(program_execution[1], minmax_module_name)
    [select_passage_attn] = get_module_outputs(program_execution[2], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    input_num_score = compute_interpretability_loss(number_input_pattn, gold_inputnum_spans)
    minmax_score = compute_interpretability_loss(minmax_pattn, gold_minmax_spans)

    final_score = select_score + input_num_score + minmax_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["number"] += input_num_score
    MODULEWISE_COUNT["number"] += 1

    MODULEWISE_INTERPRETABILITY["minmax"] += minmax_score
    MODULEWISE_COUNT["minmax"] += 1

    return final_score


def num_minmax_filter_select(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_num (select_min_num (filter_passage select_passage)))
    # "select", "filter", "input_numbers", and "minmax"
    gold_select_spans = faithfulness_annotation["select"]
    gold_filter_spans = faithfulness_annotation["filter"]
    gold_inputnum_spans = faithfulness_annotation["input_numbers"]
    gold_minmax_spans = faithfulness_annotation["minmax"]

    # [dict_keys(['select_num']), dict_keys(['select_max/min_num']), 'filter', dict_keys(['select_passage'])]
    program_execution: List[Dict] = prediction_instance.program_execution

    # select_min_num or select_max_num
    minmax_module_name = list(program_execution[1].keys())[0]
    [number_input_pattn, minmax_pattn] = get_module_outputs(program_execution[1], minmax_module_name)
    [filter_passage_attn] = get_module_outputs(program_execution[2], "filter_passage")
    [select_passage_attn] = get_module_outputs(program_execution[3], "select_passage")

    select_score = compute_interpretability_loss(select_passage_attn, gold_select_spans)
    filter_score = compute_interpretability_loss(filter_passage_attn, gold_filter_spans)
    input_num_score = compute_interpretability_loss(number_input_pattn, gold_inputnum_spans)
    minmax_score = compute_interpretability_loss(minmax_pattn, gold_minmax_spans)

    final_score = select_score + input_num_score + minmax_score + filter_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["filter"] += filter_score
    MODULEWISE_COUNT["filter"] += 1

    MODULEWISE_INTERPRETABILITY["number"] += input_num_score
    MODULEWISE_COUNT["number"] += 1

    MODULEWISE_INTERPRETABILITY["minmax"] += minmax_score
    MODULEWISE_COUNT["minmax"] += 1

    return final_score


def num_compare(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_passagespan_answer (compare_num_gt select_passage select_passage))
    # "numbers", "select"
    gold_number_spans = faithfulness_annotation["numbers"]
    gold_select_spans = faithfulness_annotation["select"]

    # select_passagespan_answer, compare_num_gt/lt, select_passage, select_passage
    program_execution: List[Dict] = prediction_instance.program_execution

    compare_module_name = list(program_execution[1].keys())[0]
    [num1_pattn, num2_pattn] = get_module_outputs(program_execution[1], compare_module_name)
    [select1_passage_attn] = get_module_outputs(program_execution[2], "select_passage")
    [select2_passage_attn] = get_module_outputs(program_execution[3], "select_passage")

    merged_select_attn = np.maximum(select1_passage_attn, select2_passage_attn)
    merged_numbers_attn = np.maximum(num1_pattn, num2_pattn)

    select_score = compute_interpretability_loss(merged_select_attn, gold_select_spans)
    numbers_score = compute_interpretability_loss(merged_numbers_attn, gold_number_spans)

    final_score = select_score + numbers_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["number"] += numbers_score
    MODULEWISE_COUNT["number"] += 1

    return final_score


def date_compare(prediction_instance: NMNPredictionInstance, faithfulness_annotation: Dict):
    # (select_passagespan_answer (compare_date_gt select_passage select_passage))
    # "dates", "select"
    gold_date_spans = faithfulness_annotation["dates"]
    gold_select_spans = faithfulness_annotation["select"]

    # select_passagespan_answer, compare_date_gt/lt, select_passage, select_passage
    program_execution: List[Dict] = prediction_instance.program_execution

    compare_module_name = list(program_execution[1].keys())[0]
    [date1_pattn, date2_pattn] = get_module_outputs(program_execution[1], compare_module_name)
    [select1_passage_attn] = get_module_outputs(program_execution[2], "select_passage")
    [select2_passage_attn] = get_module_outputs(program_execution[3], "select_passage")

    merged_select_attn = np.maximum(select1_passage_attn, select2_passage_attn)
    merged_dates_attn = np.maximum(date1_pattn, date2_pattn)

    select_score = compute_interpretability_loss(merged_select_attn, gold_select_spans)
    dates_score = compute_interpretability_loss(merged_dates_attn, gold_date_spans)

    final_score = select_score + dates_score

    MODULEWISE_INTERPRETABILITY["select"] += select_score
    MODULEWISE_COUNT["select"] += 1

    MODULEWISE_INTERPRETABILITY["date"] += dates_score
    MODULEWISE_COUNT["date"] += 1

    return final_score


lisp2prediction_function = {
    "(year_difference_single_event select_passage)": yeardiff_single_event,
    "(year_difference_two_events select_passage select_passage)": yeardiff_two_event,
    "(aggregate_count select_passage)": count_select,
    "(aggregate_count (filter_passage select_passage))": count_filter_select,
    "(select_passagespan_answer (project_passage select_passage))": project_select,
    "(select_passagespan_answer (project_passage (select_max_num select_passage)))": project_minmax_select,
    "(select_passagespan_answer (project_passage (select_min_num select_passage)))": project_minmax_select,
    "(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))": project_minmaxfilter_select,
    "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))": project_minmaxfilter_select,
    "(select_num select_passage)": num_select,
    "(select_num (filter_passage select_passage))": num_filter_select,
    "(select_num (select_max_num select_passage))": num_minmax_select,
    "(select_num (select_min_num select_passage))": num_minmax_select,
    "(select_num (select_max_num (filter_passage select_passage)))": num_minmax_filter_select,
    "(select_num (select_min_num (filter_passage select_passage)))": num_minmax_filter_select,
    "(select_passagespan_answer (compare_num_gt select_passage select_passage))": num_compare,
    "(select_passagespan_answer (compare_num_lt select_passage select_passage))": num_compare,
    "(select_passagespan_answer (compare_date_gt select_passage select_passage))": date_compare,
    "(select_passagespan_answer (compare_date_lt select_passage select_passage))": date_compare,
}


def compute_interpretability_loss(passage_attention: np.array, spans: List[Tuple]):
    # Gold span (start, end) are inclusive
    interpretability_loss = 0.0
    for span in spans:
        span_prob = np.sum(passage_attention[span[0]:span[1] + 1])
        # span_prob = max(1e-20, span_prob)
        if span_prob > 1e-20:
            span_neg_log_prob = -1.0 * np.log(span_prob)
            interpretability_loss += span_neg_log_prob
        else:
            span_neg_log_prob = -1.0 * np.log(1e-20)
            interpretability_loss += span_neg_log_prob
    # interpretability_loss /= float(len(spans))
    return interpretability_loss


def compute_faithfulness_score(nmn_predictions: List[NMNPredictionInstance], faithfulness_data):

    total_q = 0
    total_faithfulness_loss = 0
    qid2prediction = {instance.query_id: instance for instance in nmn_predictions}
    for passage_id, passage_info in faithfulness_data.items():
        for qa in passage_info[constants.qa_pairs]:
            qid = qa[constants.query_id]
            if qid not in qid2prediction:
                print("QID:{} not found in predictions!".format(qid))
            pred_instance: NMNPredictionInstance = qid2prediction[qid]

            program_node = node_from_dict(qa[constants.program_supervision])
            program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())

            pred_lisp = pred_instance.top_logical_form
            if pred_lisp != program_lisp:
                print("Prediction lisp: {}  !=  Gold: {}".format(pred_lisp, program_lisp))
                continue

            if program_lisp not in lisp2prediction_function:
                print("Program: {}  not supported!".format(program_lisp))
                continue

            total_q += 1
            compute_func = lisp2prediction_function[program_lisp]
            annotations: Dict = program_node.extras["faithful_annotation"]
            faithfulness_loss = compute_func(pred_instance, annotations)
            total_faithfulness_loss += faithfulness_loss


    avg_faithfulness_loss = total_faithfulness_loss / total_q
    avg_faithfulness_loss = round_all(avg_faithfulness_loss, 1)

    MODULEWISE_INTERPRETABILITY["num-date"] = MODULEWISE_INTERPRETABILITY["number"] + \
                                                MODULEWISE_INTERPRETABILITY["date"]
    MODULEWISE_COUNT["num-date"] = MODULEWISE_COUNT["number"] + MODULEWISE_COUNT["date"]
    AVG_MODULE_FAITHFULNESS_LOSS = {}
    for module_name, module_loss in MODULEWISE_INTERPRETABILITY.items():
        module_count = MODULEWISE_COUNT[module_name]
        if module_count > 0:
            avg_module_loss = module_loss / module_count
        else:
            avg_module_loss = 0
        AVG_MODULE_FAITHFULNESS_LOSS[module_name] = avg_module_loss

    AVG_MODULE_FAITHFULNESS_LOSS = round_all(AVG_MODULE_FAITHFULNESS_LOSS, 1)

    print("Total Q: {}".format(total_q))
    print(f"Avg. Faithfulness loss: {avg_faithfulness_loss}")
    print(AVG_MODULE_FAITHFULNESS_LOSS)
    print(MODULEWISE_COUNT)


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def main(args):
    nmn_predictions: List[NMNPredictionInstance] = read_nmn_prediction_file(args.nmn_pred_jsonl)
    faithfulness_data: Dict = readDataset(args.faithful_gold_json)

    compute_faithfulness_score(nmn_predictions, faithfulness_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmn_pred_jsonl")
    parser.add_argument("--faithful_gold_json")
    args = parser.parse_args()

    main(args)
