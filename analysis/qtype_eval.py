from typing import List, Tuple, Dict, Union, Any
import json
import argparse
from collections import defaultdict, OrderedDict

from utils.util import round_all
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    avg_em


def count_accuracy_distribution(pred_instances: List[NMNPredictionInstance]):
    count_instances = []
    for instance in pred_instances:
        gold_lisp = instance.gold_logical_form
        if gold_lisp == "(aggregate_count select_passage)":
            count_answer = instance.gold_answers[0]["number"]
            if count_answer is not "" and float(count_answer) < 10:
                count_instances.append(instance)

    print("Num count questions: {}".format(len(count_instances)))

    countans2num = OrderedDict({x: 0 for x in range(0, 10)})
    countans2totalacc = OrderedDict({x: 0 for x in range(0, 10)})
    for instance in count_instances:
        count_ans = float(instance.gold_answers[0]["number"])
        countans2num[count_ans] += 1
        countans2totalacc[count_ans] += instance.exact_match


    countans2avgacc = OrderedDict()
    for count_ans, f1 in countans2totalacc.items():
        total = countans2num[count_ans]
        if total > 0:
            avgf1 = round_all((100.0 * f1) / total, 2)
            countans2avgacc[count_ans] = avgf1

    print(countans2num)
    print(countans2avgacc)


def print_about_model(nmn_jsonl: str, output_file: str = None):
    qtype2lisps = OrderedDict({
        "date_compare": ["(select_passagespan_answer (compare_date_lt select_passage select_passage))",
                         "(select_passagespan_answer (compare_date_gt select_passage select_passage))"],
        "year_diff": ["(year_difference_single_event select_passage)",
                      "(year_difference_two_events select_passage select_passage)"],
        "num_compare": ["(select_passagespan_answer (compare_num_lt select_passage select_passage))",
                        "(select_passagespan_answer (compare_num_gt select_passage select_passage))"],
        "num_minmax": ["(select_num (select_max_num select_passage))",
                       "(select_num (select_min_num select_passage))",
                       "(select_num (select_max_num (filter_passage select_passage)))",
                       "(select_num (select_min_num (filter_passage select_passage)))"],
                       # "(select_num select_passage)"],
        "project_minmax": ["(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                           "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
                           "(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))",
                           "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))"
                           ],
        "minmax": ["(select_num (select_max_num select_passage))",
                   "(select_num (select_min_num select_passage))",
                   "(select_num (select_max_num (filter_passage select_passage)))",
                   "(select_num (select_min_num (filter_passage select_passage)))",
                   "(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                   "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
                   "(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))",
                   "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))"],
        "add_diff": ["(passagenumber_difference (select_num select_passage) (select_num select_passage))",
                     "(passagenumber_addition (select_num select_passage) (select_num select_passage))"],
        "count": ["(aggregate_count select_passage)",
                  "(aggregate_count (filter_passage select_passage))"]
    })

    pred_instances: List[NMNPredictionInstance] = read_nmn_prediction_file(nmn_jsonl)
    num_q = len(pred_instances)
    f1, em = avg_f1(pred_instances), avg_em(pred_instances)
    total_f1 = round_all(100.0 * f1, 1)
    total_em = round_all(100.0 * em, 1)

    metric_dict = {
        "NMN": nmn_jsonl,
        "NumQ": num_q,
        "F1": total_f1,
        "EM": total_em
    }

    qtype2instances = defaultdict(list)
    for instance in pred_instances:
        pred_lisp = instance.gold_logical_form
        # pred_lisp = instance.top_logical_form
        for qtype, lisps in qtype2lisps.items():
            if pred_lisp in lisps:
                qtype2instances[qtype].append(instance)

    qtype2avgf1 = defaultdict(float)
    for qtype in qtype2lisps:
        instances = qtype2instances[qtype]
        qtype2avgf1[qtype] = round_all(100.0 * avg_f1(instances), 1)

    print("--------------")
    for key, value in metric_dict.items():
        print(f"{key:5}\t: {value}")
    print("--------------")
    print()
    print("--------------")
    for qtype in qtype2lisps:
        f1 = qtype2avgf1[qtype]
        print(f"{qtype:10}\t: {f1}")
    print("--------------")

    if output_file is not None:
        with open(output_file, 'w') as outf:
            outf.write(json.dumps(metric_dict, indent=4))
            outf.write("\n")
            outf.write(json.dumps(qtype2avgf1, indent=4))

    count_accuracy_distribution(pred_instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmn_jsonl")
    parser.add_argument("--output_file")
    args = parser.parse_args()



    print()
    print("Model 1 : {}".format(args.nmn_jsonl))
    print_about_model(args.nmn_jsonl, args.output_file)
    print()
