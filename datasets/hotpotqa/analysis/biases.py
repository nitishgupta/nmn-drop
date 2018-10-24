import sys
import ujson as json

import argparse
from typing import List, Tuple, Any, Dict
from datasets.hotpotqa.utils import constants
from utils import util
import datasets.hotpotqa.analysis.hotpot_evaluate_v1 as evaluation

"""
For finding the maximum performance possible by predicting only NER spans as answers
"""

ner_type: List[Any]

def updateScores(final_f1, final_pr, final_re, final_em, f1, pr, re, em):
    final_f1 += f1
    final_pr += pr
    final_re += re
    final_em += em

    return final_f1, final_pr, final_re, final_em


def _avgScores(final_f1, final_pr, final_re, final_em, numexamples):
    final_f1 = float(final_f1) / numexamples
    final_pr = float(final_pr) / numexamples
    final_re = float(final_re) / numexamples
    final_em = float(final_em) / numexamples

    [final_f1, final_pr, final_re, final_em] = util.round_all([final_f1, final_pr, final_re, final_em], 4)

    return final_f1, final_pr, final_re, final_em



def _bestNERSpanPerfPerQA(qaexample: Dict) -> Tuple:
    """ Given a QA example json dict, find the best NER answer possible.

    Parameters:
    -----------
    qaexample: JsonDict for the qa example

    Returns:
    --------
    ner_span: If ans_type == span, best scoring NER span, else 'yes' or 'no'
    gold_ans: true ans
    (f1, pr, re, em): For best prediction
    """

    ans = qaexample[constants.ans_field]

    if ans == 'yes' or ans == 'no':
        prediction = ans
        f1, pr, re = evaluation.f1_score(prediction=prediction, ground_truth=ans)
        em = evaluation.exact_match_score(prediction=prediction, ground_truth=ans)
        return prediction, ans, (f1, pr, re, em)
    else:
        # List of contexts, for each context a list of sentences, for each sent a list of ners
        context_ners = qaexample[constants.context_ner_field]

        best_f1, best_pr, best_re, best_em = 0.0, 0.0, 0.0, 0.0
        prediction = ""

        for context in context_ners:
            for sentence in context:
                for ner in sentence:
                    ner_text = ner[0]
                    f1, pr, re = evaluation.f1_score(prediction=ner_text, ground_truth=ans)
                    em = evaluation.exact_match_score(prediction=ner_text, ground_truth=ans)

                    if f1 > best_f1:
                        best_f1, best_pr, best_re, best_em = f1, pr, re, em
                        prediction = ner_text

        return prediction, ans, (best_f1, best_pr, best_re, best_em)


def compartiveQuestionAnalysis(qaexample):
    ques = qaexample[constants.q_field]
    ans = qaexample[constants.ans_field]
    q_ners = qaexample[constants.q_ner_field]
    ners = [ner[0] for ner in q_ners]

    print(ques)
    print(ans)
    print(ners)
    print()

    correct_idx = -1

    for i, ner in enumerate(ners):
        if ner == ans:
            correct_idx = i

    return correct_idx



def boolBias(input_jsonl: str) -> None:
    print("Reading dataset: {}".format(input_jsonl))
    qa_examples: List[Dict] = util.readJsonlDocs(input_jsonl)

    print("Computing maximum NER Perf ... ")

    num_yes_ans = 0
    num_no_ans = 0
    num_examples = 0

    qtype_count = {}

    correct_idx_dist = {}
    comparison_noans = 0
    comparison_nonbool = 0

    for qaexample in qa_examples:
        answer = qaexample[constants.ans_field]
        num_examples += 1

        qtype = qaexample[constants.qtyte_field]


        if qtype == 'comparison':
            if answer == 'yes':
                num_yes_ans += 1
            elif answer == 'no':
                num_no_ans += 1
            else:
                comparison_nonbool += 1
                correct_idx = compartiveQuestionAnalysis(qaexample)
                if correct_idx == -1:
                    comparison_noans += 1
                else:
                    correct_idx_dist[correct_idx] = correct_idx_dist.get(correct_idx, 0) + 1

        qtype_count[qtype] = qtype_count.get(qtype, 0) + 1


    for k,v in qtype_count.items():
        qtype_count[k] = qtype_count[k]*100.0 / float(num_examples)

    print(f"Number of examples: {num_examples}")
    numboolqas = num_yes_ans + num_no_ans
    perc_boolqas = float(numboolqas)*100.0/num_examples
    print(f"Bool: {numboolqas} Yes: {num_yes_ans} No: {num_no_ans}")
    print(f"Percentage of Bool Ques: {perc_boolqas}")
    print(f"Qtype : {qtype_count}")
    print(f"Comparison non bool: {comparison_nonbool}")
    print(f"Comparison No Ans: {comparison_noans}")
    print(f"Comparison Idx Dist: {correct_idx_dist}")

def main(args):
    boolBias(input_jsonl=args.input_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)

