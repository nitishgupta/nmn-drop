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




def bestNERPerf(input_jsonl: str) -> None:
    print("Reading dataset: {}".format(input_jsonl))
    qa_examples: List[Dict] = util.readJsonlDocs(input_jsonl)

    print("Computing maximum NER Perf ... ")

    final_f1, final_pr, final_re, final_em = 0.0, 0.0, 0.0, 0.0
    num_examples = 0

    predictions, answers = [], []
    scores = []

    for qaexample in qa_examples:
        prediction, ans, (f1, pr, re, em) = _bestNERSpanPerfPerQA(qaexample)
        (final_f1, final_pr, final_re, final_em) = updateScores(final_f1, final_pr, final_re, final_em,
                                                                f1, pr, re, em)
        num_examples += 1

        predictions.append(prediction)
        answers.append(ans)
        scores.append((f1, pr, re, em))

        print(prediction)
        print(ans)
        print("f1: {} pr: {} re: {} em: {}".format(f1, pr, re, em))
        print("\n")

    final_f1, final_pr, final_re, final_em = _avgScores(final_f1, final_pr, final_re, final_em, num_examples)

    print(f"Number of examples: {num_examples}")
    print("Final F1: {} Pr: {} Re: {} EM: {}".format(final_f1, final_pr, final_re, final_em))



def main(args):

    bestNERPerf(input_jsonl=args.input_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)

