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


def bestNERPerf(input_jsonl: str, print_preds: bool = False, output_txt: str = None) -> None:
    '''
    For a given data_file and gold-truth, compute the max perf from predicting NERs

    Parameters:
    ----------
    input_jsonl: Input with NERs marked
    print_preds: Gold-truth file
    output_txt: Optional, Output the answer and prediction for lowest F1 scores
    '''


    print("Reading dataset: {}".format(input_jsonl))
    qa_examples: List[Dict] = util.readJsonlDocs(input_jsonl)

    print("Computing maximum NER Perf ... ")

    final_f1, final_pr, final_re, final_em = 0.0, 0.0, 0.0, 0.0
    num_examples = 0

    predictions, answers = [], []
    scores = []

    # Dict of F1 score to list of (ans, pred) tuples
    f12answers = {}

    for qaexample in qa_examples:
        prediction, ans, (f1, pr, re, em) = _bestNERSpanPerfPerQA(qaexample)
        (final_f1, final_pr, final_re, final_em) = updateScores(final_f1, final_pr, final_re, final_em,
                                                                f1, pr, re, em)
        num_examples += 1

        predictions.append(prediction)
        answers.append(ans)
        scores.append((f1, pr, re, em))

        if f1 not in f12answers:
            f12answers[f1] = []
        f12answers[f1].append((ans, prediction))

        if print_preds:
            print(prediction)
            print(ans)
            print("f1: {} pr: {} re: {} em: {}".format(f1, pr, re, em))
            print("\n")

    final_f1, final_pr, final_re, final_em = _avgScores(final_f1, final_pr, final_re, final_em, num_examples)

    print(f"Number of examples: {num_examples}")
    print("Final F1: {} Pr: {} Re: {} EM: {}".format(final_f1, final_pr, final_re, final_em))

    sorted_f12answers = util.sortDictByKey(f12answers)

    # Ouput the Maximum NER performance.
    # For the lowest 50 F1 scores, output each of the ans || pred
    if output_txt is not None:
        outf = open(output_txt, 'w')
        outf.write(f"File: {input_jsonl}" + '\n')
        outf.write(f"Number of examples: {num_examples}" + '\n')
        outf.write("Max F1: {} Pr: {} Re: {} EM: {}\n".format(final_f1, final_pr, final_re, final_em))
        outf.write("Lowest F1 predictions: \n\n")

        for i in range(50):
            (f1, ans_pred_tuples) = sorted_f12answers[i]
            outf.write(f"F1: {f1} \t Count: {len(ans_pred_tuples)}\n")
            for ans, pred in ans_pred_tuples:
                pred = '@EMPTY@' if pred == '' else pred
                outf.write(f"{ans} \t||\t  {pred} \n")
            outf.write('\n\n')

        outf.close()


def main(args):

    bestNERPerf(input_jsonl=args.input_jsonl, print_preds=args.print_preds, output_txt=args.output_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--print_preds', action='store_true')
    parser.add_argument('--output_txt')

    args = parser.parse_args()

    main(args)
