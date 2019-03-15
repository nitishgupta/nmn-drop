import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any

import dateparser
from dateparser.search import search_dates

from utils import TAUtils, util, spacyutils
from datasets.hotpotqa.utils import constants
import datasets.hotpotqa.analysis.hotpot_evaluate_v1 as hotpot_evaluate_v1

spacy_nlp = spacyutils.getSpacyNLP()
# ccg_nlp = TAUtils.getCCGNLPLocalPipeline()

anstypecounts = {constants.NUM_TYPE: 0,
                 constants.DATE_TYPE: 0,
                 constants.ENTITY_TYPE: 0,
                 constants.BOOL_TYPE: 0,
                 constants.STRING_TYPE: 0}


def _countAnsTypes(ans_type):
    anstypecounts[ans_type] = anstypecounts.get(ans_type) + 1


def normalizeD1byD2(dict1, dict2):
    d = {}
    for k,v in dict1.items():
        d[k] = float(v)/float(dict2[k])
    return d


def normalizeDictbyK(dict1, constant):
    d = {}
    for k,v in dict1.items():
        d[k] = float(v)/constant
    return d


def predictedAnsPerf(ans_str: str, predicted_answer: str):
    """ perf for grounded answer """

    f1, pr, re = hotpot_evaluate_v1.f1_score(prediction=predicted_answer, ground_truth=ans_str)
    em = hotpot_evaluate_v1.exact_match_score(prediction=predicted_answer, ground_truth=ans_str)

    return (pr, re, f1, em)


def getAnsFromGrounding(answer_type, answer_grounding, contexts, contexts_ws, contexts_mentions,
                        ent2mens):
    """ Return the answer string from the type, grounding, contexts, and contexts mentions

    For
        Bool: Based on grounding, yes / no
        Num / Date / Entity : The string of any of the mentions
        String: The span from the context
    """

    if answer_type == constants.BOOL_TYPE:
        answer_str = "yes" if answer_grounding == 1 else "no"

    elif answer_type in [constants.NUM_TYPE, constants.DATE_TYPE, constants.ENTITY_TYPE]:
        ans_entity_idx = [i for i, val in enumerate(answer_grounding) if val == 1][0]
        ans_mentions = ent2mens[ans_entity_idx]
        ans_mention = ans_mentions[0]
        (context_idx, men_idx) = ans_mention
        mention = contexts_mentions[context_idx][men_idx]
        answer_str = mention[0]

    elif answer_type == constants.STRING_TYPE:
        if answer_grounding == constants.NO_ANS_GROUNDING:
            answer_str = constants.NO_ANS_GROUNDING
        else:
            (context_idx, (start, end)) = answer_grounding[0]
            tokenized_context = contexts[context_idx][1].split(" ")
            context_ws = contexts_ws[context_idx]
            ans_tokens = tokenized_context[start:end]
            ans_token_ws = context_ws[start:end]
            answer_str = ''.join([t + ws for (t, ws) in zip(ans_tokens, ans_token_ws)])

    else:
        print("UNRECOGNIZED ANSWER TYPE")
        answer_str = constants.NO_ANS_GROUNDING

    return answer_str


def ansTypeAnalysis(input_jsonl: str, output_txt: str, skipeasy: bool) -> None:
    """ The input jsonl contains types and grounded answers.
    Perform the following:
        1. Distribution over types
        2. Max-score from grounding
    Output:
        1. For each Date / Num / Entity type answer: Output the answer / predicted answer / F1

    Parameters:
    -----------
    input_jsonl: Pre-process jsonl with answer typing and grounding
    output_txt: File to write analysis
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_txt))

    # Reading all objects a priori since the input_jsonl can be overwritten
    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    anstype_counts = {}
    anstype_f1perf = {}
    anstype_emperf = {}
    perf_dict = {}
    numexamples = len(jsonobjs)

    with open(output_txt, 'w') as outf:
        for jsonobj in jsonobjs:
            if skipeasy is True and jsonobj[constants.qlevel_field] == 'easy':
                continue

            contexts = jsonobj[constants.context_field]
            contexts_ws = jsonobj[constants.context_whitespaces_field]

            contexts_ent_ners = jsonobj[constants.context_ent_ner_field]
            contexts_num_ners = jsonobj[constants.context_num_ner_field]
            contexts_date_ners = jsonobj[constants.context_date_ner_field]

            context_entmens2entidx = jsonobj[constants.context_nemens2entidx]
            context_nummens2entidx = jsonobj[constants.context_nummens2entidx]
            context_datemens2entidx = jsonobj[constants.context_datemens2entidx]

            context_eqent2entmens = jsonobj[constants.context_eqent2entmens]
            context_eqent2nummens = jsonobj[constants.context_eqent2nummens]
            context_eqent2datemens = jsonobj[constants.context_eqent2datemens]

            answer = jsonobj[constants.ans_field]
            answer_type = jsonobj[constants.ans_type_field]
            answer_grounding = jsonobj[constants.ans_grounding_field]

            anstype_counts[answer_type] = anstype_counts.get(answer_type, 0) + 1

            mentions = None
            ent2mens = None
            if answer_type == constants.ENTITY_TYPE:
                mentions = contexts_ent_ners
                ent2mens = context_eqent2entmens
            elif answer_type == constants.NUM_TYPE:
                mentions = contexts_num_ners
                ent2mens = context_eqent2nummens
            elif answer_type == constants.DATE_TYPE:
                mentions = contexts_date_ners
                ent2mens = context_eqent2datemens
            else:
                mentions = None
                ent2mens = None

            grounded_answer = getAnsFromGrounding(answer_type, answer_grounding, contexts, contexts_ws,
                                                  mentions, ent2mens)

            p, r, f, e = predictedAnsPerf(answer, grounded_answer)
            (p, r, f, e) = util.round_all((p, r, f, e), 3)

            anstype_f1perf[answer_type] = anstype_f1perf.get(answer_type, 0.0) + f
            anstype_emperf[answer_type] = anstype_emperf.get(answer_type, 0.0) + e

            perf_dict['f1'] = perf_dict.get('f1', 0.0) + f
            perf_dict['pr'] = perf_dict.get('pr', 0.0) + p
            perf_dict['re'] = perf_dict.get('re', 0.0) + r
            perf_dict['em'] = perf_dict.get('em', 0.0) + e

            if e != 1.0:
                outf.write(f"A:{answer} \t P:{grounded_answer} \t T:{answer_type} \t F1:{f}\n")

            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    # Prints fraction of each ans type
    print(f"Perf Dict: {normalizeDictbyK(perf_dict, numexamples)}")
    print(f"Answer Type Dist: {normalizeDictbyK(anstype_counts, numexamples)}")
    print(f"Answer Type EM Perf: {normalizeD1byD2(anstype_emperf, anstype_counts)}")
    print(f"Answer Type F1 Perf: {normalizeD1byD2(anstype_f1perf, anstype_counts)}")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('Assigning Answer Types: {}'.format(args.input_jsonl))

    # args.input_jsonl --- is the output from preprocess.tokenize
    ansTypeAnalysis(input_jsonl=args.input_jsonl, output_txt=args.output_txt,
                    skipeasy=args.skipeasy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_txt', required=True)
    parser.add_argument('--skipeasy', action='store_true', default=False)

    args = parser.parse_args()


    main(args)