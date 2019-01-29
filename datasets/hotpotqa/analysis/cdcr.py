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


def _printAnsTypeCounts():
    totalAns = 0
    for _, count in anstypecounts.items():
        totalAns += count

    for k, count in anstypecounts.items():
        anstypecounts[k] = util.round_all((float(count)/totalAns)*100, 3)

    print(anstypecounts)


def ansType(ans_str: str, q_ners, context_ners, f1_threshold):
    """ Type answer based on the context NEs
    If a high-scoring NE is found, assign its type, else STRING

    Parameters:
    -----------
    ans_str: Original Answer string
    *q_ners: ``not used`` list of question ner spans
    context_ners: List of ners for each context, for each sentence in a context
    f1_threshold: ```float``` If bestF1 is above this, keep the NE type, else STRING

    Return:
    -------
    ans_type: One of NUM_TYPE, DATE_TYPE, ENTITY_TYPE, BOOl_TYPE, STRING_TYPE
    """

    if ans_str == 'yes' or ans_str == 'no':
        return constants.BOOL_TYPE, ans_str, 1.0

    else:
        all_ners = []
        all_ners.extend(q_ners)
        for one_context_ner in context_ners:
            for sentence_ner in one_context_ner:
                all_ners.extend(sentence_ner)

        best_f1 = 0.0
        best_ner = None
        for ner in all_ners:
            ner_text = ner[0]
            f1, pr, re = hotpot_evaluate_v1.f1_score(prediction=ner_text, ground_truth=ans_str)

            if f1 > best_f1:
                best_f1 = f1
                best_ner = ner

        if best_ner is not None:
            best_nerstr = best_ner[0]
        else:
            best_nerstr = "NONE"
        if best_f1 >= f1_threshold:
            ner_type = best_ner[-1]
            if ner_type == constants.DATE_TYPE:
                ans_type = constants.DATE_TYPE

            elif ner_type == constants.NUM_TYPE:
                ans_type = constants.NUM_TYPE
            # No explicit check since the entity type is stored with fine-grained types
            else:
                ans_type = constants.ENTITY_TYPE
        else:
            ans_type = constants.STRING_TYPE

    return (ans_type, best_nerstr, best_f1)


def cdcrAnalysis(input_jsonl: str, output_txt: str) -> None:
    """ Analysis code for CDCR. Provide tokenized json with pruned NER.
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_txt))

    # Reading all objects a priori since the input_jsonl can be overwritten
    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    with open(output_txt, 'w') as outf:
        for jsonobj in jsonobjs:

            contexts_ners = jsonobj[constants.context_ner_field]

            # NERs per context para
            contexts_ners = [[ner for s in sc for ner in s] for sc in contexts_ners]

            outf.write(f"Question: {jsonobj[constants.id_field]}\n\n")

            for i, singlecontext_ners in enumerate(contexts_ners):
                outf.write(f"*****  CONTEXT {i}  ******\n")
                for ner in singlecontext_ners:
                    if (ner[-1] != constants.NUM_TYPE) and (ner[-1] != constants.DATE_TYPE):
                        outf.write(ner[0])
                        outf.write("\n")
                outf.write("\n")

            outf.write("\n\n")

            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('CDCR Analysis: {}'.format(args.input_jsonl))

    # args.input_jsonl --- is the output from preprocess.tokenize
    cdcrAnalysis(input_jsonl=args.input_jsonl, output_txt=args.output_txt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_txt', required=True)

    args = parser.parse_args()


    main(args)