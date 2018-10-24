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
        return constants.BOOL_TYPE, ans_str

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
            best_nerstr = best_ner[-1]
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

    return (ans_type, best_nerstr)


def flattenContexts(input_jsonl: str, output_jsonl: str) -> None:
    """ Tokenized input_jsonl contains contexts that are sentence split. We need to flatten this into a single sentence.
    context (List of sentences) and context_ner (list of lists) needs to be flatten

    In the output:
    context: Is a single string with space-delimited
    context_ner: List of (text, start, end, type), but start/end are now w.r.t. flattened single sentence context
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output jsonl: {}".format(output_jsonl))

    # Reading all objects a priori since the input_jsonl can be overwritten
    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    with open(output_jsonl, 'w') as outf:
        for jsonobj in jsonobjs:

            new_doc = copy.deepcopy(jsonobj)

            # List of contexts, (title, list of sentences)
            contexts = new_doc[constants.context_field]
            contexts_ners = new_doc[constants.context_ner_field]
            contexts_whitespaces = new_doc[constants.context_whitespaces_field]

            flattened_contexts = []
            flattened_contexts_whitespaces = []
            flattened_context_ners = []
            for ((title, sentences),
                 singlecontext_whitespaces,
                 single_context_ners) in zip(contexts, contexts_whitespaces, contexts_ners):

                # Get (sentid, tokenidx) --> global token idx
                # List[List[token]]
                tokenized_context = [sent.strip().split(' ') for sent in sentences]
                sentIdxTokenIdx2GlobalTokenIdx = TAUtils.getGlobalTokenOffset(tokenized_context)

                # Setting NER mentions' spans offset to be global token offset
                singlecontext_flattened_ners = []
                for sent_idx, sent_ners in enumerate(single_context_ners):
                    for ner in sent_ners:
                        # print(ner)
                        (start, end) = (sentIdxTokenIdx2GlobalTokenIdx[(sent_idx, ner[1])],
                                        sentIdxTokenIdx2GlobalTokenIdx[(sent_idx, ner[2] - 1)] + 1)
                        new_ner = (ner[0], start, end, ner[3])
                        singlecontext_flattened_ners.append(new_ner)

                # Flat context: Join sentences by space
                flattened_context = ' '.join(sentences).strip()

                # Each sentence ends with a space, so for ever sent's ws, make last element a space
                for sentws in singlecontext_whitespaces:
                    if len(sentws) == 0:
                        print(sentences)
                    sentws[-1] = ' '
                flattened_whitespace = [ws for sentws in singlecontext_whitespaces for ws in sentws]

                flattened_contexts.append((title, flattened_context))
                flattened_context_ners.append(singlecontext_flattened_ners)
                flattened_contexts_whitespaces.append(flattened_whitespace)

                '''
                token_idx = 0
                for sent in tokenized_context:
                    for token in sent:
                        print(f"{token}_{token_idx}", end=' ')
                        token_idx += 1
                print("\n")
                print(singlecontext_flattened_ners)
                print("\n\n")
                '''

            new_doc[constants.context_field] = flattened_contexts
            new_doc[constants.context_ner_field] = flattened_context_ners
            new_doc[constants.context_whitespaces_field] = flattened_contexts_whitespaces

            outf.write(json.dumps(new_doc))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('Flattening contexts: {}'.format(args.input_jsonl))

    if args.replace:
        print("Replacing original file")
        output_jsonl = args.input_jsonl
    else:
        print("Outputting a new file")
        assert args.output_jsonl is not None, "Output_jsonl needed"
        output_jsonl = args.output_jsonl

    # args.input_jsonl --- is the output from preprocess.tokenize
    flattenContexts(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl')
    parser.add_argument('--replace', action="store_true", default=False)
    args = parser.parse_args()

    main(args)