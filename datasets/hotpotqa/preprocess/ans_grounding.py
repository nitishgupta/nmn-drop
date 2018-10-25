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


def _findStringAnswerGroundingInContext(answer_tokenized, contexts):
    """ Find the string answer in the given contexts.

    Parameters:
    -----------
    answer_tokenized: String of tokenized answer (space delimited)
    contexts: List of tokenized contexts (space delimited)


    Returns:
    --------
    answer_grounding = List of (context_idx, (start, end)) spans. (exclusive end)
    """

    answer_tokens = answer_tokenized.split(' ')
    answer_grounding = []

    for context_idx, (_, context) in enumerate(contexts):
        context_tokens = context.split(' ')
        matching_spans = util.getMatchingSubSpans(seq=context_tokens, pattern=answer_tokens)
        for span in matching_spans:
            answer_grounding.append((context_idx, span))

    return answer_grounding


def answerGrounding(contexts, answer_tokenized, answer_type, best_mentions):
    """ Ground answers in the contexts.

    Groundings for different answer types:
    BOOL:  grounding is 0 (false) / 1 (true)
    ENTITY, NUM, DATE: all mentions that are the correct answer. Return as: List of (context_id, mention_idx)
    STRING: All spans in the contexts stored as: (context_id, (start,end))
    """

    answer_grounding = None

    if answer_type == constants.BOOL_TYPE:
        if answer_tokenized == "yes":
            answer_grounding = 1
        else:
            answer_grounding = 0

    elif answer_type == constants.ENTITY_TYPE or answer_type == constants.NUM_TYPE or answer_type == constants.DATE_TYPE:
        answer_grounding =  best_mentions

    elif answer_type == constants.STRING_TYPE:
        # List of (context, (start, end)) spans
        answer_grounding = _findStringAnswerGroundingInContext(answer_tokenized, contexts)

    return answer_grounding


def ansTyping(ans_str: str, context_ners, f1_threshold):
    """ Type answer based on the context NEs
    If a high-scoring NE is found, assign its type, else STRING

    Parameters:
    -----------
    ans_str: Original Answer string
    context_ners: List of ners for each context
    f1_threshold: ```float``` If bestF1 is above this, keep the NE type, else STRING

    Return:
    -------
    ans_type: One of NUM_TYPE, DATE_TYPE, ENTITY_TYPE, BOOl_TYPE, STRING_TYPE
    best_mentions:
        BOOLEAN questions: yes / no
        Other questions: List of best scoring mentions as (context_idx, mention_idx)
                         These will later be used as answer grounding, if the answer_type == ENTITY
    """

    if ans_str == 'yes' or ans_str == 'no':
        return constants.BOOL_TYPE, ans_str

    else:
        best_f1 = 0.0
        best_mentions = []
        for context_idx, single_context_mentions in enumerate(context_ners):
            for mention_idx, mention in enumerate(single_context_mentions):
                mention_str = mention[0]
                f1, pr, re = hotpot_evaluate_v1.f1_score(prediction=mention_str, ground_truth=ans_str)

                if f1 > best_f1:
                    best_f1 = f1
                    best_mentions = [(context_idx, mention_idx)]
                elif f1 == best_f1:
                    best_mentions.append((context_idx, mention_idx))

        if best_f1 >= f1_threshold:
            context_idx, mention_idx = best_mentions[0]
            mention = context_ners[context_idx][mention_idx]
            mention_type = mention[-1]

            if mention_type == constants.DATE_TYPE:
                ans_type = constants.DATE_TYPE

            elif mention_type == constants.NUM_TYPE:
                ans_type = constants.NUM_TYPE

            # Context contains only three classes: DATES, NUM, ENTITIES (stored as finegrained types)
            else:
                ans_type = constants.ENTITY_TYPE
        else:
            ans_type = constants.STRING_TYPE

    return (ans_type, best_mentions)


def assignAnsTypesAndGround(input_jsonl: str, output_jsonl: str, f1_threshold: float) -> None:
    """ Assign types to answer for the training/dev data and ground them in context

    *** Since the input_jsonl can be replaced, make sure it is completely read before overwritting ***
    *** CONTEXT IS NOW FLATTENED, TAKE GOOD CARE ***

    For a given answer, find the best scoring NEs from context. (We don't consider question NEs)
    If:   Score above the F1-threshold, assign its type
    Else: Assign String type
    Since regular entities are fine-grained (PER, LOC, FAC etc.), make sure to resolve them to ENTITY type

    Groundings for different answer types:
    BOOL:  grounding is 0 (false) / 1 (true)
    ENTITY, NUM, DATE: all mentions that are the correct answer. Stored as: List of (context_id, mention_idx)
    STRING: All spans in the contexts that match the answer. Stored as: (context_id, (start,end))

    Parameters:
    -----------
    input_jsonl: Tokenized, pruned ner jsonl file from which answers will be assigned type
    f1_threshold: F1 threshold for type assignment
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_jsonl))

    # Reading all objects a priori since the input_jsonl can be overwritten
    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    missinggroudningcases = {}

    with open(output_jsonl, 'w') as outf:
        for jsonobj in jsonobjs:

            new_doc = copy.deepcopy(jsonobj)

            contexts = new_doc[constants.context_field]
            contexts_ners = new_doc[constants.context_ner_field]

            answer = new_doc[constants.ans_field]
            answer_tokenized = new_doc[constants.ans_tokenized_field]

            (answer_type, best_mentions) = ansTyping(ans_str=answer,
                                                     context_ners=contexts_ners,
                                                     f1_threshold=f1_threshold)

            answer_grounding = answerGrounding(contexts, answer_tokenized, answer_type, best_mentions)

            if answer_type != constants.BOOL_TYPE:
                if len(answer_grounding) == 0:
                    if answer_type not in missinggroudningcases:
                        missinggroudningcases[answer_type] = 0
                    missinggroudningcases[answer_type] += 1
                    answer_grounding = constants.NO_LINK

            _countAnsTypes(answer_type)

            new_doc[constants.ans_type_field] = answer_type
            new_doc[constants.ans_grounding_field] = answer_grounding

            outf.write(json.dumps(new_doc))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    # Prints fraction of each ans type
    _printAnsTypeCounts()

    print("Number of docs written: {}".format(numdocswritten))
    print(f"Missing grounding cases: {missinggroudningcases}")


def main(args):
    print('Assigning Answer Types: {}'.format(args.input_jsonl))
    print('F1 Threshold: {}'.format(args.f1thresh))

    if args.replace:
        print("Replacing original file")
        output_jsonl = args.input_jsonl
    else:
        print("Outputting a new file")
        assert args.output_jsonl is not None, "Output_jsonl needed"
        output_jsonl = args.output_jsonl

    # args.input_jsonl --- is the output from preprocess.tokenize
    assignAnsTypesAndGround(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl,
                            f1_threshold=args.f1thresh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl')
    parser.add_argument('--replace', action="store_true", default=False)
    parser.add_argument('--f1thresh', type=float, required=True)
    args = parser.parse_args()

    main(args)