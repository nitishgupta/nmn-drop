import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any

import dateparser
from dateparser.search import search_dates

from utils import util, spacyutils
from datasets.hotpotqa.utils import constants
import datasets.hotpotqa.analysis.hotpot_evaluate_v1 as hotpot_evaluate_v1

spacy_nlp = spacyutils.getSpacyNLP()

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


def answerGrounding(contexts, answer_tokenized, answer_type, best_mentions, mens2entidxs, ent2mens):
    """ Ground answers in the contexts.

    Groundings for different answer types:
    BOOL:  grounding is 0 (false) / 1 (true)
    ENTITY, NUM, DATE: all mentions that are the correct answer. Return as: List of (context_id, mention_idx)
    STRING: All spans in the contexts stored as: (context_id, (start,end))
    """

    answer_grounding = None

    assert answer_type in anstypecounts, f"Unknown answer type: {answer_type}"

    if answer_type == constants.BOOL_TYPE:
        if answer_tokenized == "yes":
            answer_grounding = 1
        else:
            answer_grounding = 0

    elif answer_type == constants.STRING_TYPE:
        # List of (context, (start, end)) spans
        answer_grounding = _findStringAnswerGroundingInContext(answer_tokenized, contexts)

    else:
        # One of Entity, Num or Date
        # Use the best mentions to make a binary entity vector, the same size as ent2mens
        answer_grounding = [0] * len(ent2mens)
        for (context_idx, mention_idx) in best_mentions:
            entity_idx = mens2entidxs[context_idx][mention_idx]
            answer_grounding[entity_idx] = 1

    return answer_grounding



def getBestScoringMentionForAns(ans_str: str, context_ners):
    """ List of mentions that get the highest F1 score against the answer. Input mention list can be of any type.
    """
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

    return best_mentions, best_f1




def ansTyping(ans_str: str, context_ent_ners, context_num_ners, context_date_ners, f1_threshold):
    """ Type answer based on the context NEs
    If a high-scoring NE is found, assign its type, else STRING

    For each type, find the best scoring mentions. Assign the type based on F1 score from type.
    To tie-break if F1 score is same, consider the number of mentions that achieve that score.

    Parameters:
    -----------
    ans_str: Original Answer string
    context_ent_ners, context_num_ners, context_date_ners: List of ners (of each type) for each context
    f1_threshold: ```float``` If bestF1 is above this, keep the NE type, else STRING

    Return:
    -------
    ans_type: One of NUM_TYPE, DATE_TYPE, ENTITY_TYPE, BOOl_TYPE, STRING_TYPE
    best_mentions:
        BOOLEAN questions: yes / no
        Other questions: List of best scoring mentions of same type as (context_idx, mention_idx)
                         These will later be used as answer grounding, if the answer_type == ENTITY
    """

    if ans_str == 'yes' or ans_str == 'no':
        return constants.BOOL_TYPE, ans_str

    best_ent_mens, best_ent_f1 = getBestScoringMentionForAns(ans_str, context_ent_ners)
    best_num_mens, best_num_f1 = getBestScoringMentionForAns(ans_str, context_num_ners)
    best_date_mens, best_date_f1 = getBestScoringMentionForAns(ans_str, context_date_ners)

    best_type_mentions = [best_ent_mens, best_num_mens, best_date_mens]
    best_type_f1s = [best_ent_f1, best_num_f1, best_date_f1]
    types_order = [constants.ENTITY_TYPE, constants.NUM_TYPE, constants.DATE_TYPE]

    max_f1 = max(best_type_f1s)
    max_f1_idxs = [i for i,j in enumerate(best_type_f1s) if j == max_f1]

    if len(max_f1_idxs) == 1:
        # If one type has the highest F1, then choose the mentions from that type
        ansType = types_order[max_f1_idxs[0]]
        best_mentions = best_type_mentions[max_f1_idxs[0]]
        best_f1 = best_type_f1s[max_f1_idxs[0]]
    else:
        # If multiple types have same best F1, tie-break with (1) num of best mentions (2) in order of ENT, NUM, DATE
        len_best_mentions = [len(x) for x in best_type_mentions]
        maxnum_best_mentions = max(len_best_mentions)
        idxs_with_maxbestmentions = [i for i,j in enumerate(len_best_mentions) if j == maxnum_best_mentions]
        idx_with_max_mentions = idxs_with_maxbestmentions[0]

        best_mentions = best_type_mentions[idx_with_max_mentions]
        ansType = types_order[idx_with_max_mentions]
        best_f1 = best_type_f1s[idx_with_max_mentions]

    # Now that we have mentions with highest F1 score, assign mention type if
    if best_f1 >= f1_threshold:
        ans_type = ansType
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
    Deprecated: Since regular entities are fine-grained (PER, LOC, FAC etc.), make sure to resolve them to ENTITY type

    Groundings for different answer types:
    BOOL: ``float``
        grounding is 0.0 (false) / 1.0 (true)
    ENTITY, NUM, DATE: ``List[entities]
        From mentions that are the correct most frequent mention type is the answer (diff possible due to noise)
        For the mentions of the correct type, ground ans as the entity of that mention.
        For eg, if 5 ENT mentions are answers, find their corresponding entities
            and make a binary-hot vector [1,0,0,...,1,..0] with size equal to num of entities of that type

    STRING: ``List[Tuple((context_id, (start, end)))]`` --- exclusive end
        All spans in the contexts that match the answer.

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
            contexts_ent_ners = new_doc[constants.context_ent_ner_field]
            contexts_num_ners = new_doc[constants.context_num_ner_field]
            contexts_date_ners = new_doc[constants.context_date_ner_field]

            # Mention to entity mapping -- used to make the grounding vector
            context_entmens2entidx = new_doc[constants.context_nemens2entidx]
            context_nummens2entidx = new_doc[constants.context_nummens2entidx]
            context_datemens2entidx = new_doc[constants.context_datemens2entidx]

            # Entity to mentions --- Used to find the number of entities of each type in the contexts
            context_eqent2entmens = new_doc[constants.context_eqent2entmens]
            context_eqent2nummens = new_doc[constants.context_eqent2nummens]
            context_eqent2datemens = new_doc[constants.context_eqent2datemens]

            answer = new_doc[constants.ans_field]
            answer_tokenized = new_doc[constants.ans_tokenized_field]

            # List of (context_idx, (start, end))
            if answer in ['yes', 'no']:
                answer_spans = []
            else:
                answer_spans = _findStringAnswerGroundingInContext(answer_tokenized, contexts)


            # Answer typing based on the F1 achieved by the mentions of different type
            (answer_type, best_mentions) = ansTyping(ans_str=answer,
                                                     context_ent_ners=contexts_ent_ners,
                                                     context_num_ners=contexts_num_ners,
                                                     context_date_ners=contexts_date_ners,
                                                     f1_threshold=f1_threshold)

            mens2entidxs = None
            ent2mens = None

            # For the correct answer type, assign best_mentions, mens2entidx, and end2mens to help grounding
            if answer_type == constants.BOOL_TYPE:
                best_mentions = None
            elif answer_type == constants.STRING_TYPE:
                best_mentions = None
            elif answer_type == constants.ENTITY_TYPE:
                best_mentions = best_mentions
                mens2entidxs = context_entmens2entidx
                ent2mens = context_eqent2entmens
            elif answer_type == constants.NUM_TYPE:
                best_mentions = best_mentions
                mens2entidxs = context_nummens2entidx
                ent2mens = context_eqent2nummens
            elif answer_type == constants.DATE_TYPE:
                best_mentions = best_mentions
                mens2entidxs = context_datemens2entidx
                ent2mens = context_eqent2datemens
            else:
                print(f"answer_type is weird: {answer_type}")

            # For Bool - 0/1, String - [(context_id, (srt, end))]
            # For entity, num, and date --- Binary-list with grounding value for each entity
            answer_grounding = answerGrounding(contexts, answer_tokenized, answer_type,
                                               best_mentions, mens2entidxs, ent2mens)

            # Checking if the entity, num, or date type grounding is not empty
            if answer_type in [constants.ENTITY_TYPE, constants.NUM_TYPE, constants.DATE_TYPE]:
                # If answer_type is an entity, grounding vec cannot be all zero.
                if all(v == 0 for v in answer_grounding):
                    print("###########   ERROR    #################")
                    print(answer_type)
                    print(answer_grounding)
                    print(best_mentions)
                    print(answer)

            # Checking if string-type grounding is not empty
            # This is sometimes empty due to bad tokenization
            elif answer_type != constants.BOOL_TYPE:
                if len(answer_grounding) == 0:
                    if answer_type not in missinggroudningcases:
                        missinggroudningcases[answer_type] = 0
                    missinggroudningcases[answer_type] += 1
                    answer_grounding = constants.NO_ANS_GROUNDING

            _countAnsTypes(answer_type)

            new_doc[constants.ans_type_field] = answer_type
            new_doc[constants.ans_grounding_field] = answer_grounding
            new_doc[constants.ans_spans] = answer_spans

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