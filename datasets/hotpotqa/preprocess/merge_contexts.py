import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any, Dict

from utils import util
from datasets.hotpotqa.utils import constants

def merged_context(contexts, contexts_whitespaces):
    new_context_text = ''
    for _, c in contexts:
        new_context_text += c
        new_context_text += ' '

    new_context_text = new_context_text.strip()
    new_context = [("MERGED_TITLE", new_context_text)]
    new_context_whitespaces = [ws for ws in contexts_whitespaces]
    # Wrapping to make a single context
    new_context_whitespaces = [new_context_whitespaces]

    return new_context, new_context_whitespaces


def merge_Mens(ent_mens, num_mens, date_mens, cumulative_numtokens):
    """ mens are list of (for contexts) list of ner tuples (for ners in this context).
        return a single list of list containing all mentions with modified start end, based on cumulative tokens
    """
    def merge_mens(context_mens, cumulative_numtokens):
        merged_mens = []
        for c_idx, mens in enumerate(context_mens):
            for men in mens:
                modified_men = (
                men[0], men[1] + cumulative_numtokens[c_idx], men[2] + cumulative_numtokens[c_idx], men[3])
                merged_mens.append(modified_men)
        return merged_mens

    merged_ent_mens = merge_mens(ent_mens, cumulative_numtokens)
    merged_num_mens = merge_mens(num_mens, cumulative_numtokens)
    merged_date_mens = merge_mens(date_mens, cumulative_numtokens)

    return [merged_ent_mens], [merged_num_mens], [merged_date_mens]


def merge_entidx_to_mens_list(entidx2mens, cumulative_nummens):
    """ entidx2mens is a list (for diffferent entities) of list (of mentions of this entity) of (c_idx, midx)
        Need to make a list of list containing (0, new_mention_idx) based on merged mens.
        Since mentions are merged in order, new_men_idx = old_men_idx + cumulative_num_mentions_in_previous_contexts
    """
    merged_entidx2menidx = []
    for entity_mens in entidx2mens:
        merged_ent_menidxs = []
        for (c_idx, m_idx) in entity_mens:
            new_men_idx = m_idx + cumulative_nummens[c_idx]
            merged_ent_menidxs.append((0, new_men_idx))
        merged_entidx2menidx.append(merged_ent_menidxs)
    return merged_entidx2menidx


def merge_mens2entidx(merge_mens2entidx):
    """ merge_mens2entidx is a list (for each context) of list (for each mention in context) of entity_idx
        Return a list (of size = 1) of merged list of entity_idxs. This can be done just by concatenating inner lists.
    """
    merged_mens2entidx = []
    for mens in merge_mens2entidx:
        merged_mens2entidx.extend(mens)
    return [merged_mens2entidx]


def num_tokens_in_contexts(contexts):
    num_tokens_each_context = []
    for _, c in contexts:
        num_tokens = len(c.split(' '))
        num_tokens_each_context.append(num_tokens)

    return num_tokens_each_context


def cumulative_tokens(num_tokens_each_context):
    cumulative_numtokens = [0]
    for i in range(0, len(num_tokens_each_context) - 1):
        cumulative_numtokens.append(cumulative_numtokens[i] + num_tokens_each_context[i])

    return cumulative_numtokens


def cumulative_lengths(list_lens):
    cumulative_lens = [0]
    for i in range(0, len(list_lens) - 1):
        cumulative_lens.append(cumulative_lens[i] + list_lens[i])

    return cumulative_lens


def mergeContextsForJsonl(input_jsonl: str, output_jsonl: str) -> None:
    """ Merge contexts into a single context for all questions.

    Th output will still be in the same format as original (have lists even if there's a single context)
    to make rest of the code work without requiring changes.

    The affected fields are as follows:
        context_field:
            Merging contexts into a (TITLE, Context) tuple.
        context_whitespaces_field:
            Merged into a single list
        ans_grounding_field:
            This will change to new span for STRING type ques
        context_ent_ner_field, context_num_ner_field, context_date_ner_field:
            New start, end for all mentions
        context_eqent2entmens, context_eqent2nummens, context_eqent2datemens:
            Inner list contains (context_id, mention_id) which needs to change to (0, new_mention_id).
        context_entmens2entidx, context_nummens2entidx, context_datemens2entidx:
            List[List[ent_idx]], all inner lists can be merged into a single list
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

            new_jsonobj = copy.deepcopy(jsonobj)

            # Merging contexts
            contexts = jsonobj[constants.context_field]      # List of (title, context)
            contexts_whitespaces = jsonobj[constants.context_whitespaces_field]

            new_context, new_context_whitespaces = merged_context(contexts, contexts_whitespaces)
            new_jsonobj[constants.context_field] = new_context
            new_jsonobj[constants.context_whitespaces_field] = new_context_whitespaces

            # context_text = new_context[0][1].split(' ')
            # print([(t, i) for i, t in enumerate(context_text)])

            num_tokens_each_context = num_tokens_in_contexts(contexts)
            cumulative_numtokens = cumulative_tokens(num_tokens_each_context)

            # Merging context mens
            ent_mens = jsonobj[constants.context_ent_ner_field]
            num_mens = jsonobj[constants.context_num_ner_field]
            date_mens = jsonobj[constants.context_date_ner_field]
            (merged_ent_mens, merged_num_mens, merged_date_mens) = merge_Mens(ent_mens, num_mens, date_mens,
                                                                              cumulative_numtokens)

            # print(merged_ent_mens)
            # print()

            new_jsonobj[constants.context_ent_ner_field] = merged_ent_mens
            new_jsonobj[constants.context_num_ner_field] = merged_num_mens
            new_jsonobj[constants.context_date_ner_field] = merged_date_mens

            num_ent_mens = [len(x) for x in ent_mens]
            num_num_mens = [len(x) for x in num_mens]
            num_date_mens = [len(x) for x in date_mens]
            cumulative_numentmens = cumulative_lengths(num_ent_mens)
            cumulative_numnummens = cumulative_lengths(num_num_mens)
            cumulative_numdatemens = cumulative_lengths(num_date_mens)

            # Merge entity_idx to mention_idx lists
            context_eqent2entmens = jsonobj[constants.context_eqent2entmens]
            context_eqent2nummens = jsonobj[constants.context_eqent2nummens]
            context_eqent2datemens = jsonobj[constants.context_eqent2datemens]

            merged_ent_entidx2mens = merge_entidx_to_mens_list(context_eqent2entmens, cumulative_numentmens)
            merged_num_entidx2mens = merge_entidx_to_mens_list(context_eqent2nummens, cumulative_numnummens)
            merged_date_entidx2mens = merge_entidx_to_mens_list(context_eqent2datemens, cumulative_numdatemens)

            new_jsonobj[constants.context_eqent2entmens] = merged_ent_entidx2mens
            new_jsonobj[constants.context_eqent2nummens] = merged_num_entidx2mens
            new_jsonobj[constants.context_eqent2datemens] = merged_date_entidx2mens
            # print(merged_ent_entidx2mens)
            # print()

            # Merging mention idx to entity idx lists
            entmens2entidx = jsonobj[constants.context_nemens2entidx]
            nummens2entidx = jsonobj[constants.context_nummens2entidx]
            datemens2entidx = jsonobj[constants.context_datemens2entidx]

            merged_entmens2entidx = merge_mens2entidx(entmens2entidx)
            merged_nummens2entidx = merge_mens2entidx(nummens2entidx)
            merged_datemens2entidx = merge_mens2entidx(datemens2entidx)

            new_jsonobj[constants.context_nemens2entidx] = merged_entmens2entidx
            new_jsonobj[constants.context_nummens2entidx] = merged_nummens2entidx
            new_jsonobj[constants.context_datemens2entidx] = merged_datemens2entidx
            # print(merged_entmens2entidx)
            # print()

            outf.write(json.dumps(new_jsonobj))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('Merging contexts for: {}'.format(args.input_jsonl))

    print("Outputting a new file")
    assert args.output_jsonl is not None, "Output_jsonl needed"
    output_jsonl = args.output_jsonl

    # args.input_jsonl --- is the output from preprocess.tokenize
    mergeContextsForJsonl(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl', required=True)
    args = parser.parse_args()

    main(args)