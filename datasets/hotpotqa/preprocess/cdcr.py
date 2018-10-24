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

# Don't make entities of these typed-mentions
NERTYPES_TO_IGNORE = [constants.DATE_TYPE, constants.NUM_TYPE, constants.STRING_TYPE, constants.BOOL_TYPE]


def exactStringMatchCDCR(contexts_ners: List[List[Tuple]]):
    """ Perform exact string match CDCR.

    Parameters:
    -----------
    contexts: List of context strings
    contexts_ners: List of context_ners, each being a list of ners in the context

    Returns:
    --------
    entity2mentions: ``Dict``
        Dict from entity_key (lnrm mention str) --> List of (context_id, mention_id)
    context_mention2ens: ``List of List``
        For each context, list of entity_ids for each mention.
        NO_LINK for DATE and NUM type mentions
    """

    # key: entity string. val: List of (context_id, mention_id) tuples
    entity2mentions = {}
    # For each context , List of entity ids for each mention
    context_mention2ens = []

    for context_idx, single_context_ners in enumerate(contexts_ners):
        mention2ens = []
        for mention_idx, ner in enumerate(single_context_ners):
            if ner[3] in NERTYPES_TO_IGNORE:
                mention2ens.append(constants.NO_LINK)
            else:
                mention_str = util._getLnrm(ner[0])
                if mention_str not in entity2mentions:
                    entity2mentions[mention_str] = []
                entity2mentions[mention_str].append((context_idx, mention_idx))
                mention2ens.append(mention_str)
        context_mention2ens.append(mention2ens)

    return entity2mentions, context_mention2ens


def groundQuestionMentions(q_ners, entity2mentions):
    """ Ground mentions in question to the identified entities in the contexts by exact string match.
    q_ners: List of ner tuples
    entity2mentions: entity string (lnrm) --> mentions in context
    """

    qner_entity_links = []
    for ner in q_ners:
        if ner[3] in NERTYPES_TO_IGNORE:
            qner_entity_links.append(constants.NO_LINK)
        else:
            mention_str = util._getLnrm(ner[0])
            if mention_str in entity2mentions:
                qner_entity_links.append(mention_str)
            else:
                qner_entity_links.append(constants.NO_LINK)

    return qner_entity_links


def performCDCR(input_jsonl: str, output_jsonl: str) -> None:
    """ Perform CDCR in the contexts for identifying and linking entities.

    First, run CDCR on contexts to identify context entities. Then, link the question entities to the context entities.

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

            # List of contexts, (title, list of sentences)
            qners = jsonobj[constants.q_ner_field]
            contexts_ners = jsonobj[constants.context_ner_field]

            entity2mentions, context_mens_entlinks = exactStringMatchCDCR(contexts_ners)
            q_mens_entlinks = groundQuestionMentions(qners, entity2mentions)

            jsonobj[constants.ENT_TO_CONTEXT_MENS] = entity2mentions
            jsonobj[constants.CONTEXT_MENS_TO_ENT] = context_mens_entlinks
            jsonobj[constants.Q_MENS_TO_ENT] = q_mens_entlinks

            outf.write(json.dumps(jsonobj))
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
    performCDCR(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl')
    parser.add_argument('--replace', action="store_true", default=False)
    args = parser.parse_args()

    main(args)