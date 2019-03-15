import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Dict

from utils import util, spacyutils
from datasets.hotpotqa.utils import constants

spacy_nlp = spacyutils.getSpacyNLP()

# Don't make entities of these typed-mentions
NERTYPES_TO_IGNORE = [constants.DATE_TYPE, constants.NUM_TYPE, constants.STRING_TYPE, constants.BOOL_TYPE]


def exactStringMatchCDCR(contexts_ent_ners: List[List[Tuple]]):
    """ Perform exact string match CDCR.

    Parameters:
    -----------
    contexts: List of context strings
    contexts_ners: List of context_ners, each being a list of ners in the context

    Returns:
    --------
    entitystr2idx: ``Dict``
        Maps entity_norm_str vals to entity_idxs. This is needed to ground mentions in questions
    ne_entidx2entstr: List[str]
        List containing entity_norm_str vals in order of ne_entity_idx
    entity2mentions: `list of list`
        For each eq_ent, a list of (context_idx, mention_idx) that refer to this grounding
        Len of outer list is the number of entities of this type in the contexts
    mens2entities: 'list of list'
        For each mention of a particular type, the entity_idx (in entity2mentions) it resolves to.
        The inner and outer lengths should exactly match contexts_ners
    """

    # For book-keeping entity_repr_string to entity_idx
    entitystr2idx = {}
    ne_entidx2entstr = []

    # List (size of number of entities), each is a list of (context_idx, mention_idx) tuples.
    entity2mens = []
    # List (size of number of contexts) of entity_idxs for the mentions. Sizes should be same as contexts_ent_ners
    entmens2entities = []

    for context_idx, single_context_ent_ners in enumerate(contexts_ent_ners):
        context_entmens2entities = []
        for mention_idx, ner in enumerate(single_context_ent_ners):
            mention_str = util._getLnrm(ner[0])
            if mention_str not in entitystr2idx:
                entitystr2idx[mention_str] = len(entity2mens)
                ne_entidx2entstr.append(mention_str)
                entity2mens.append([])

            entityidx = entitystr2idx[mention_str]
            entity2mens[entityidx].append((context_idx, mention_idx))
            context_entmens2entities.append(entityidx)

        entmens2entities.append(context_entmens2entities)

    return entitystr2idx, ne_entidx2entstr, entity2mens, entmens2entities


def normalizableMensCDCR(contexts_ners: List[List[Tuple]], q_ners: List[Tuple], normalization_dict: Dict):
    """ Performs CDCR for NUM / DATE mentions, i.e. mentions whose normalization can be looked up in a Dict.

    Parameters:
    -----------
    contexts_ners: Mentions of a particular type for all contexts
    normalization_dict: Dict mapping from mention_str to its normalized value

    Returns:
    --------
    entity2mentions: `list of list`
        For each eq_ent, a list of (context_idx, mention_idx) that refer to this grounding
        Len of outer list is the number of entities of this type in the contexts
    entityidx2normval: List[entity_normalized_val]
    mens2entities: 'list of list'
        For each mention of a particular type, the entity_idx (in entity2mentions) it resolves to.
        The inner and outer lengths should exactly match contexts_ners
    """

    # For booking entity_repr_string to entity_idx
    entitynorm2idx = {}
    entityidx2normval = []

    # For question
    q_men2entidx = []

    # For contexts
    # List (size of number of entities), each is a list of (context_idx, mention_idx) tuples.
    entity2mens = []
    # List (size of number of contexts) of entity_idxs for the mentions. Sizes should be same as contexts_ent_ners
    entmens2entities = []

    for mention_idx, ner in enumerate(q_ners):
        normalized_val = normalization_dict[ner[0]]
        if isinstance(normalized_val, list):
            normalized_val = tuple(normalized_val)
        if normalized_val not in entitynorm2idx:
            entitynorm2idx[normalized_val] = len(entity2mens)
            entityidx2normval.append(normalized_val)
            entity2mens.append([])
        entityidx = entitynorm2idx[normalized_val]
        q_men2entidx.append(entityidx)

    for context_idx, single_context_ent_ners in enumerate(contexts_ners):
        context_entmens2entities = []
        for mention_idx, ner in enumerate(single_context_ent_ners):
            normalized_val = normalization_dict[ner[0]]
            if isinstance(normalized_val, list):
                normalized_val = tuple(normalized_val)
            if normalized_val not in entitynorm2idx:
                entitynorm2idx[normalized_val] = len(entity2mens)
                entityidx2normval.append(normalized_val)
                entity2mens.append([])

            entityidx = entitynorm2idx[normalized_val]
            entity2mens[entityidx].append((context_idx, mention_idx))
            context_entmens2entities.append(entityidx)

        entmens2entities.append(context_entmens2entities)

    return q_men2entidx, entity2mens, entityidx2normval, entmens2entities


def groundQuestionMentions(q_entners, entitystr2idx):
    """ Ground mentions in question to the identified entities in the contexts by exact string match.
    q_ners: List of ner tuples
    entity2mentions: entity string (lnrm) --> mentions in context
    """

    q_entner2entidx = []
    for ner in q_entners:
        mention_str = util._getLnrm(ner[0])
        if mention_str in entitystr2idx:
            q_entner2entidx.append(entitystr2idx[mention_str])
        else:
            q_entner2entidx.append(-1)

    return q_entner2entidx


def performCDCR(input_jsonl: str, output_jsonl: str) -> None:
    """ Perform CDCR in the contexts for identifying and linking entities of different types of mentions: ENT, NUM, DATE

    For each list of contexts, and for each type, generate
        eq_entity2mens: ``list of list`` list of entities, for each entity (context_idx, men_idx)
        mens2entities: ``list of list`` Each mention's entity_idx (size same as the number of mentions of that type)
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

            num_normalization_dict = jsonobj[constants.nums_normalized_field]
            date_normalization_dict = jsonobj[constants.dates_normalized_field]

            q_entners = jsonobj[constants.q_ent_ner_field]
            q_numners = jsonobj[constants.q_num_ner_field]
            q_dateners = jsonobj[constants.q_date_ner_field]
            (contexts_ent_ners, contexts_num_ners, contexts_date_ners) = (jsonobj[constants.context_ent_ner_field],
                                                                          jsonobj[constants.context_num_ner_field],
                                                                          jsonobj[constants.context_date_ner_field])

            # entitystr2idx: {ent_norm_str: entity_idx} --- map from entity_norm_val to entity_idx
            # ne_entidx2entstr: List[str] -- list of canonical entity_strings in order of ne_entity_idx
            # eq_entity2ent_mens: [[(c_idx, m_idx)], ...] --- all mentions for each entity
            # entmens2entities: [[ent_idx, ...], ...] --- Entity_idx for each ent_men
            (entitystr2idx, ne_entidx2entstr,
             eq_entity2ent_mens, entmens2entities) = exactStringMatchCDCR(contexts_ent_ners)
            jsonobj[constants.context_nemens2entidx] = entmens2entities
            jsonobj[constants.context_eqent2entmens] = eq_entity2ent_mens
            jsonobj[constants.ne_entidx2entstr] = ne_entidx2entstr

            # Same as above for NUM mens
            (q_nummen2entidx,
             eq_entity2num_mens,
             num_entidx2numval,
             nummens2entities)= normalizableMensCDCR(contexts_num_ners, q_numners, num_normalization_dict)
            jsonobj[constants.q_nummens2entidx] = q_nummen2entidx
            jsonobj[constants.context_nummens2entidx] = nummens2entities
            jsonobj[constants.context_eqent2nummens] = eq_entity2num_mens
            jsonobj[constants.num_entidx2numval] = num_entidx2numval

            # Same as above for DATE mens
            (q_datemen2entidx,
             eq_entity2date_mens,
             date_entidx2dateval,
             datemens2entities) = normalizableMensCDCR(contexts_date_ners, q_dateners, date_normalization_dict)
            jsonobj[constants.q_datemens2entidx] = q_datemen2entidx
            jsonobj[constants.context_datemens2entidx] = datemens2entities
            jsonobj[constants.context_eqent2datemens] = eq_entity2date_mens
            jsonobj[constants.date_entidx2dateval] = date_entidx2dateval

            # List grounding the ques_ent_mens to entity_idxs. If doesnt ground, then -1 is the entity_idx used
            q_entner2entidx = groundQuestionMentions(q_entners, entitystr2idx)
            jsonobj[constants.q_entmens2entidx] = q_entner2entidx

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