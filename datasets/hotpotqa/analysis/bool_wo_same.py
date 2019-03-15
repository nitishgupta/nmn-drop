import sys
from typing import List, Tuple, Dict
from utils import util
from datasets.hotpotqa.utils import constants
from functools import cmp_to_key

input_jsonl = sys.argv[1]


def get_ngram_spans(length: int, ngram: int) -> List[Tuple[int, int]]:
    """ Return all possible spans positions (exclusive-end) for a given ngram.
        Eg. for bigram and length = 4, spans returned will be (0,2) (1,3), (2,4)
    """
    if ngram < 1:
        raise Exception(f"Ngram cannot be less than 1. Given: {ngram}")
    spans = [(i, i + ngram) for i in range(0, length - ngram + 1)]
    return spans


def _get_gold_actions(span2qentactions, span2qstractions) -> Tuple[str, str, str]:
    def longestfirst_cmp(t1, t2):
        # when arranging in ascending, longest with greater end should come first
        len1 = t1[1] - t1[0]
        len2 = t2[1] - t2[0]
        if len2 < len1:
            return -1
        elif len1 < len2:
            return 1
        else:
            if t1[1] < t2[1]:
                return 1
            elif t1[1] > t2[1]:
                return -1
            else:
                return 0

    def startfirst_cmp(t1, t2):
        # When arranging in ascending, lower start but longest within should come first
        if t1[0] < t2[0]:
            return -1
        elif t1[0] > t2[0]:
            return 1
        else:
            if t1[1] < t2[1]:
                return 1
            elif t1[1] > t2[1]:
                return -1
            else:
                return 0

    """ For boolean questions of the kind: XXX E1 yyy E2 zzz zzz zzz

    We want
        Two Qent -> QENT:Tokens@DELIM@START@DELIM@END' style actions
        One Qstr -> QSTR:Tokens@DELIM@START@DELIM@END' action for zz zzz zzz

    Assume that the first two entities are first/second Qent actions, and longest span that starts right after
    second entity is the Qstr action
    """

    sorted_qentspans = sorted(span2qentactions.keys(),
                              key=cmp_to_key(startfirst_cmp),
                              reverse=False)
    sorted_qstrspans = sorted(span2qstractions.keys(),
                              key=cmp_to_key(longestfirst_cmp),
                              reverse=False)
    if len(sorted_qentspans) >= 2:
        gold_qent1span = sorted_qentspans[0]
        gold_qent2span = sorted_qentspans[1]
    else:
        gold_qent1span = sorted_qentspans[0]
        gold_qent2span = sorted_qentspans[0]

    gold_qent2end = gold_qent2span[1]
    gold_qstr_span = None
    for qstrspan in sorted_qstrspans:
        if qstrspan[0] > gold_qent2end:
            gold_qstr_span = qstrspan
            break

    if gold_qstr_span is None:
        gold_qstr_span = sorted_qstrspans[-1]

    qent1_action = span2qentactions[gold_qent1span]
    qent2_action = span2qentactions[gold_qent2span]
    qstr_action = span2qstractions[gold_qstr_span]

    # print(qent1_action)
    # print(qent2_action)
    # print(qstr_action)

    return qent1_action, qent2_action, qstr_action



def getGoldMentionsAndQStr(ques: str, q_mens: List):
    ques_tokens = ques.split(' ')
    span2qentactions = {}
    for qmen in q_mens:
        span2qentactions[(qmen[1], qmen[2] - 1)] = ques_tokens[qmen[1]:qmen[2]]


    qlen = len(ques_tokens)

    bigram_spans: List[Tuple[int, int]] = get_ngram_spans(length=qlen, ngram=2)
    trigam_spans: List[Tuple[int, int]] = get_ngram_spans(length=qlen, ngram=3)
    fourgram_spans: List[Tuple[int, int]] = get_ngram_spans(length=qlen, ngram=4)
    fivegram_spans: List[Tuple[int, int]] = get_ngram_spans(length=qlen, ngram=5)

    span2qstractions = {}
    for spans in [bigram_spans, trigam_spans, fourgram_spans, fivegram_spans]:
        for span in spans:
            span2qstractions[span] = ques_tokens[span[0]:span[1]+1]

    ent1tokens, ent2tokens, qstrtokens = _get_gold_actions(span2qentactions, span2qstractions)
    return ent1tokens, ent2tokens, qstrtokens



def identifyBestContext(entitytokens: List[str], contexts_tokenized: List[List[str]]):
    set_enttokens = set(entitytokens)
    best_context = 0
    best_intersection = 0
    for i, c in enumerate(contexts_tokenized):
        ctokens_set = set(c)
        intersectionsize = len(set_enttokens.intersection(ctokens_set))
        if intersectionsize > best_intersection:
            best_context = i
            best_intersection = intersectionsize

    return best_context, contexts_tokenized[best_context]


def identifyTokenIntersection(tokens: List[str], contexts_tokenized: List[List[str]]):
    set_tokens = set(tokens)
    best_context = 0
    best_intersection = 0
    context_intersections = []

    for i, c in enumerate(contexts_tokenized):
        ctokens_set = set(c)
        intersectionsize = len(set_tokens.intersection(ctokens_set))
        context_intersections.append(intersectionsize)

    return context_intersections


def tokenMatchAcc(json_instances):
    total_correct = 0
    total = len(json_instances)
    for jsonobj in json_instances:

        question = jsonobj[constants.q_field]
        answer = jsonobj[constants.ans_field]
        q_mens = jsonobj[constants.q_ent_ner_field]

        ent1tokens, ent2tokens, qstrtokens = getGoldMentionsAndQStr(question, q_mens)

        # List of title, text tuples
        contexts: List[Tuple[str, str]] = jsonobj[constants.context_field]

        context_texts = [x[1] for x in contexts]
        contexts_tokenized = [x.split(' ') for x in context_texts]

        ent1cidx, ent1_closest_context = identifyBestContext(ent1tokens, contexts_tokenized)
        ent2cidx, ent2_closest_context = identifyBestContext(ent2tokens, contexts_tokenized)

        qstr_context_intersections = identifyTokenIntersection(qstrtokens, contexts_tokenized)

        pred1 = qstr_context_intersections[ent1cidx] > 0
        pred2 = qstr_context_intersections[ent2cidx] > 0
        predbool = pred1 and pred2
        pred = 'yes' if predbool else 'no'

        # print(f"{question} \t ans:{answer}")
        # for c in contexts_tokenized:
        #     print(' '.join(c))
        # print(f"{ent1tokens}: {ent1cidx}")
        # print(f"{ent2tokens}: {ent2cidx}")
        # print(f"{qstrtokens}: {qstr_context_intersections}")
        # print(f"Prediction: {pred}")

        correct = 1 if answer == pred else 0
        total_correct += correct

    print(f"Total Correct: {total_correct}. Total: {total}. Acc: {float(total_correct)/float(total)}")

def main():
    json_docs = util.readJsonlDocs(input_jsonl)

    tokenMatchAcc(json_docs)


if __name__=='__main__':
    main()



