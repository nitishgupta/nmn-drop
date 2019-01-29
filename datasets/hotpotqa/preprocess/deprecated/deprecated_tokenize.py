import os
import sys
import time
import json
import argparse
from typing import List, Tuple

from utils import TAUtils, util, spacyutils
from datasets.hotpotqa.utils import constants

spacy_nlp = spacyutils.getSpacyNLP()
# ccg_nlp = TAUtils.getCCGNLPLocalPipeline()


def removeOverlappingSpans(spans, srt_idx, end_idx):
    """
    Remove overlapping spans by keeping the longer one. Span ends are exclusive

    Parameters:
    ----------
    spans: List of (..., start, ..., end, ...) tuples
    srt_idx: Index of span_srt in the span tuple
    end_idx: Index of span_end in the span tuple

    Returns:
    --------
    final_spans: List of tuples of spans that don't overlap
    """

    if len(spans) == 0:
        return spans
    # Spans sorted by increasing order
    sorted_spans = sorted(spans, key=lambda x: x[srt_idx])

    final_spans = [sorted_spans[0]]
    for i in range(1, len(sorted_spans)):
        last_span = final_spans[-1]
        span = sorted_spans[i]
        if not util.isSpanOverlap(last_span, span, srt_idx, end_idx):
            final_spans.append(span)
        else:
            len1 = last_span[end_idx] - last_span[srt_idx]
            len2 = span[end_idx] - span[srt_idx]
            # If incoming span is longer, delete last span and put
            if len2 > len1:
                final_spans.pop()
                final_spans.append(span)

    return final_spans


def tokenizeTitles(titles: List[str]):
    ''' Tokenize list of Wikipedia titles using spacy. '''
    spacydocs = spacyutils.getSpacyDocs(titles, spacy_nlp)
    tokenized_titles = []
    for titledoc in spacydocs:
        tokenized_titles.append(spacyutils.getTokens(titledoc))

    return tokenized_titles


def matchingWikiTitlesInSent(spacydoc: spacyutils.Doc, titles: List[List[str]]):
    """
    Match all the title spans in the given spacyDoc of a single sentence

    Parameters:
    -----------
    spacydoc: Input spacydoc
    titles: List of tokenized titles

    Returns:
    --------
    ners: List of (text, start, end, type) of the matched title spans
    """
    tokens = spacyutils.getTokens(spacydoc)
    ners = []

    # List of (start, end) spans
    matching_spans = []
    for t in titles:
        matching_subspans = util.getMatchingSubSpans(tokens, t)
        matching_spans.extend(matching_subspans)

    # Remove overlapping matching spans by keeping the longer ones.
    matching_spans = removeOverlappingSpans(matching_spans, 0, 1)
    for (srt, end) in matching_spans:
        ners.append((spacydoc[srt:end].text, srt, end, "TITLE"))

    return ners


def getTitleNames(input_titles):
    """ Clean titles by keeping everything before the first instance of "(". If result is empty, remove it.
    Length of output list can be different from input.

    Parameters:
    -----------
    input_titles: List of wikipedia title strings

    Returns:
    --------
    new_titles: List of titles with string upto first (. If the result is empty, remove it
    """

    new_titles = []
    for t in input_titles:
        idx = t.find("(")
        if idx is not -1:
            t = t[0:idx]
        if len(t) != 0:
            new_titles.append(t)

    return new_titles


def titleMatchAndNER(spacydoc: spacyutils.Doc, tokenized_titles: List[List[str]], mark_propn: bool):
    ''' Get all Entity spans in the sentence, Titles, NER, and PropNoun spans.

    Parameters:
    -----------
    spacydoc: SpacyDoc for a single sentence
    tokenized_titles: List of tokenized titles for titles in the contexts
    mark_propn: Boolean indicating whether spans of PropNoun should be marked or not

    Returns:
    --------
    tokenized_sent: Space-delimited input sentence
    finalEntitySpans: List of (text, start, end, label)-tuples for all entities
    '''

    tokens = spacyutils.getTokens(spacydoc)
    tokenized_sent = ' '.join(tokens)

    finalEntitySpans = []

    titleMatchingSpans = matchingWikiTitlesInSent(spacydoc, tokenized_titles)
    # Remove overlaps if any, keep the first argument's span in case of overlap
    finalEntitySpans = util.mergeSpansAndRemoveOverlap(finalEntitySpans, titleMatchingSpans, 1, 2)

    spacy_ners = spacyutils.getNER(spacydoc)
    # Remove overlaps if any, keep the first argument's spans (titles) in case of overlap
    finalEntitySpans = util.mergeSpansAndRemoveOverlap(finalEntitySpans, spacy_ners, 1, 2)

    if mark_propn:
        spacy_propns = spacyutils.getPropnSpans(spacydoc)
        # Remove overlaps if any, keep the first argument's spans (titles + ner) in case of overlap
        finalEntitySpans = util.mergeSpansAndRemoveOverlap(finalEntitySpans, spacy_propns, 1, 2)

    # If using CCG NER
    # ccg_ners = TAUtils.getOntonotesNER(tokens, ccg_nlp)
    # titleMatch_and_NERs = util.mergeSpansAndRemoveOverlap(titleMatch_and_NERs, ccg_ners, 1, 2)

    return tokenized_sent, finalEntitySpans


def tokenizeDocs(input_json: str, output_jsonl: str, mark_propn: bool) -> None:
    """ Tokenize the question, answer and context in the HotPotQA Json.

    Returns:
    --------
    Jsonl file with same datatypes as input with the modification/addition of:
    Modifications:
        q_field: The question is tokenized
        ans_field: The answer is now modified
        context_field: Context sentences are now tokenized, but stored with white-space delimition

    Additions:
        q_ner_field: NER tags for question. Each NER tag is (spantext, start, end, label) with exclusive-end.
        ans_ner_field: NER tags in answer
        context_ner_field: NER tags in each of the context sentences
    """

    print("Reading input jsonl: {}".format(input_json))
    print("Output filepath: {}".format(output_jsonl))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, 'r') as f:
        jsonobjs = json.load(f)

    print("Number of docs: {}".format(len(jsonobjs)))
    numdocswritten = 0

    stime = time.time()

    with open(output_jsonl, 'w') as outf:
        for jsonobj in jsonobjs:
            new_doc = {}
            new_doc[constants.id_field] = jsonobj[constants.id_field]
            new_doc[constants.suppfacts_field] = jsonobj[constants.suppfacts_field]
            new_doc[constants.qtyte_field] = jsonobj[constants.qtyte_field]
            new_doc[constants.qlevel_field] = jsonobj[constants.qlevel_field]

            # List of paragraphs, each represented as a tuple (title, sentences)
            contexts = jsonobj[constants.context_field]
            titles = [t for (t, _) in contexts]
            question: str = jsonobj[constants.q_field]
            answer: str = jsonobj[constants.ans_field]
            titles: List[str] = getTitleNames(titles)
            tokenized_titles: List[List[str]] = tokenizeTitles(titles)

            # Remove trailing and multiple spaces
            question = util.pruneMultipleSpaces(question)
            q_spacydoc = spacyutils.getSpacyDoc(question, spacy_nlp)
            q_tokenized, q_ners = titleMatchAndNER(q_spacydoc, tokenized_titles, mark_propn)

            new_doc[constants.q_field] = q_tokenized
            new_doc[constants.q_ner_field] = q_ners

            answer = answer.strip()
            a_spacydoc = spacyutils.getSpacyDoc(answer, spacy_nlp)
            a_tokenized, a_ners = titleMatchAndNER(a_spacydoc, tokenized_titles, mark_propn)

            new_doc[constants.ans_field] = answer
            new_doc[constants.ans_tokenized_field] = a_tokenized
            new_doc[constants.ans_ner_field] = a_ners

            tokenized_contexts = []
            contexts_ners = []
            for para in contexts:
                (title, sentences) = para
                sentences = [util.pruneMultipleSpaces(sent) for sent in sentences]
                # List of tuples of (sent, ner_tags)
                sentences_spacydocs = spacyutils.getSpacyDocs(sentences, spacy_nlp)
                # List of (sentence, ners)
                sent_ner_tuples = [titleMatchAndNER(spacydoc, tokenized_titles, mark_propn) \
                                   for spacydoc in sentences_spacydocs]
                # tokenized_sentences: List of tokenized sentences. ners_per_sent: List of ner_tags per sent
                [tokenized_sentences, ners_per_sent] = list(zip(*sent_ner_tuples))
                assert len(tokenized_sentences) == len(ners_per_sent)

                tokenized_contexts.append((title, tokenized_sentences))
                contexts_ners.append(ners_per_sent)

            new_doc[constants.context_field] = tokenized_contexts
            new_doc[constants.context_ner_field] = contexts_ners

            outf.write(json.dumps(new_doc))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 100 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_jsonl', default=True)
    parser.add_argument('--propn', action='store_true', default=False)
    args = parser.parse_args()

    print('Using propn: {}'.format(args.propn))

    # args.input_json --- is the raw json from the HotpotQA dataset
    tokenizeDocs(input_json=args.input_json, output_jsonl=args.output_jsonl,
                 mark_propn=args.propn)