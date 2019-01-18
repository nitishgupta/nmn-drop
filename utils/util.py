import os
import sys
import json
import unicodedata
from typing import Dict, List, Any, Tuple

stopwords = None


def countList(input_list: List[Any], depth: int=1) -> int:
    """
    Count the number of elements in a nested list
    :param input_list: nested list
    :param depth: depth of the list. depth = 1 indicates vanilla list
    :return: int
    """
    num_elems = 0
    if depth == 1:
        return len(input_list)

    else:
        return sum([countList(l, depth - 1) for l in input_list])


def pruneMultipleSpaces(sentence: str):
    """ Prune multiple spaces in a sentence and replace with single space
    Parameters:
    -----------
    sentence: Sentence string with mulitple spaces

    Returns:
    --------
    cleaned_sentence: String with only single spaces.
    """

    sentence = sentence.strip()
    tokens = sentence.split(' ')
    tokens = [t for t in tokens if t !=  '']
    if len(tokens) == 1:
        return tokens[0]
    else:
        return ' '.join(tokens)


def stopWords():
    """Returns a set of stopwords."""

    global stopwords
    if stopwords is None:
        f = open("utils/stopwords.txt", 'r')
        lines = f.readlines()
        words = [w.strip() for w in lines if w.strip()]
        stopwords = set(words)
    return stopwords


def readlines(fp):
    ''' Read all lines from a filepath. '''
    with open(fp, 'r') as f:
        text = f.read().strip()
        lines = text.split('\n')
    return lines


def readJsonlDocs(jsonlfp: str) -> List[Dict]:
    ''' Read all docs from jsonl file. '''
    lines = readlines(jsonlfp)
    docs = [json.loads(line) for line in lines]
    return docs


def isSpanOverlap(s1, s2, srt_idx=0, end_idx=1):
    """ Returns True if the spans overlap. Works with exclusive end spans

    s1, s2 : Tuples containing span start and end (exclusive)
    srt_idx, end_idx: Idxs of span_srt and span_end in the input tuples
    """
    start1, end1 = s1[srt_idx], s1[end_idx]
    start2, end2 = s2[srt_idx], s2[end_idx]
    return max(start1, start2) <= (min(end1, end2) - 1)


def doSpansIntersect(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    span1 = set(range(span1[0], span1[1]))
    span2 = set(range(span2[0], span2[1]))

    if len(span1.intersection(span2)) > 0:
        return True
    else:
        return False


def getContiguousSpansOfElement(l: List[Any], element: Any) -> List[Tuple[int, int]]:
    """ Get contiguous spans of element in list

    (start, end) -- start is inclusie, end exclusive

    Returns:
        List of (start, end) tuples
    """

    start = -1
    end = -1
    spans = []

    for i, e in enumerate(l):
        if e == element:
            if start == -1:
                start = i
                end = i + 1
            else:
                end = i + 1
        else:
            if start != -1:
                spans.append((start, end))
                start = -1

    if start != -1:
        spans.append((start, end))

    return spans


def mostFreqKeysInDict(d: Dict) -> List[Any]:
    """ d is a dict with values as frequencies """
    sortedDict = sortDictByValue(d, True)
    maxVal = sortedDict[0][1]
    mostFreqElems = []
    for elem, frq in sortedDict:
        if frq == maxVal:
            mostFreqElems.append(elem)
        else:
            break

    return mostFreqElems


def sortDictByValue(d, decreasing=False):
    return sorted(d.items(), key=lambda x: x[1], reverse=decreasing)


def sortDictByKey(d, decreasing=False):
    return sorted(d.items(), key=lambda x: x[0], reverse=decreasing)


def getContextAroundSpan(seq: List[Any], span: Tuple[int, int], context_len: int) -> Tuple[List[Any], List[Any]]:
    """ For a given seq of items and a span in, Return the left and right context

    Args:
        seq: input list of items
        span: Start/End positions of span (inclusive/exclusive)
        context_len: length of context on each side

    Returns:
        (left_context, right_context): list of context items. Max length of both is context_len
    """
    assert context_len > 0, f"Context length cannot be <= 0:  {context_len}"
    assert span[0] != span[1], f"Span length cannot be zero: {span}"
    assert span[0] >= 0, f"Span start invalid: {span}"
    assert span[1] <= len(seq), f"Span end invalid: {span}"

    left_start_index = max(0, span[0] - context_len)
    right_end_index = min(len(seq), span[1] + context_len)

    left_context = seq[left_start_index:span[0]]
    right_context = seq[span[1]:right_end_index]

    return (left_context, right_context)


def getMatchingSubSpans(seq: List[Any], pattern: List[Any]) -> List[Tuple[int, int]]:
    """ Returns the (start, end) spans of the pattern in the text list

    Like always, end is exclusive

    text: [5,1,5,1,1,2,1,1,2,1,2,1]
    pattern: [2,1]
    Return: [(5,7), (8, 10), (10, 12)]
    """

    startPositions = list(_KnuthMorrisPratt(seq, pattern))
    endPositions = [spos + len(pattern) for spos in startPositions]
    matchingSpans = [(s, e) for s, e in zip(startPositions, endPositions)]

    return matchingSpans


def _KnuthMorrisPratt(text, pattern):
    # Knuth-Morris-Pratt string matching
    # David Eppstein, UC Irvine, 1 Mar 2002

    # from http://code.activestate.com/recipes/117214/
    '''Yields all starting positions of copies of the pattern in the text.
    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


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

def round_all(stuff, prec):
    """ Round all the number elems in nested stuff. """
    if isinstance(stuff, list):
        return [round_all(x, prec) for x in stuff]
    if isinstance(stuff, tuple):
        return tuple(round_all(x, prec) for x in stuff)
    if isinstance(stuff, float):
        return round(float(stuff), prec)
    if isinstance(stuff, dict):
        d = {}
        for k, v in stuff.items():
            d[k] = round(v, prec)
        return d
    else:
        return stuff


def removeOverlappingSpans(spans):
    """
    Remove overlapping spans by keeping the longest ones. Span end are exclusive

    spans: List of (start, end) tuples
    """
    def spanOverlap(s1, s2):
        """ Works with exclusive end spans """
        start1, end1 = s1
        start2, end2 = s2
        return max(start1, start2) <= (min(end1, end2) - 1)

    if len(spans) == 0:
        return spans
    # Spans sorted by increasing order
    sorted_spans = sorted(spans, key=lambda x: x[0])

    final_spans = [sorted_spans[0]]
    for i in range(1, len(sorted_spans)):
        last_span = final_spans[-1]
        span = sorted_spans[i]
        if not spanOverlap(last_span, span):
            final_spans.append(span)
        else:
            len1 = last_span[1] - last_span[0]
            len2 = span[1] - span[0]
            # If incoming span is longer, delete last span and put
            if len2 > len1:
                final_spans.pop()
                final_spans.append(span)

    return final_spans


def mergeSpansAndRemoveOverlap(orig_spans, new_spans, srt_idx, end_idx):
    ''' Merge a list of spans in another given list resulting in non-overlapping spans.
    Assumes that both span lists are independently non-overlapping.

    While merging, if incoming span overlaps, keep the original.

    Parameters:
    -----------
    orig_spans: List of (..., start, ..., end, ...) tuples. Original spans
    new_spans: New spans. Same as above
    srt_idx: Index of span_srt in the span tuple
    end_idx: Index of span_end in the span tuple

    Returns:
    --------
    final_spans: List of merged spans sorted by span start.

    '''

    if len(orig_spans) == 0:
        return new_spans
    if len(new_spans) == 0:
        return orig_spans

    # Spans sorted by increasing order
    sorted_orig_spans = sorted(orig_spans, key=lambda x: x[srt_idx])
    sorted_new_spans = sorted(new_spans, key=lambda x: x[srt_idx])

    # These will act as the head pointers in the two lists
    orig_span_idx = 0
    new_span_idx = 0

    spans_to_add = []

    while True:
        # No new spans to merge
        if new_span_idx == len(sorted_new_spans):
            break

        # Original List is done
        if orig_span_idx == len(sorted_orig_spans):
            spans_to_add.append(sorted_new_spans[new_span_idx])
            new_span_idx += 1
        else:
            orig_span = sorted_orig_spans[orig_span_idx]
            new_span = sorted_new_spans[new_span_idx]

            # Spans overlap, move head to the next new span
            if isSpanOverlap(orig_span, new_span, srt_idx, end_idx):
                new_span_idx += 1
            else:
                # If new span starts after the current original span's end, then move the current original span head
                if new_span[srt_idx] >= orig_span[end_idx]:
                    orig_span_idx += 1
                    # continue

                # New span ends before the current head start
                # Previous condition ensures that the new_span_start is after the previous orig_span_end
                # Hence this span doesn't overlap with original spans ==> merge
                if new_span[end_idx] <= orig_span[srt_idx]:
                    spans_to_add.append(new_span)
                    new_span_idx += 1
                    # continue

    sorted_orig_spans.extend(spans_to_add)
    sorted_orig_spans = sorted(sorted_orig_spans, key=lambda x: x[srt_idx])

    return sorted_orig_spans


def cleanMentionSurface(arg):
    return _getLnrm(arg)


def _getLnrm(arg):
    """Normalizes the given arg by stripping it of diacritics, lowercasing, and
    removing all non-alphanumeric characters.
    """
    arg = ''.join(
        [c for c in unicodedata.normalize('NFD', arg) if unicodedata.category(c) != 'Mn'])
    arg = arg.lower()
    arg = ''.join(
        [c for c in arg if c in set('abcdefghijklmnopqrstuvwxyz0123456789')])
    return arg



if __name__=='__main__':
    words = stopWords()

    l = [5,1,5,1,5,2,1,1,2,1,2,1]

    pattern = [5,1,5,0,1]

    input_list = [[[1,2,3], [3,4,5]], [[1,2,3], [3,4,5], [9,10,10,101]], [[1,2,3,3,4,5]]]


    spans = [[2,4], [8,10], [3,6], [0, 1]]
    spans = removeOverlappingSpans(spans)
    new_spans = [[1,2], [6,7]]
    print(spans)
    print(new_spans)
    print(mergeSpansAndRemoveOverlap(spans, new_spans, 0, 1))

    print(cleanMentionSurface("Daniel José Older"))

    a = {'1': 0.1288361823, '2':9.1298836128736}
    print(round_all(a, 3))


