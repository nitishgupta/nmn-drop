import os
import sys
import json
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


def readJsonlDocs(jsonlfp: str):
    ''' Read all docs from jsonl file. '''
    lines = readlines(jsonlfp)
    docs = [json.loads(line) for line in lines]
    return docs


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
    sortedDict = sortDict(d, True)
    maxVal = sortedDict[0][1]
    mostFreqElems = []
    for elem, frq in sortedDict:
        if frq == maxVal:
            mostFreqElems.append(elem)
        else:
            break

    return mostFreqElems


def sortDict(d, decreasing=False):
    return sorted(d.items(), key=lambda x: x[1], reverse=decreasing)


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


if __name__=='__main__':
    words = stopWords()

    l = [5,1,5,1,5,2,1,1,2,1,2,1]

    pattern = [5,1,5,0,1]

    input_list = [[[1,2,3], [3,4,5]], [[1,2,3], [3,4,5], [9,10,10,101]], [[1,2,3,3,4,5]]]
    print(countList(input_list, 3))



