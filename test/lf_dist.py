from typing import List, Dict, Union
import random

def isSpanOverlap(s1, s2):
    """ Returns True if the spans overlap. Works with exclusive end spans

    s1, s2 : Tuples containing span start and end (exclusive)
    srt_idx, end_idx: Idxs of span_srt and span_end in the input tuples
    """
    start1, end1 = s1[0], s1[1]
    start2, end2 = s2[0], s2[1]

    return max(start1, start2) <= min(end1, end2)

def sample_spans(seqlen, span_lengths: List[int]):
    sum_lengths = sum(span_lengths)
    num_heads = seqlen - (sum_lengths + len(span_lengths) - 1)

    if num_heads < 0:
        return None

    res = set()
    for _, spanlen in enumerate(span_lengths):
        s = random.randint(0, seqlen - spanlen)
        e = s + spanlen - 1
        while any(isSpanOverlap((s,e), span) for span in res):
            s = random.randint(0, seqlen - spanlen)
            e = s + spanlen - 1
        res.add((s, e))

    return res

res = sample_spans(10, span_lengths=[1, 3, 4])

print(res)


