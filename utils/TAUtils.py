from ccg_nlpy import TextAnnotation
from typing import List, Tuple
from ccg_nlpy.local_pipeline import LocalPipeline


def getCCGNLPLocalPipeline():
    ccg_nlp = LocalPipeline()
    return ccg_nlp


def getOntonotesNER(sentence: List[str], ccg_nlp: LocalPipeline):
    try:
        ccgdoc: TextAnnotation = ccg_nlp.doc([sentence], pretokenized=True)
    except:
        return []

    ner_view = ccgdoc.get_ner_ontonotes

    ners = []

    if ner_view.cons_list is not None:
        for cons in ner_view.cons_list:
            ners.append((cons['tokens'], cons['start'], cons['end'], cons['label'] + '_ccg'))

    return ners


def thresholdSentLength(ta: TextAnnotation, lp: LocalPipeline,
                        maxlen: int = 120) -> TextAnnotation:
    """ Make a new text annotation by spliting sentences longer than maxlen tokens 
    :rtype: TextAnnotation
    """

    sentences: List[List[str]] = get_sentences(ta)

    new_sentences = []
    for sent in sentences:
        if len(sent) <= maxlen:
            new_sentences.append(sent)
        else:
            # Make chunks of maxlen
            while len(sent) > maxlen:
                new_sentences.append(sent[0:maxlen])
                sent = sent[maxlen:]
            # Put the last remaining sent i.e. < maxlen
            new_sentences.append(sent)

    new_ta = lp.doc(new_sentences, True)

    return new_ta


def getNPChunks_perSent(ta: TextAnnotation) -> List[List[Tuple[int, int]]]:
    """TA -> For each sentence, within sentence token offsets for NPs."""

    token_sentIdxWithinSentIdx = getAll_SentIdAndTokenOffset(ta)
    numSents = len(ta.sentence_end_position)

    nps_withinSentIdxs = [[] for i in range(0, numSents)]

    chunk_cons = ta.get_shallow_parse.cons_list
    for cons in chunk_cons:
        if cons['label'] == 'NP':
            (sentidx1, startOffset) = token_sentIdxWithinSentIdx[cons['start']]
            (sentidx2, endOffset) = token_sentIdxWithinSentIdx[cons['end'] - 1]
            if sentidx1 != sentidx2:
                # Ignore cross sentence NPs
                continue
            nps_withinSentIdxs[sentidx1].append((startOffset, endOffset + 1))

    return nps_withinSentIdxs


def getPOS_forDoc(ta: TextAnnotation) -> List[str]:
    """TA -> For complete doc get POS."""

    pos_cons = ta.get_pos.cons_list

    pos_tags = [cons['label'] for cons in pos_cons]

    assert len(ta.tokens) == len(pos_tags)

    return pos_tags



def getPOS_perSent(ta: TextAnnotation) -> List[List[str]]:
    """TA -> For each sentence, within sentence token offsets for NPs."""

    token_sentIdxWithinSentIdx = getAll_SentIdAndTokenOffset(ta)
    numSents = len(ta.sentence_end_position)

    pos_perSent = [[] for i in range(0, numSents)]

    pos_cons = ta.get_pos.cons_list

    assert len(ta.tokens) == len(pos_cons)

    for idx in range(0, len(ta.tokens)):
        pos_label = pos_cons[idx]['label']
        (sentidx1, _) = token_sentIdxWithinSentIdx[idx]
        pos_perSent[sentidx1].append(pos_label)
    return pos_perSent


def getNPsWithGlobalOffsets(ta: TextAnnotation) -> List[Tuple[int, int]]:
    """TA -> Token offsets for all NPs (with global offsets)."""
    chunk_cons = ta.get_shallow_parse.cons_list
    np_chunk_cons = []
    for cons in chunk_cons:
        if cons['label'] == 'NP':
            np_chunk_cons.append(cons)

    np_chunk_startend = [(cons['start'], cons['end']) for cons in np_chunk_cons]

    return np_chunk_startend


def get_sentences(ta: TextAnnotation) -> List[List[str]]:
    """Get tokenized sentences from TextAnnotation as list of list of str."""

    start = 0
    sentences = []
    tokens = ta.tokens
    sentence_end_positions = ta.sentence_end_position
    for end in sentence_end_positions:
        sentences.append(tokens[start:end])
        start = end
    assert len(sentences) == len(sentence_end_positions)
    return sentences


def getAll_SentIdAndTokenOffset(ta: TextAnnotation) -> List[Tuple[int, int]]:
    """Get (sentence idx, withinSentOffset) for all tokens."""
    tokens = ta.tokens
    numTokens = len(tokens)
    tokenIdxs = []

    sentence_end_pos = ta.sentence_end_position
    # sentence_start_pos = [0]
    # sentence_start_pos.extend(sentence_end_pos)
    # sentence_start_pos = sentence_start_pos[:-1]

    sent_idx = 0
    withinsent_tokenidx = 0
    for i in range(0, numTokens):
        if i == sentence_end_pos[sent_idx]:
            sent_idx += 1
            withinsent_tokenidx = 0

        tokenIdxs.append((sent_idx, withinsent_tokenidx))
        withinsent_tokenidx += 1

    return tokenIdxs


def getSentIdAndTokenOffset(ta: TextAnnotation) -> List[Tuple[int, int]]:
    tokenIdxs = []
    sentence_end_pos = ta.sentence_end_position
    for i in range(0, len(ta.tokens)):
        tokenIdxs.append(getSentIdx_WithinSentTokenIdx(sentence_end_pos, i))

    return tokenIdxs

def getSentEndPosArray(sentences: List[List[str]]) -> List[int]:
    """
    Get sentences end position array for tokenized sentences
    :param sentences: Tokenized sentences
    :return: sentence_end_pos: List containing (ex) tokenoffsets of sentence ends. Length == number of sentences
    """
    sum = 0
    sentence_end_pos = []
    for i, sent in enumerate(sentences):
        sum += len(sent)
        sentence_end_pos.append(sum)
    return sentence_end_pos


def tokenizedSentToStr(sentences: List[List[str]]) -> str:
    docstr = ""
    for sent in sentences:
        docstr += ' '.join(sent)
        docstr += "\n"
    return docstr.strip()


def getSentIdx_WithinSentTokenIdx(sentence_end_pos: List[int], tokenidx: int) -> Tuple[int, int]:
    """ Get sent idx and within sent idx from global token offset
    :param sentence_end_pos: List containing (ex) global tokenoffsets of sentence ends. Length == number of sentences
    :param tokenidx: Global token offset
    :return: (sent_id, within_sent_tokenidx)
    """
    sentidx = get_closest_value(sentence_end_pos, tokenidx)
    if sentidx == 0:
        return (sentidx, tokenidx)
    else:
        withinsentidx = tokenidx - sentence_end_pos[sentidx - 1]

        return (sentidx, withinsentidx)


def get_closest_value(arr, target):
    arr = [(i - 1) for i in arr]
    n = len(arr)
    left = 0
    right = n - 1
    mid = 0

    if target > arr[-1] or target < 0:
        print("Token Offset not in range: target:{} max:{} ".format(
            target, arr[-1]))
        raise Exception

    # edge case - last or above all
    if target == arr[-1]:
        return n - 1
    # edge case - first or below all
    if target <= arr[0]:
        return 0
    # BSearch solution: Time & Space: Log(N)
    while left < right:
        mid = (left + right) // 2  # find the mid

        if target < arr[mid]:
            right = mid
        elif target > arr[mid]:
            left = mid + 1
        else:
            return mid

    if arr[mid] < target:
        return mid + 1
    else:
        return mid


if __name__=='__main__':
    a = [32, 79, 127, 173, 196, 225, 248, 266, 287, 314, 325, 349, 395, 426, 457, 470, 492, 525, 537, 560, 572, 620, 649, 670, 693, 716]
    print(a)
    print(len(a))
    print("\n")
    print(a[get_closest_value(a, 31)])
    print(a[get_closest_value(a, 32)])
    print(a[get_closest_value(a, 0)])
    print(a[get_closest_value(a, 715)])
    print(a[get_closest_value(a, 198)])
    print(a[get_closest_value(a, 356)])
    print(a[get_closest_value(a, 542)])
    print(a[get_closest_value(a, 675)])

    print("\n")
    print(getSentIdx_WithinSentTokenIdx(a, 21))
    print(getSentIdx_WithinSentTokenIdx(a, 31))
    print(getSentIdx_WithinSentTokenIdx(a, 32))
    print(getSentIdx_WithinSentTokenIdx(a, 560))
    print(getSentIdx_WithinSentTokenIdx(a, 564))

