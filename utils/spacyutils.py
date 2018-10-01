import json
import spacy
from typing import List, Tuple
from spacy.tokens import Doc, Span, Token

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def getWhiteTokenizerSpacyNLP():
    nlp = spacy.load('en',  disable=['textcat'])
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    return nlp

def getSpacyNLP():
    nlp = spacy.load('en', disable=['textcat'])
    return nlp


def getSpacyDocs(sents: List[str], nlp):
    """ Batch processing of sentences into Spacy docs."""
    return list(nlp.pipe(sents))

def getSpacyDoc(sent: str, nlp):
    """ Single sent to Spacy doc """
    return nlp(sent)

def getSpanHead(doc: Doc, span: Tuple[int, int]):
    """
    Returns token idx of the span root.
    :param doc: Spacy doc
    :param span_srt: Span start
    :param span_end: Span end (exclusive)
    :return: Token idx of the span head
    """
    assert doc.is_parsed, "Doc isn't dep parsed."
    doclength = len(doc)
    (span_srt, span_end) = span
    assert (span_srt >= 0) and (span_srt < doclength)
    assert (span_end > 0) and (span_end <= doclength)
    span: Span = doc[span_srt:span_end]
    spanroot: Token = span.root
    return spanroot.i


def getNERInToken(doc: Doc, token_idx: int):
    """
    If the given token is a part of NE, return the NE span, otherwise the input token's span
    :param doc: Spacy doc
    :param token_idx: int idx of the token
    :return: (srt-inclusive, end-exclusive)  of the NER (if matches) else (token_idx, token_idx + 1)
    """
    token: Token = doc[token_idx]
    ner_spans = [(ent.start, ent.end) for ent in doc.ents]

    if token.ent_iob_ == 'O':
        # Input token is not a NER
        return (token_idx, token_idx + 1)
    else:
        # Token is an NER, find which span
        # NER spans (srt, end) are in increasing order
        for (srt, end) in ner_spans:
            if token_idx >= srt and token_idx < end:
                return (srt, end)
    print("I SHOULDN'T BE HERE")
    return (token_idx, token_idx + 1)


if __name__=='__main__':
    nlp = getWhiteTokenizerSpacyNLP()
    sent = "In another instance , the Pakistani security official said , Americans in a sport utility vehicle last week fled after police officers tried to search their car at a checkpoint on the outskirts of Islamabad , the capital ."
    doc = nlp(sent)
    print(doc.ents)
    for i, t in enumerate(doc):
        print(f"{i} {t}")

    head = getSpanHead(doc, (34, 38))
    print(head)
    span = getNERInToken(doc, 34)
    print(span)


    # with open('/save/ngupta19/datasets/WDW/pruned_cloze/val_temp.jsonl', 'r') as inpf:
    #     for line in inpf:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         jsonobj = json.loads(line)
    #         sentences: List[List[str]] = jsonobj['context']
    #         sentences: List[str] = [' '.join(sent) for sent in sentences]
    #
    #         sents = getSpacyDocs(sentences, nlp)
    #         for sent in sents:
    #             span_srt = 0
    #             span_end = min(5, len(sent))
    #             span = (span_srt, span_end)
    #             # print(sent[span[0]:span[1]])
    #             spanhead = getSpanHead(sent, span)
    #             nerspan = getNERInToken(sent, spanhead)

