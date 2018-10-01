import os
import time
import json
import argparse
from typing import List, Tuple

import spacy
from spacy.pipeline import SentenceSegmenter
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from utils import TAUtils, util, spacyutils



spacy_nlp = spacyutils.getSpacyNLP()


def tokenizedSent_and_NER(sent: str) -> Tuple[str, List]:
    """ For a given untokenized-sentence, get tokenized sentence and NERs.

    Parameters:
    ----------
    sent: a string

    Returns:
    --------
    tokenized_sentence: a string with whitespace delimition
    ner_tags: List of (text, start, end, label) tuples. end is exclusive
    """
    spacydoc = spacyutils.getSpacyDoc(sent, spacy_nlp)
    tokenized_sentence = spacyutils.getWhiteSpacedSent(spacydoc)
    ner_tags = spacyutils.getNER(spacydoc)

    return (tokenized_sentence, ner_tags)


def tokenizedSent_and_NER_MultipleSents(sents: List[str]) -> List[Tuple[str, List]]:
    """ For a given untokenized-sentence, get tokenized sentence and NERs.

    Parameters:
    ----------
    sent: a string

    Returns:
    --------
    tokenized_sentence: a string with whitespace delimition
    ner_tags: List of (text, start, end, label) tuples. end is exclusive
    """
    spacydocs = spacyutils.getSpacyDocs(sents, spacy_nlp)

    sent_and_tags = []

    for spacydoc in spacydocs:
        tokenized_sentence = spacyutils.getWhiteSpacedSent(spacydoc)
        ner_tags = spacyutils.getNER(spacydoc)
        sent_and_tags.append((tokenized_sentence, ner_tags))

    return sent_and_tags


def tokenizeDocs(input_json: str, output_jsonl: str) -> None:
    """ Tokenize the question, answer and context in the HotPotQA Json. """

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
            new_doc[id_field] = jsonobj[id_field]
            new_doc[suppfacts_field] = jsonobj[suppfacts_field]
            new_doc[qtyte_field] = jsonobj[qtyte_field]
            new_doc[qlevel_field] = jsonobj[qlevel_field]

            # List of paragraphs, each represented as a tuple (title, sentences)
            contexts = jsonobj[context_field]
            question = jsonobj[q_field]
            answer = jsonobj[ans_field]

            (q_tokenized, q_ners) = tokenizedSent_and_NER(question)
            (a_tokenized, a_ners) = tokenizedSent_and_NER(answer)

            new_doc[q_field] = q_tokenized
            new_doc[q_field + "_ner"] = q_ners

            new_doc[ans_field] = a_tokenized
            new_doc[ans_field + "_ner"] = a_ners

            tokenized_contexts = []
            contexts_ners = []
            for para in contexts:
                (title, sentences) = para

                # List of tuples of (sent, ner_tags)
                sent_ner_tuples = tokenizedSent_and_NER_MultipleSents(sentences)

                '''
                # List of tuples of (sent, ner_tags)
                # sent_ner_tuples = [tokenizedSent_and_NER(sent) for sent in sentences]
                '''

                # tokenized_sentences: List of tokenized sentences. ners_per_sent: List of ner_tags per sent
                [tokenized_sentences, ners_per_sent] = list(zip(*sent_ner_tuples))
                assert len(tokenized_sentences) == len(ners_per_sent)

                tokenized_contexts.append((title, tokenized_sentences))
                contexts_ners.append(ners_per_sent)

            new_doc[context_field] = tokenized_contexts
            new_doc[context_field + "_ner"] = contexts_ners

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
    args = parser.parse_args()

    tokenizeDocs(input_json=args.input_json, output_jsonl=args.output_jsonl)