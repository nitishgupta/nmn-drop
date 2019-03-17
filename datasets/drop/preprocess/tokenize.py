import os
import sys
import copy
import time
import json
import string
import unicodedata
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from utils import util, spacyutils
from datasets.drop import constants
from datasets.drop.preprocess import ner_process

import multiprocessing

IGNORED_TOKENS = {'a', 'an', 'the'}
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])

spacy_nlp = spacyutils.getSpacyNLP()

ANSWER_TYPE_NOT_FOUND = 0

def split_on_hyphens(text: str):
    """ Adds spaces around hyphens in text. """
    text = util.pruneMultipleSpaces(" - ".join(text.split("-")))
    text = util.pruneMultipleSpaces(" – ".join(text.split("–")))
    text = util.pruneMultipleSpaces(" ~ ".join(text.split("~")))
    return text


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

    chunk_size = n
    return [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]


def _check_validity_of_spans(spans: List[Tuple[int, int]], len_seq: int):
    for span in spans:
        assert span[0] >= 0
        assert span[1] < len_seq

def find_valid_spans(passage_tokens: List[str],
                     answer_texts: List[str]) -> List[Tuple[int, int]]:

    # debug = False
    # if 'T. J. Houshmandzadeh' in answer_texts:
    #     debug = True
    normalized_tokens = [token.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
    # if debug:
    #     print('\n')
    #     print(normalized_tokens)
    #     print()

    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        # answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        answer_text_tokens = answer_text.split()
        answer_tokens = [token.lower().strip(STRIPPED_CHARACTERS) for token in answer_text_tokens]
        # if debug:
        #     print(answer_tokens)

        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue


        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans

def convert_answer(answer_annotation: Dict[str, Union[str, Dict, List]]) -> Tuple[str, List]:
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts = []
    if answer_type is None:  # No answer
        return None
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [answer_content[key]
                       for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def processPassage(input_args):
    """ Helper function for multiprocessing. See tokenizeDocs for details. """

    passage_id, passage_info = input_args
    new_passage_info = {}

    original_passage_text: str = passage_info[constants.passage].strip()
    original_passage_text = unicodedata.normalize("NFKD", original_passage_text)
    passage_text = split_on_hyphens(original_passage_text)

    passage_spacydoc = spacyutils.getSpacyDoc(passage_text, spacy_nlp)
    passage_tokens = [token.text for token in passage_spacydoc]
    passage_token_charidxs = [token.idx for token in passage_spacydoc]
    new_passage_info[constants.passage] = ' '.join(passage_tokens)
    new_passage_info[constants.original_passage] = original_passage_text
    new_passage_info[constants.passage_charidxs] = passage_token_charidxs

    passage_ners = spacyutils.getNER(passage_spacydoc)

    (parsed_dates, normalized_date_idxs,
     normalized_date_values, num_date_entities) = ner_process.parseDateNERS(passage_ners)
    (parsed_nums, normalized_num_idxs,
     normalized_number_values, num_num_entities) = ner_process.parseNumNERS(passage_ners, passage_tokens)

    new_passage_info[constants.passage_date_mens] = parsed_dates
    new_passage_info[constants.passage_date_entidx] = normalized_date_idxs
    new_passage_info[constants.passage_date_normalized_values] = normalized_date_values

    new_passage_info[constants.passage_num_mens] = parsed_nums
    new_passage_info[constants.passage_num_entidx] = normalized_num_idxs
    new_passage_info[constants.passage_num_normalized_values] = normalized_number_values

    # Maybe add whitespace info later

    qa_pairs: List[Dict] = passage_info[constants.qa_pairs]
    new_qa_pairs = []
    for qa in qa_pairs:
        new_qa = {}
        query_id = qa[constants.query_id]
        new_qa[constants.query_id] = query_id
        original_question: str = qa[constants.question].strip()
        original_question = unicodedata.normalize("NFKD", original_question)
        question = split_on_hyphens(original_question)

        q_spacydoc = spacyutils.getSpacyDoc(question, spacy_nlp)
        question_tokens: List[str] = [token.text for token in q_spacydoc]
        question_token_charidxs = [token.idx for token in q_spacydoc]
        new_qa[constants.question] = ' '.join(question_tokens)
        new_qa[constants.original_question] = original_question
        new_qa[constants.question_charidxs] = question_token_charidxs

        q_ners = spacyutils.getNER(q_spacydoc)
        (parsed_dates, normalized_date_idxs,
         normalized_date_values, num_date_entities) = ner_process.parseDateNERS(q_ners)
        (parsed_nums, normalized_num_idxs,
         normalized_number_values, num_num_entities) = ner_process.parseNumNERS(q_ners, question_tokens)

        new_qa[constants.q_date_mens] = parsed_dates
        new_qa[constants.q_date_entidx] = normalized_date_idxs
        new_qa[constants.q_date_normalized_values] = normalized_date_values

        new_qa[constants.q_num_mens] = parsed_nums
        new_qa[constants.q_date_mens] = normalized_num_idxs
        new_qa[constants.q_num_normalized_values] = normalized_number_values

        answer = qa[constants.answer]
        new_qa[constants.answer] = answer

        answer_content = convert_answer(answer)
        if answer_content is None:
            # Some qa don't have any answer annotated
            print(f"Couldn't resolve answer: {answer}")
            continue

        answer_type, answer_texts = answer_content

        # answer_texts_for_evaluation = [' '.join(answer_texts)]
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_text = unicodedata.normalize("NFKD", answer_text)
            answer_text = split_on_hyphens(answer_text)
            answer_spacydoc = spacyutils.getSpacyDoc(answer_text, spacy_nlp)
            answer_tokens = spacyutils.getTokens(answer_spacydoc)
            tokenized_answer_texts.append(' '.join(answer_tokens))

        valid_passage_spans = \
            find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        valid_question_spans = \
            find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []

        _check_validity_of_spans(valid_passage_spans, len(passage_tokens))
        _check_validity_of_spans(valid_question_spans, len(question_tokens))

        new_qa[constants.answer_passage_spans] = valid_passage_spans
        new_qa[constants.answer_question_spans] = valid_question_spans

        if answer_type == "spans":
            if valid_passage_spans or valid_question_spans:
                new_qa[constants.answer_type] = constants.SPAN_TYPE
            else:
                print(f"Answer span not found in passage. passageid: {passage_id}. queryid: {query_id}")
                print(f"Answer Texts: {answer_texts}")
                print(f"Tokenized_Answer Texts: {tokenized_answer_texts}")
                print(new_passage_info[constants.passage])
                print()
                continue
        elif answer_type == "number":
            new_qa[constants.answer_type] = constants.NUM_TYPE
        elif answer_type == "date":
            new_qa[constants.answer_type] = constants.DATE_TYPE

        new_qa_pairs.append(new_qa)
    new_passage_info[constants.qa_pairs] = new_qa_pairs

    return {passage_id: new_passage_info}


def tokenizeDocs(input_json: str, output_json: str, nump: int) -> None:
    """ Tokenize the question, answer and context in the HotPotQA Json.

    Returns:
    --------
    Jsonl file with same datatypes as input with the modification/addition of:
    Modifications:
        q_field: The question is tokenized
        context_field: Context sentences are now tokenized, but stored with white-space delimition

    Additions:
        ans_tokenized_field: tokenized answer if needed
        q_ner_field: NER tags for question. Each NER tag is (spantext, start, end, label) with exclusive-end.
        ans_ner_field: NER tags in answer
        context_ner_field: NER tags in each of the context sentences
    """

    print("Reading input jsonl: {}".format(input_json))
    print("Output filepath: {}".format(output_json))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, 'r') as f:
        dataset = json.load(f)

    print("Number of docs: {}".format(len(dataset)))
    numdocswritten = 0

    process_pool = multiprocessing.Pool(nump)

    num_input_qas = 0

    # List of tuples with (passage_id, passage_info)
    passage_id_infos = list(dataset.items())
    for (_, pinfo) in passage_id_infos:
        num_input_qas += len(pinfo[constants.qa_pairs])

    print("Making jsonobj chunks")
    passage_id_info_chunks = grouper(100, passage_id_infos)
    print(f"Number of chunks made: {len(passage_id_info_chunks)}")

    output_passage_id_info_dict = {}
    group_num = 1

    num_qa_parsed = 0

    stime = time.time()
    for chunk in passage_id_info_chunks:
        # The main function that processes the input jsonobj
        result = process_pool.map(processPassage, chunk)
        for output_dict in result:
            for (pid, pinfo) in output_dict.items():
                num_qa_parsed += len(pinfo[constants.qa_pairs])
            output_passage_id_info_dict.update(output_dict)

        ttime = time.time() - stime
        ttime = float(ttime) / 60.0
        print(f"Groups done: {group_num} in {ttime} mins")
        group_num += 1

    with open(output_json, 'w') as outf:
        json.dump(output_passage_id_info_dict, outf, indent=4)

    print(f"Number of QA pairs input: {num_input_qas}")
    print(f"Number of QA pairs parsed: {num_qa_parsed}")
    print(f"Multiprocessing finished. Total elems in output: {len(output_passage_id_info_dict)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_json', default=True)
    parser.add_argument('--nump', type=int, default=10)
    args = parser.parse_args()

    # args.input_json --- is the raw json from the DROP dataset
    tokenizeDocs(input_json=args.input_json, output_json=args.output_json, nump=args.nump)




