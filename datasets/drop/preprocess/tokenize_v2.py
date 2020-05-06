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

from allennlp.data.tokenizers import Token
from utils import util, spacyutils
from datasets.drop import constants
from datasets.drop.preprocess import ner_process

import multiprocessing

IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])

spacy_nlp = spacyutils.getSpacyNLP()
spacy_whitespacetokenizer = spacyutils.getWhiteTokenizerSpacyNLP()

ANSWER_TYPE_NOT_FOUND = 0

# def split_on_hyphens(text: str):
#     """ Adds spaces around hyphens in text. """
#     text = util.pruneMultipleSpaces(" - ".join(text.split("-")))
#     text = util.pruneMultipleSpaces(" – ".join(text.split("–")))
#     text = util.pruneMultipleSpaces(" ~ ".join(text.split("~")))
#     return text


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

    chunk_size = n
    return [iterable[i : i + chunk_size] for i in range(0, len(iterable), chunk_size)]


def _check_validity_of_spans(spans: List[Tuple[int, int]], len_seq: int):
    """All spans are inclusive start and end"""
    for span in spans:
        assert span[0] >= 0
        assert span[1] < len_seq


def find_valid_spans(passage_tokens: List[str], answer_texts: List[str]) -> List[Tuple[int, int]]:

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
        date_tokens = [
            answer_content[key] for key in ["month", "day", "year"] if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def _convert_orig_token_span_to_hyphen_tokenization_idxs(orig_token_idx_spans: List[Tuple[int, int]],
                                                         original_tokens: List[Token], tokens: List[Token]):
    token_charidxs = [token.idx for token in tokens]
    charidx2tokenidx = {}
    for tokenidx, (char_idx, t) in enumerate(zip(token_charidxs, tokens)):
        charidx2tokenidx[char_idx] = tokenidx                       # start of token
        charidx2tokenidx[char_idx + len(t.text) - 1] = tokenidx     # end of token

    # orig_token_idx_spans: (start-inclusive, end-exclusive)
    span_charidxs = [(original_tokens[x].idx, original_tokens[y - 1].idx + len(original_tokens[y - 1]))
                     for (x, y) in orig_token_idx_spans]

    span_token_idxs = [(charidx2tokenidx[start_char_idx], charidx2tokenidx[end_char_idx - 1] + 1) for
                           (start_char_idx, end_char_idx) in span_charidxs]
    return span_token_idxs

def processPassage(input_args):
    """ This function needs to accomplish the following:
    1. Tokenize the passage and maintain char-idxs into original passage
    2. Tokenize the question and maintain char-idxs into original passage
    3. Find spans that are numbers and dates in the question and the passage, represent tokens-ids and normalized values
    4. Find the answer type, if span (or date) find the token-ids

    """

    passage_id, passage_info = input_args

    passage_info_additions = {}

    passage_text: str = passage_info[constants.passage]
    passage_spacydoc = spacyutils.getSpacyDoc(passage_text, spacy_nlp)
    original_passage_tokens = [t for t in passage_spacydoc]
    passage_tokens: List[Token] = split_tokens_by_hyphen(original_passage_tokens)
    passage_token_charidxs = [token.idx for token in passage_tokens]
    passage_token_texts: List[str] = [t.text for t in passage_tokens]

    passage_info["passage_tokens"] = passage_token_texts
    passage_info[constants.passage_charidxs] = passage_token_charidxs

    # List[Tuple[int, int]] -- start (inclusive) and end (exclusive) token idxs for sentence boundaries
    # These token_idxs are in terms of the original tokenization, before splitting on hypens
    sentence_original_tokens_idxs = sorted([(sentence.start, sentence.end) for sentence in passage_spacydoc.sents],
                                           key=lambda x: x[0])
    sentence_token_idxs = _convert_orig_token_span_to_hyphen_tokenization_idxs(
        orig_token_idx_spans=sentence_original_tokens_idxs, original_tokens=original_passage_tokens,
        tokens=passage_tokens)

    passage_info[constants.passage_sent_idxs] = sentence_token_idxs

    # List of (text, ent.start, ent.end, ent.label_) tuples -- these token-idxs are original idxs
    passage_ners = spacyutils.getNER(passage_spacydoc)
    ner_orig_token_idx_spans = [(x, y) for (_, x, y, _) in passage_ners]
    ner_text_labels = [(x, y) for (x, _, _, y) in passage_ners]
    ner_token_idxs = _convert_orig_token_span_to_hyphen_tokenization_idxs(ner_orig_token_idx_spans,
                                                                          original_passage_tokens, passage_tokens)
    passage_ners = [(a, x, y, b) for ((a, b), (x, y)) in zip(ner_text_labels, ner_token_idxs)]

    (parsed_dates, normalized_date_idxs, normalized_date_values, num_date_entities) = ner_process.parseDateNERS(
        passage_ners, passage_token_texts
    )
    _check_validity_of_spans(spans=[(s, e) for _, (s, e), _ in parsed_dates], len_seq=len(passage_tokens))
    # normalized_number_values - is sorted and mentions are arranged accordingly
    (parsed_nums, normalized_num_idxs, normalized_number_values, num_num_entities) = ner_process.parseNumNERS(
        passage_ners, passage_token_texts
    )

    # Adding NER Value
    passage_info[constants.passage_date_mens] = parsed_dates
    passage_info[constants.passage_date_entidx] = normalized_date_idxs
    passage_info[constants.passage_date_normalized_values] = normalized_date_values

    passage_info[constants.passage_num_mens] = parsed_nums
    passage_info[constants.passage_num_entidx] = normalized_num_idxs
    passage_info[constants.passage_num_normalized_values] = normalized_number_values

    # Maybe add whitespace info later
    qa_pairs: List[Dict] = passage_info[constants.qa_pairs]
    for qa in qa_pairs:
        additional_qa_info = {}
        question: str = qa[constants.question]
        q_spacydoc = spacyutils.getSpacyDoc(question, spacy_nlp)
        original_question_tokens = [t for t in q_spacydoc]
        question_tokens: List[Token] = split_tokens_by_hyphen(original_question_tokens)
        question_token_charidxs = [token.idx for token in question_tokens]
        question_token_texts = [t.text for t in question_tokens]

        qa["question_tokens"] = question_token_texts
        qa[constants.question_charidxs] = question_token_charidxs

        q_ners = spacyutils.getNER(q_spacydoc)
        ner_orig_token_idx_spans = [(x, y) for (_, x, y, _) in q_ners]
        ner_text_labels = [(x, y) for (x, _, _, y) in q_ners]
        ner_token_idxs = _convert_orig_token_span_to_hyphen_tokenization_idxs(ner_orig_token_idx_spans,
                                                                              original_question_tokens, question_tokens)
        q_ners = [(a, x, y, b) for ((a, b), (x, y)) in zip(ner_text_labels, ner_token_idxs)]
        (parsed_dates, normalized_date_idxs, normalized_date_values, num_date_entities) = ner_process.parseDateNERS(
            q_ners, question_token_texts
        )
        _check_validity_of_spans(spans=[(s, e) for _, (s, e), _ in parsed_dates], len_seq=len(question_tokens))
        (parsed_nums, normalized_num_idxs, normalized_number_values, num_num_entities) = ner_process.parseNumNERS(
            q_ners, question_token_texts
        )

        qa[constants.q_date_mens] = parsed_dates
        qa[constants.q_date_entidx] = normalized_date_idxs
        qa[constants.q_date_normalized_values] = normalized_date_values

        qa[constants.q_num_mens] = parsed_nums
        qa[constants.q_num_entidx] = normalized_num_idxs
        qa[constants.q_num_normalized_values] = normalized_number_values

        answer = qa[constants.answer]

        # Answer type and list of answer-texts
        answer_content: Tuple[str, List] = convert_answer(answer)
        if answer_content is None:
            # Some qa don't have any answer annotated
            print(f"Couldn't resolve answer: {answer}")
            continue

        answer_type, answer_texts = answer_content

        # answer_texts_for_evaluation = [' '.join(answer_texts)]
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_spacydoc = spacyutils.getSpacyDoc(answer_text, spacy_nlp)
            original_answer_tokens = [t for t in answer_spacydoc]
            answer_tokens: List[Token] = split_tokens_by_hyphen(original_answer_tokens)
            answer_token_texts = [t.text for t in answer_tokens]
            tokenized_answer_texts.append(" ".join(answer_token_texts))

        valid_passage_spans = (
            find_valid_spans(passage_token_texts, tokenized_answer_texts) if tokenized_answer_texts else []
        )
        valid_question_spans = (
            find_valid_spans(question_token_texts, tokenized_answer_texts) if tokenized_answer_texts else []
        )

        _check_validity_of_spans(valid_question_spans, len(question_tokens))

        # Adding question/passage spans as answers
        qa[constants.answer_passage_spans] = valid_passage_spans
        qa[constants.answer_question_spans] = valid_question_spans

        # ANSWER TYPE
        if answer_type == "spans":
            if valid_passage_spans or valid_question_spans:
                qa[constants.answer_type] = constants.SPAN_TYPE
            else:
                print(f"Answer span not found in passage. passageid: {passage_id}. queryid: {qa[constants.query_id]}")
                print(f"Answer Texts: {answer_texts}")
                print(f"Tokenized_Answer Texts: {tokenized_answer_texts}")
                print(passage_info[constants.passage])
                qa[constants.answer_type] = constants.UNK_TYPE
                print()
        elif answer_type == "number":
            qa[constants.answer_type] = constants.NUM_TYPE
        elif answer_type == "date":
            qa[constants.answer_type] = constants.DATE_TYPE

    return {passage_id: passage_info}


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
    with open(input_json, "r") as f:
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

    with open(output_json, "w") as outf:
        json.dump(output_passage_id_info_dict, outf, indent=4)

    print(f"Number of QA pairs input: {num_input_qas}")
    print(f"Number of QA pairs parsed: {num_qa_parsed}")
    print(f"Multiprocessing finished. Total elems in output: {len(output_passage_id_info_dict)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is the original drop_old json file
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", default=True)
    parser.add_argument("--nump", type=int, default=10)
    args = parser.parse_args()

    # args.input_json --- is the raw json from the DROP dataset
    tokenizeDocs(input_json=args.input_json, output_json=args.output_json, nump=args.nump)
