import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any

import dateparser
from dateparser.search import search_dates

from utils import TAUtils, util, spacyutils
from datasets.hotpotqa.utils import constants

spacy_nlp = spacyutils.getSpacyNLP()
# ccg_nlp = TAUtils.getCCGNLPLocalPipeline()


string2num = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
              'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


UNCHANGED_NER_TYPES = ["PERSON", "NORP", "FAC", "ORG", "LOC", "GPE",
                       "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE", "LAW",
                       "TITLE", "PROPN"]

REMOVE_NER_TYPES = ["TIME", "ORDINAL"]

dateStr2DateObj_cache = {}

NUM_TYPE = constants.NUM_TYPE
DATE_TYPE = constants.DATE_TYPE


def _str2float(string_val):
    # Remove , for strings like 70,0000
    string_val = string_val.replace(",", "")

    if string_val.lower() in string2num:
        return float(string2num[string_val.lower()])

    try:
        float_val = float(string_val)
        return float_val
    # If error in float parse, return None
    except:
        return None


def normalizeCARDINAL(cardinal_ner_span):
    """ Normalize CARDINAL ner span

    CARDINAL: 1, 2-time, three, 40,000, one,
    Solution: If
                (1) single token
                (2) string in one-ten map or parses into float
                parse into float and assign NUM type
              else
                return None
    """

    (nertext, start, end, type) = cardinal_ner_span
    assert type == "CARDINAL"

    # Only parse single token cardinals
    if len(nertext.split(' ')) > 1:
        return None

    floatval = _str2float(nertext)

    if floatval is None:
        return None
    else:
        new_ner_span = (nertext, start, end, NUM_TYPE)
        return (new_ner_span, floatval)


def normalizePERCENT(percent_ner_span):
    """ Normalize PERCENT ner span

    PERCENT:  30% / 68.2% / 1%
    Solution: If
                (1) single token
                (2) ending in the % char
                parse rest into float and assign NUM type
              else
                return None
    """

    (nertext, start, end, type) = percent_ner_span
    assert type == "PERCENT"

    # Only parse single token cardinals
    # Since % is a separate token, resort to space based tokenization
    if len(nertext.split(' ')) > 1:
        return None

    stringval = nertext.lower()

    if stringval[-1] == "%":
        floatval = _str2float(stringval[:-1])
    else:
        floatval = None

    if floatval is None:
        return None
    else:
        new_ner_span = (nertext, start, end, NUM_TYPE)
        return (new_ner_span, floatval)


def normalizeMONEY(money_ner_span):
    """ Normalize MONEY ner span

    PERCENT:  up to $600,000 / 70,000 / 400
    Solution: If
                (1) If any token gets parsed to float
                keep that value as normalization and assign NUM type
              else
                return None

    TODO: Need solution for million / billion / trillion
    """

    (nertext, start, end, type) = money_ner_span

    # assert type == "MONEY"

    ner_text_tokens = nertext.split(' ')
    ner_text_tokens = [t.lower() for t in ner_text_tokens]

    floatval = None
    for token in ner_text_tokens:
        if token[0] == "$":
            token = token[1:]
        floatval = _str2float(token)
        if floatval is not None:
            break

    if floatval is None:
        return None
    else:
        new_ner_span = (nertext, start, end, NUM_TYPE)
        return (new_ner_span, floatval)


def normalizeQUANTITY(quantity_ner_span):
    """ Normalize QUANTITY ner span

    PERCENT:  20 fluid ounces / 600 ml / 140 acres
    Solution: If
                (1) If any token gets parsed to float
                keep that value as normalization and assign NUM type
              else
                return None

    TODO: Need solution for million / billion / trillion
    """
    # Same solution as Money.
    return normalizeMONEY(quantity_ner_span)


def parseDate(date_str, dateparser_en):
    if date_str not in dateStr2DateObj_cache:
        date = dateparser_en.get_date_data(date_str)
        dateStr2DateObj_cache[date_str] = date['date_obj']
    # if parse fails, date['date_obj'] is None
    return dateStr2DateObj_cache[date_str]


def normalizeDATE(date_ner_span, dateparser_en):
    """ Normalize QUANTITY ner span

    PERCENT:  20 fluid ounces / 600 ml / 140 acres
    Solution: If
                (1) If any token gets parsed to float
                keep that value as normalization and assign NUM type
              else
                return None

    TODO: Need solution for million / billion / trillion
    """

    (nertext, start, end, type) = date_ner_span
    assert type == "DATE"

    if nertext.lower() == "today" or nertext.lower() == "tomorrow" or nertext.lower() == "yesterday":
        return None

    try:
        # Single date in the span
        date = parseDate(nertext, dateparser_en)

        year = date.year
        month = date.month
        day = date.day

        # If span is incomplete date
        # Only Year: 1980 / 2017
        if len(nertext.split(' ')) == 1 and len(nertext) == 4:
            month = -1
            day = -1
        # Month Year -- January 2012
        elif len(nertext.split(' ')) == 2 and len(nertext.split(' ')[1]) == 4:
            day = -1

        normalized_val = (day, month, year)

        new_ner_span = (nertext, start, end, DATE_TYPE)

        return [(new_ner_span, normalized_val)]

    except:
        # # Try searching dates in the string
        # dates = search_dates(nertext)
        #
        # if dates is None:
        #     return None
        #
        # if len(dates) != 2:
        #     return None

        # dates: List of (string, datetime.datetime) tuples

        # To avoid noise
        # Try parsing strings of the kind:
        #   July 5, 1942 – January 11, 2012   start:5 end:14
        #   September 14, 2006 to March 18, 2007 start:5 end:14
        #   14 December 1875 – 1 January 1964  start:7 end:14
        #   1899 to 1968  start: 21 end: 24

        tokens = nertext.split(' ')
        if ("to" in tokens or "-" in tokens):
            # Covers first three cases
            if len(tokens) == 7:

                string1 = ' '.join(tokens[0:3])
                string2 = ' '.join(tokens[4:7])

                date1, date2 = None, None
                try:
                    date1 = parseDate(string1, dateparser_en)
                    date2 = parseDate(string2, dateparser_en)
                except:
                    date1, date2 = None, None

                if date1 is None or date2 is None:
                    return None

                text1 = string1
                date1 = (date1.day, date1.month, date1.year)
                text2 = string2
                date2 = (date2.day, date2.month, date2.year)

                # Covers first 2 cases
                if end - start == 9:
                    start1 = start
                    end1 = start + 4

                    start2 = start + 5
                    end2 = end

                # Covers 3rd case
                elif end - start == 7:
                    start1 = start
                    end1 = start + 3

                    start2 = start + 4
                    end2 = end

                else:
                    return None

            # Covers 4th case
            elif len(tokens) == 3:
                string1 = tokens[0]
                string2 = tokens[2]

                date1, date2 = None, None
                try:
                    date1 = parseDate(string1, dateparser_en)
                    date2 = parseDate(string2, dateparser_en)
                except:
                    date1, date2 = None, None

                if date1 is None or date2 is None:
                    return None

                text1 = string1
                start1 = start
                end1 = start + 1
                date1 = (-1, -1, date1.year)

                text2 = string2
                start2 = start + 2
                end2 = end
                date2 = (-1, -1, date2.year)
            else:
                return None

            new_ner_span_1 = (text1, start1, end1, DATE_TYPE)
            normalized_val_1 = date1

            new_ner_span_2 = (text2, start2, end2, DATE_TYPE)
            normalized_val_2 = date2

            return [(new_ner_span_1, normalized_val_1), (new_ner_span_2, normalized_val_2)]

        else:
            return None


def cleanNERList(ners: List[Tuple], NUMBER_DICT, DATE_DICT, dateparser_en):

    normalized_ner_spans = []


    for ner_span in ners:
        nertype = ner_span[-1]
        if nertype in UNCHANGED_NER_TYPES:
            normalized_ner_spans.append(ner_span)
        elif nertype in REMOVE_NER_TYPES:
            continue
        else:
            # For each nertype, returnval = (ner_span, normalized_val)
            # Add the ner_span to the new list, and add the text:normalized_val mapping to the apt dict
            if nertype == "CARDINAL":
                returnval = normalizeCARDINAL(ner_span)
                apt_dict = NUMBER_DICT
            elif nertype == "PERCENT":
                returnval = normalizePERCENT(ner_span)
                apt_dict = NUMBER_DICT
            elif nertype == "MONEY":
                returnval = normalizeMONEY(ner_span)
                apt_dict = NUMBER_DICT
            elif nertype == "QUANTITY":
                returnval = normalizeQUANTITY(ner_span)
                apt_dict = NUMBER_DICT
            elif nertype == "DATE":
                returnval = normalizeDATE(ner_span, dateparser_en)
                apt_dict = DATE_DICT
            else:
                returnval = None
                apt_dict = None
                print(ner_span)
                print("WHY HERE *************")

            if returnval is not None:
                if nertype != "DATE":
                    span, normalized_val = returnval
                    span_text = span[0]
                    normalized_ner_spans.append(span)
                    apt_dict[span_text] = normalized_val
                # DATE Normalization returns a list
                else:
                    returnval_list = returnval
                    # if len(returnval_list) > 1:
                    #     print(ner_span)
                    for returnval in returnval_list:
                        span, normalized_val = returnval
                        # if len(returnval_list) > 1:
                        #     print(f"{span}  -->  {normalized_val}")
                        span_text = span[0]
                        normalized_ner_spans.append(span)
                        apt_dict[span_text] = normalized_val
                    # if len(returnval_list) > 1:
                    #     print("\n")

            else:
                continue

    return (normalized_ner_spans, NUMBER_DICT, DATE_DICT)



def cleanNERSForJsonl(input_jsonl: str, output_jsonl: str) -> None:
    """ Normalize numbers and dates in a tokenized Jsonl and output a new jsonl file
    Remove number and date entities that don't normalize or are ill formated

    Returns:
    --------
    Jsonl file with same datatypes as input with the modification/addition of:
    Modifications:
        q_ner_field: The question is tokenized
        ans_ner_field: The answer is now modified
        context_ner_field: Context sentences are now tokenized, but stored with white-space delimition

    Additions:
        q_dates_field
        ans_dates_field
        context_dates_field
        q_nums_field
        ans_nums_field
        context_nums_field
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_jsonl))

    # Input file contains single json obj with list of questions as jsonobjs inside it

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    dateparser_en = dateparser.date.DateDataParser(languages=['en'])

    with open(output_jsonl, 'w') as outf:
        for jsonobj in jsonobjs:

            NUMBER_DICT, DATE_DICT = {}, {}

            new_doc = copy.deepcopy(jsonobj)

            q_ners = new_doc[constants.q_ner_field]
            contexts_ners = new_doc[constants.context_ner_field]

            (q_normalized_ners, NUMBER_DICT, DATE_DICT) = cleanNERList(q_ners, NUMBER_DICT, DATE_DICT, dateparser_en)
            new_doc[constants.q_ner_field] = q_normalized_ners

            context_normalized_ners = []
            for context in contexts_ners:
                one_context_normalized_ners = []
                for sentence_ner in context:
                    (normalized_ner_spans, NUMBER_DICT, DATE_DICT) = cleanNERList(sentence_ner, NUMBER_DICT,
                                                                                  DATE_DICT, dateparser_en)
                    one_context_normalized_ners.append(normalized_ner_spans)
                context_normalized_ners.append(one_context_normalized_ners)

            new_doc[constants.context_ner_field] = context_normalized_ners

            new_doc[constants.nums_normalized_field] = NUMBER_DICT
            new_doc[constants.dates_normalized_field] = DATE_DICT


            outf.write(json.dumps(new_doc))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 10 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('Parsing dates: {}'.format(args.dates))
    print('Parsing numbers: {}'.format(args.numbers))

    # args.input_jsonl --- is the output from preprocess.tokenize
    cleanNERSForJsonl(input_jsonl=args.input_jsonl, output_jsonl=args.output_jsonl)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl', default=True)
    parser.add_argument('--dates', action='store_true', default=False)
    parser.add_argument('--numbers', action='store_true', default=False)
    args = parser.parse_args()

    main(args)