import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any

import dateparser

from utils import util, spacyutils
from datasets.hotpotqa.utils import constants
import multiprocessing

spacy_nlp = spacyutils.getSpacyNLP()


string2num = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
              'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


UNCHANGED_NER_TYPES = ["PERSON", "NORP", "FAC", "ORG", "LOC", "GPE",
                       "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE", "LAW",
                       "TITLE", "PROPN"]

REMOVE_NER_TYPES = ["TIME", "ORDINAL"]

dateStr2DateObj_cache = {}

NUM_TYPE = constants.NUM_TYPE
DATE_TYPE = constants.DATE_TYPE


# Context mentions can be of three types:

# All unchanged types will be mapped to the ENT type in out domain
ENT_TYPE_TO_NERTYPE = UNCHANGED_NER_TYPES

# All these NER types are mapped ot NUM type in out domain
NUM_TYPE_TO_NERTYPE = ["QUANTITY", "CARDINAL", "PERCENT", "MONEY"]

# "DATE" NER type is the only type getting mapped to DATE type in our domain
DATE_TYPE_TO_NERTYPE = ["DATE"]




def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

    chunk_size = n
    return [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]


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

    (nertext, start, end, type) = quantity_ner_span

    # ner_text_tokens = nertext.split(' ')
    # ner_text_tokens = [t.lower() for t in ner_text_tokens]
    #
    # floatval = None
    # for token in ner_text_tokens:
    #     floatval = _str2float(token)
    #     if floatval is not None:
    #         break
    #
    # if floatval is None:
    #     return None
    # else:
    #     new_ner_span = (nertext, start, end, NUM_TYPE)
    #     return (new_ner_span, floatval)

    # TODO: Same solution as Money.
    return normalizeMONEY(quantity_ner_span)


def parseDate(date_str, dateparser_en):
    if date_str not in dateStr2DateObj_cache:
        date = dateparser_en.get_date_data(date_str)
        dateStr2DateObj_cache[date_str] = date['date_obj']
    # if parse fails, date['date_obj'] is None
    return dateStr2DateObj_cache[date_str]


def normalizeDATE(date_ner_span, dateparser_en):
    """ Normalize DATE ner span

        If normalized -- return
        Else: Try the two patterns that contain two dates, but parsed as single mention.
        For eg. "July 5, 1942 – January 11, 2012"

        Parse such into two dates, and return the two values.

        Normalized date value: (date, month, year)
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
        # Additionally try parsing strings of the kind:
        #   "July 5, 1942 – January 11, 2012"   start:5 end:14
        #   "September 14, 2006 to March 18, 2007" start:5 end:14
        #   "14 December 1875 – 1 January 1964"  start:7 end:14
        #   "1899 to 1968"  start: 21 end: 24

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
    """ Clean and normalize a list of NEs. Return new list
    TYPES include all from UNCHANGED_NER_TYPES + NUM + DATE
    :param ners:
    :param NUMBER_DICT:
    :param DATE_DICT:
    :param dateparser_en:
    :return:
    """
    normalized_ner_spans = []

    for ner_span in ners:
        nertype = ner_span[-1]

        # These type NEs are kept as is
        if nertype in UNCHANGED_NER_TYPES:
            ner_span[-1] = constants.ENTITY_TYPE
            normalized_ner_spans.append(ner_span)

        # These type NEs are removed
        elif nertype in REMOVE_NER_TYPES:
            continue

        else:
            # These NEs will be tried for normalization
            # Each normalization function returns a returnval
            # If: returnval == None ==> Normalization failed. Remove NER
            # Else: returnval = (ner_span, normalized_val)
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
                # Shouldn't need to come here since all NE types are covered above
                returnval = None
                apt_dict = None
                print(ner_span)
                print("************  WHY HERE  *************")

            if returnval is not None:
                if nertype != "DATE":
                    # normalized_span contains new type NUM
                    normalized_span, normalized_val = returnval
                    span_text = normalized_span[0]
                    normalized_ner_spans.append(normalized_span)
                    apt_dict[span_text] = normalized_val
                # DATE Normalization returns a list since a single span can be broken to two spans
                # Eg. "12 September 2004 to 19th October 2017"
                else:
                    returnval_list = returnval
                    # if len(returnval_list) > 1:
                    #     print(ner_span)
                    for returnval in returnval_list:
                        # normalized_span contains new type DATE
                        normalized_span, normalized_val = returnval
                        # if len(returnval_list) > 1:
                        #     print(f"{span}  -->  {normalized_val}")
                        span_text = normalized_span[0]
                        normalized_ner_spans.append(normalized_span)
                        apt_dict[span_text] = normalized_val
                    # if len(returnval_list) > 1:
                    #     print("\n")

            else:
                # Normalization failed, skip NE
                continue

    return (normalized_ner_spans, NUMBER_DICT, DATE_DICT)


def getNERSpansByType(ners: List):
    ent_mens, num_mens, date_mens = [], [], []
    for ner_span in ners:
        nertype = ner_span[-1]
        if nertype == constants.ENTITY_TYPE:
            ent_mens.append(ner_span)
        elif nertype == constants.NUM_TYPE:
            num_mens.append(ner_span)
        elif nertype == constants.DATE_TYPE:
            date_mens.append(ner_span)
        else:
            print(f"NER type doesn't belong to contexts. Type: {ner_span}")

    return (ent_mens, num_mens, date_mens)



def processJsonObj(input_jsonobj):
    dateparser_en = dateparser.date.DateDataParser(languages=['en'])

    # number/date string 2 normalized value dict
    NUMBER_DICT, DATE_DICT = {}, {}

    new_doc = copy.deepcopy(input_jsonobj)

    q_ners = new_doc[constants.q_ner_field]
    contexts_ners = new_doc[constants.context_ner_field]

    (q_normalized_ners, NUMBER_DICT, DATE_DICT) = cleanNERList(q_ners, NUMBER_DICT, DATE_DICT, dateparser_en)
    (q_ent_ners, q_num_ners, q_date_ners) = getNERSpansByType(q_normalized_ners)

    # Delete the NEr key and instead make three corresponding to ENT, NUM, and DATE
    new_doc.pop(constants.q_ner_field)
    new_doc[constants.q_ent_ner_field] = q_ent_ners
    new_doc[constants.q_num_ner_field] = q_num_ners
    new_doc[constants.q_date_ner_field] = q_date_ners

    # context_normalized_ners = []
    context_ent_normalized_ners = []
    context_num_normalized_ners = []
    context_date_normalized_ners  = []

    for context in contexts_ners:
        ent_onecontext_normalized_ners = []
        num_onecontext_normalized_ners = []
        date_onecontext_normalized_ners = []

        for sentence_ner in context:
            # NUMBER_DICT: String to float num
            # DATE_DICT: String to number-tuple of [day, month, year]. -1 if invalid
            (normalized_ner_spans, NUMBER_DICT, DATE_DICT) = cleanNERList(sentence_ner, NUMBER_DICT,
                                                                          DATE_DICT, dateparser_en)
            ent_mens, num_mens, date_mens = getNERSpansByType(normalized_ner_spans)
            ent_onecontext_normalized_ners.append(ent_mens)
            num_onecontext_normalized_ners.append(num_mens)
            date_onecontext_normalized_ners.append(date_mens)

        # context_normalized_ners.append(one_context_normalized_ners)
        context_ent_normalized_ners.append(ent_onecontext_normalized_ners)
        context_num_normalized_ners.append(num_onecontext_normalized_ners)
        context_date_normalized_ners.append(date_onecontext_normalized_ners)

    new_doc.pop(constants.context_ner_field)
    new_doc[constants.context_ent_ner_field] = context_ent_normalized_ners
    new_doc[constants.context_num_ner_field] = context_num_normalized_ners
    new_doc[constants.context_date_ner_field] = context_date_normalized_ners

    new_doc[constants.nums_normalized_field] = NUMBER_DICT
    new_doc[constants.dates_normalized_field] = DATE_DICT

    return new_doc


def cleanNERSForJsonl(input_jsonl: str, output_jsonl: str, num_processes: float) -> None:
    """ Clean and normalize NEs in a tokenized jsonl

    Normalize dates and numbers. Remove number and date entities that don't normalize or are ill formated

    Returns:
    --------
    Jsonl file with same datatypes as input with the modification/addition of:
    Modifications:
        q_ner_field
        context_ner_field

    Additions:
        dates_normalized_field: Dict from date string to normalized val
        nums_normalized_field: Dict from num string to normalized val
    """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_jsonl))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    # dateparser_en = dateparser.date.DateDataParser(languages=['en'])
    # Create pool (p)
    process_pool = multiprocessing.Pool(num_processes)

    print("Making jsonobj chunks")
    jsonobj_chunks = grouper(100, jsonobjs)
    print(f"Number of chunks made: {len(jsonobj_chunks)}")

    output_jsonobjs = []
    group_num = 1

    stime = time.time()
    for chunk in jsonobj_chunks:
        # Main function that does the processing
        result = process_pool.map(processJsonObj, chunk)
        output_jsonobjs.extend(result)

        ttime = time.time() - stime
        ttime = float(ttime) / 60.0
        print(f"Groups done: {group_num} in {ttime} mins")
        group_num += 1

    print(f"Multiprocessing finished. Total elems in output: {len(output_jsonobjs)}")

    print(f"Writing processed documents to jsonl: {output_jsonl}")
    with open(output_jsonl, 'w') as outf:
        for jsonobj in output_jsonobjs:
            outf.write(json.dumps(jsonobj))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):

    if args.replace:
        print("Replacing original file")
        output_jsonl = args.input_jsonl
    else:
        print("Outputting a new file")
        assert args.output_jsonl is not None, "Output_jsonl needed"
        output_jsonl = args.output_jsonl

    # args.input_jsonl --- is the output from preprocess.tokenize
    cleanNERSForJsonl(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl,
                      num_processes=args.nump)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl')
    parser.add_argument('--replace', action="store_true", default=False)
    parser.add_argument('--nump', type=int, default=10)
    args = parser.parse_args()

    main(args)