from typing import List, Tuple
import dateparser
from datasets.drop import constants
import utils.util as util

dateparser_en = dateparser.date.DateDataParser(languages=['en'])

WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december']

NUM_NER_TYPES = ["QUANTITY", "CARDINAL", "PERCENT", "MONEY"]

dateStr2DateObj_cache = {}

def parseDateNERS(ner_spans, passage_tokens: List[str]) -> Tuple[List, List, List, int]:
    """ Returns (List1, List2, int)
        1. List of (text, (start, end), normalized_value_tuple) tuples (end inclusive)
        2. List of date_ent_idxs for equivalent dates (same length as 1.)
        3. List of (normalized_date) values in order of idxs
        4. Number of date_entities
    """

    parsed_dates: List[str, Tuple, Tuple] = []
    for ner in ner_spans:
        if ner[-1] == constants.DATE_TYPE:
            # normalized_dates = List of (text, (start, end), normalized_value_tuple)
            normalized_dates = normalizeDATE(ner, dateparser_en)
            if normalized_dates is not None:
                parsed_dates.extend(normalized_dates)

    year_mentions: List[str, Tuple, Tuple] = extract_years_from_text(passage_tokens=passage_tokens)

    parsed_dates = merge_datener_with_yearmentions(parsed_dates, year_mentions)

    date2idx = {}
    normalized_date_idxs = []
    normalized_date_values = []
    for (_, _, value) in parsed_dates:
        if value not in date2idx:
            date2idx[value] = len(date2idx)
            normalized_date_values.append(value)
        normalized_date_idxs.append(date2idx[value])

    num_date_entities = len(normalized_date_values)
    assert len(parsed_dates) == len(normalized_date_idxs)

    return (parsed_dates, normalized_date_idxs, normalized_date_values, num_date_entities)



def parseNumNERS(ner_spans, tokens: List[str]) -> Tuple[List, List, List, int]:
    """ Returns (List1, List2, int)
        1. List of (text, token_idx, normalized_value)
        2. List of num_ent_idxs for equivalent numbers - same length as 1.
        3. List of normalized_num values in order of idxs
        4. Number of number_entities
    """
    # List of (token_str, token_idx, normalized_value) -- if going by single token version
    # List of ((menstr, start, end, NUM), normalized_value) - if going by the mention route
    parsed_nums = []
    # for ner in ner_spans:
    #     if ner[-1] in NUM_NER_TYPES:
    #         # (token_str, token_idx, normalized_value)
    #         normalized_num = normalizeNUM(ner, tokens)
    #         if normalized_num is not None:
    #             parsed_nums.append(normalized_num)

    for token_idx, token in enumerate(tokens):
        normalized_value = _str2float(token)
        if normalized_value is not None:
            # The number is int, store as one.
            normalized_value = int(normalized_value) if int(normalized_value) == normalized_value else normalized_value
            parsed_nums.append((token, token_idx, normalized_value))

    # Store the passage number values in a sorted manner -- this makes number computations easier in the model
    sorted_parsed_numbers = sorted(parsed_nums, key=lambda x: x[2])

    num2idx = {}
    normalized_num_idxs = []
    normalized_number_values = []
    for (_, _, value) in sorted_parsed_numbers:
        if value not in num2idx:
            num2idx[value] = len(num2idx)
            normalized_number_values.append(value)
        normalized_num_idxs.append(num2idx[value])

    num_number_entities = len(normalized_number_values)
    assert len(parsed_nums) == len(normalized_num_idxs)

    return (sorted_parsed_numbers, normalized_num_idxs, normalized_number_values, num_number_entities)


def merge_datener_with_yearmentions(date_mentions, year_mentions):
    """ Year mentions are single token long. Using an expensive (O(n^2)) process here.
        All mentions are (text, (start, end(inclusive)), normalized_value)
    """
    year_mentions_to_keep = []
    for year_mention in year_mentions:
        token_idx = year_mention[1][0]
        keep = True
        for datemen in date_mentions:
            start, end = datemen[1][0], datemen[1][1]
            if token_idx >= start and token_idx <= end:
                keep = False
        if keep:
            year_mentions_to_keep.append(year_mention)

    final_mentions = date_mentions
    final_mentions.extend(year_mentions_to_keep)
    final_mentions = sorted(final_mentions, key=lambda  x: x[1][0])

    return final_mentions


def extract_years_from_text(passage_tokens) -> List[Tuple[str, Tuple, Tuple]]:
    """ Extract 4 digit and 3 digit year mentions.

        Normalized date value: (day, month, year)
    """
    year_date_mentions = []
    for idx, token in enumerate(passage_tokens):
        if len(token) == 4 or len(token) == 3:
            try:
                int_token = int(token)
                year_date_mentions.append((token, (idx, idx), (-1, -1, int_token)))
            except:
                continue

    return year_date_mentions


def normalizeDATE(date_ner_span, dateparser_en):
    def parseDate(date_str, dateparser_en):
        if date_str not in dateStr2DateObj_cache:
            date = dateparser_en.get_date_data(date_str)
            dateStr2DateObj_cache[date_str] = date['date_obj']
        # if parse fails, date['date_obj'] is None
        return dateStr2DateObj_cache[date_str]

    """ Normalize DATE ner span

        If normalized -- return
        Else: Try the two patterns that contain two dates, but parsed as single mention.
        For eg. "July 5, 1942 – January 11, 2012"

        Parse such into two dates, and return the two values.

        Normalized date value: (day, month, year)
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
        if len(nertext.split(' ')) == 1:
            if len(nertext) == 4:
                month = -1
                day = -1
            elif nertext.lower in MONTHS:
                day = -1
                year = -1
            else:
                # These are usually words like "Monday", "year", etc.
                return None
        # Month Year -- January 2012 OR Day Month
        elif len(nertext.split(' ')) == 2:
            if len(nertext.split(' ')[1]) == 4 and nertext.split(' ')[1].lower() not in MONTHS:
                day = -1
            if nertext.split(' ')[1].lower() in MONTHS:
                year = -1

        normalized_val = (day, month, year)

        if year == 2019:
            return None

        return [(nertext, (start, end-1), normalized_val)]

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

            # We need inclusive ends now
            new_ner_span_1 = (text1, start1, end1 - 1, constants.DATE_TYPE)
            normalized_val_1 = date1

            # We need inclusive ends now
            new_ner_span_2 = (text2, start2, end2 - 1, constants.DATE_TYPE)
            normalized_val_2 = date2

            if normalized_val_1[-1] == 2019 or normalized_val_2[-1] == 2019:
                return None

            return [(text1, (start1, end1-1), normalized_val_1), (text2, (start2, end2-1), normalized_val_2)]

        else:
            return None

def _str2float(string_val):
    # Remove , for strings like 70,0000
    no_comma_string = string_val.replace(",", "")

    if no_comma_string.lower() in WORD_NUMBER_MAP:
        return int(WORD_NUMBER_MAP[no_comma_string.lower()])

    try:
        val = float(no_comma_string)
        return val
    # If error in float parse, return None
    except:
        return None



def normalizeNUM(num_ner, tokens: List[str]):
    """ This normalized num mention in a way to extract a single token.
        For given ner mention, try to resolve the tokens from left2right in the mention.
        If any token resolves, return the (token_str, tokenidx, normalized_value) else NONE
    """
    # End is exclusive
    (text, start, end, nertype) = num_ner
    relevant_tokens = tokens[start:end]
    for idx, token in enumerate(relevant_tokens):
        normalized_value = _str2float(token)
        if normalized_value is not None:
            return (token, start+idx, normalized_value)

    # None of the tokens could be normalized
    return None


# def normalizeNUM(num_ner):
#     if num_ner[-1] == "CARDINAL":
#         return normalizeCARDINAL(num_ner)
#     elif num_ner[-1] == "MONEY":
#         return normalizeMONEY(num_ner)
#     if num_ner[-1] == "PERCENT":
#         return normalizePERCENT(num_ner)
#     if num_ner[-1] == "QUANTITY":
#         return normalizeQUANTITY(num_ner)
#     else:
#         raise NotImplementedError


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
        new_ner_span = (nertext, start, end - 1, constants.NUM_TYPE)
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
        new_ner_span = (nertext, start, end - 1, constants.NUM_TYPE)
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
        new_ner_span = (nertext, start, end - 1, constants.NUM_TYPE)
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
    return normalizeMONEY(quantity_ner_span)