import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from analysis.qdmr.program_diagnostics import is_potential_filter_num
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp
from allennlp.data.tokenizers import SpacyTokenizer

from datasets.drop import constants

""" These post-processing strategies are written after diagnostics from `analysis.qdmr.program_diagnostics` which should
    be consulted to see what conditions were put to filter questions of a certain kind. 
    Analysis can be found here: 
    https://docs.google.com/spreadsheets/d/1tDQOolYV_J5T9TcogTswz-MSfZ0gAH_J4OUxOfKYMCI/edit#gid=0
"""

spacy_tokenizer = SpacyTokenizer()


def tokenize(text):
    tokens = spacy_tokenizer.tokenize(text)
    tokens = [t.text for t in tokens]
    tokens = [x for t in tokens for x in t.split("-")]
    return tokens


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def filter_num_classifier(string_arg: str, module: str):
    """ Returns if a QSPAN string matches any of the FILTER_NUM_COND. If yes, finds the COND and the question-number
        Returns:
            matches: bool denoting if the qspan matched any patterns
            filter_type: one of "LT", "GT", "EQ", "LT_EQ", "GT_EQ".
            number: The number extracted from the input qspan string for the condition
    """

    def extract_number(string_arg):
        number_regex_prog = re.compile("[0-9]+(|\+)")
        number = None
        tokens = tokenize(string_arg)
        reversed_tokens = tokens[::-1]
        for t in reversed_tokens:
            if number_regex_prog.fullmatch(t) is not None:
                number = t
                break
        return number

    patterns_for_gt = [
        "over [0-9]+",                                      # "over #NUM"
        "over [0-9]+( |\-)\w+",                             # "over #NUM-yards"
        "that (are|was|were) over [0-9]+$",                 # "that are over #NUM"
        "that (are|was|were) over [0-9]+( |\-)\w+$",        # "that are over #NUM yards"
        "that (are|was|were) over [0-9]+( |\-)\w+\slong$",  # "that are over #NUM yards long"
        "that (are|was|were) from over [0-9]+$",            # "that are over #NUM"
        "that (are|was|were) from over [0-9]+( |\-)\w+$",   # "that are over #NUM yards"
        "that (are|was|were) longer than [0-9]+$",          # "that are/was longer than 40"
        "that (are|was|were) more than [0-9]+$",            # "that are/was longer than 40"
        "that (are|was|were) longer than [0-9]+( |\-)\w+$",  # "that are/was longer than 40 yards"
        "that (are|was|were) more than [0-9]+( |\-)\w+$",  # "that are/was longer than 40 yards"
        "that (are|was|were) [0-9]+\+ \w+$",                 # "that are 40+ yard"
        "is (higher|longer) than [0-9]+$",
        "is (higher|longer) than [0-9]+( |\-)\w+$$",
    ]

    patterns_for_gt_eq = [
        "that (are|was|were) atleast [0-9]+$",  # "that are/was atleast 40"
        "that (are|was|were) atleast [0-9]+( |\-)\w+$",  # "that are/was atleast 40 yards"
        "that (are|was|were) at least [0-9]+$",  # "that are/was at least 40"
        "that (are|was|were) at least [0-9]+( |\-)\w+$",  # "that are/was at least 40 yards"
        "where the size is atleast [0-9]+$",
        "where the size is atleast [0-9]+( |\-)\w+$",
        "where the size is at least [0-9]+$",
        "where the size is at least [0-9]+( |\-)\w+$",
        "is atleast [0-9]+$",
        "is atleast [0-9]+( |\-)\w+$",
        "is at least [0-9]+$",
        "is at least [0-9]+( |\-)\w+$",
        "atleast [0-9]+$",
        "atleast [0-9]+( |\-)\w+$$",
        "at least [0-9]+$",
        "at least [0-9]+( |\-)\w+$$",
        "[0-9]+\s\w+\sor longer",  # 10 yards or longer
        "[0-9]+\s\w+\sor more",  # 10 yards or more
        "that (are|was|were) [0-9]+\s\w+\sor longer",  # 10 yards or longer
        "that (are|was|were) [0-9]+\s\w+\sor more",  # 10 yards or longer
    ]

    patterns_for_eq = [
        "equal to [0-9]+$",                                  # "equal to #NUM"
        "equal to [0-9]+( |\-)\w+$",                         # "equal to #NUM-yards"
        "that (are|was|were) equal to [0-9]+$",              # "that are equal to #NUM"
        "that (are|was|were) equal to [0-9]+( \-)\w+$$",     # "that are equal to #NUM yard"
        "that (are|was|were) [0-9]+$",                       # "that are 40"
        "that (are|was|were) [0-9]+( |\-)\w+$",              # "that are 40 yards"
    ]

    patterns_for_lt = [
        "under [0-9]+$",                                       # "under #NUM"
        "under [0-9]+( |\-)\w+$",                              # "under #NUM-yards"
        "that (are|was|were) under [0-9]+$",                   # "that are under #NUM"
        "that (are|was|were) under [0-9]+( |\-)\w+$",          # "that are under #NUM-yard"
        "that (are|was|were) from under [0-9]+$",              # "that are under #NUM"
        "that (are|was|were) from under [0-9]+( |\-)\w+$",     # "that are under #NUM-yard"
        "that (are|was|were) less than [0-9]+$",               # "that are/was less than 40"
        "that (are|was|were) less than [0-9]+( |\-)\w+$",      # "that are/was less than 40 yards"
        "that (are|was|were) shorter than [0-9]+( |\-)\w+$",   # "that are/was shorter than 40 yards"
        "that (are|was|were) shorter than [0-9]+$",            # "that are/was shorter than 40"
        "is lower than [0-9]+$",                               # "is lower than #NUM"
        "less than [0-9]+$",
        "less than [0-9]+( |\-)\w+$$",
        "is less than [0-9]+$",
        "is less than [0-9]+( |\-)\w+$$",
    ]

    patterns_for_lt_eq = [
        "that (are|was|were) atmost [0-9]+$",  # "that are/was atmost 40"
        "that (are|was|were) atmost [0-9]+( |\-)\w+$",  # "that are/was atmost 40 yards"
        "that (are|was|were) at most [0-9]+$",  # "that are/was at most 40"
        "that (are|was|were) at most [0-9]+( |\-)\w+$",  # "that are/was at most 40 yards"
        "atmost [0-9]+$",
        "atmost [0-9]+( |\-)\w+$$",
        "at most [0-9]+$",
        "at most [0-9]+( |\-)\w+$$",
        "that (is|are|was|were) at most [0-9]+( |\-)\w+",  # that is at most 10 yards
        "that (is|are|was|were) at most [0-9]+",  # that is at most 10 yards

        "that (are|was|were) [0-9]+\sor shorter",  # 10 yards or shorter
        "that (are|was|were) [0-9]+\sor less",  # 10 yards or less
        "that (are|was|were) [0-9]+\sor fewer",  # 10 yards or fewer
        "that (are|was|were) [0-9]+\s\w+\sor shorter",  # 10 yards or shorter
        "that (are|was|were) [0-9]+\s\w+\sor less",  # 10 yards or less
        "that (are|was|were) [0-9]+\s\w+\sor fewer",  # 10 yards or fewer
    ]

    regex_progs_gt = [re.compile(p) for p in patterns_for_gt]
    regex_progs_lt = [re.compile(p) for p in patterns_for_lt]
    regex_progs_gt_eq = [re.compile(p) for p in patterns_for_gt_eq]
    regex_progs_lt_eq = [re.compile(p) for p in patterns_for_lt_eq]
    regex_progs_eq = [re.compile(p) for p in patterns_for_eq]

    if module == "filter":
        # If string-arg comes from filter - perform full-match and return the filter-num-type and the number
        match_bool = False
        # One of ["EQ", "LT", "GT", "GT_EQ", "LT_EQ"]
        filter_type = None
        for regex in regex_progs_eq:
            if regex.match(string_arg) is not None:
                match_bool = True
                filter_type = "EQ"

        for regex in regex_progs_lt:
            if regex.match(string_arg) is not None:
                match_bool = True
                filter_type = "LT"

        for regex in regex_progs_gt:
            if regex.match(string_arg) is not None:
                match_bool = True
                filter_type = "GT"

        for regex in regex_progs_gt_eq:
            if regex.match(string_arg) is not None:
                match_bool = True
                filter_type = "GT_EQ"

        for regex in regex_progs_lt_eq:
            if regex.match(string_arg) is not None:
                match_bool = True
                filter_type = "LT_EQ"

        # Matches #NUM or #NUM+
        number = extract_number(string_arg)

        return match_bool, filter_type, number

    elif module == "select":
        # If string-arg comes from select - perform partial-find and select the one with the longest match,
        #  remove the part that matched from the string-arg, and return the remaining string-arg.
        #  Also return filter-num-type and
        match_bool = False
        matches = []
        matches_filter_type = []
        # One of ["EQ", "LT", "GT", "GT_EQ", "LT_EQ"]
        filter_type = None

        # EQ regexes
        for regex in regex_progs_eq:
            regex_matches = list(regex.finditer(string_arg))
            if regex_matches:   # If len(list) > 0
                match_bool = True
                matches.extend(regex_matches)
                matches_filter_type.extend(["EQ"] * len(regex_matches))
        # LT regexes
        for regex in regex_progs_lt:
            regex_matches = list(regex.finditer(string_arg))
            if regex_matches:  # If len(list) > 0
                match_bool = True
                matches.extend(regex_matches)
                matches_filter_type.extend(["LT"] * len(regex_matches))
        # GT regexes
        for regex in regex_progs_gt:
            regex_matches = list(regex.finditer(string_arg))
            if regex_matches:  # If len(list) > 0
                match_bool = True
                matches.extend(regex_matches)
                matches_filter_type.extend(["GT"] * len(regex_matches))
        # GT-EQ regexes
        for regex in regex_progs_gt_eq:
            regex_matches = list(regex.finditer(string_arg))
            if regex_matches:  # If len(list) > 0
                match_bool = True
                matches.extend(regex_matches)
                matches_filter_type.extend(["GT_EQ"] * len(regex_matches))
        # LT-EQ regexes
        for regex in regex_progs_lt_eq:
            regex_matches = list(regex.finditer(string_arg))
            if regex_matches:  # If len(list) > 0
                match_bool = True
                matches.extend(regex_matches)
                matches_filter_type.extend(["LT_EQ"] * len(regex_matches))

        select_string_arg = string_arg
        number = None

        if match_bool:
            longest_match_idx = 0
            matches_lens = [m.end() - m.start() for m in matches]
            longest_match_idx = argmax(matches_lens)
            longest_match = matches[longest_match_idx]
            match_start, match_end = longest_match.start(), longest_match.end()
            select_string_arg = string_arg[0: match_start].strip() + " " + string_arg[match_end:]
            select_string_arg = select_string_arg.strip()
            number = extract_number(string_arg[match_start:match_end])
            filter_type = matches_filter_type[longest_match_idx]

        return match_bool, filter_type, select_string_arg, number

    else:
        raise NotImplementedError



def select_to_filternum(qdmr_node: Node, question: str):
    """ Recursively, convert FILTER(SET, QSPAN) node into
        FILTER_NUM_CONDITION(SET, GET_Q_NUM) if the QSPAN matches any of the pre-defined regexes
        CONDITION is one of LT, GT, LT_EQ, GT_EQ
    """
    blocklist = ["years", "population", "old"]
    question_tokens = tokenize(question)
    change = 0
    if qdmr_node.predicate == "select_passage":
        qspan_string_arg = qdmr_node.string_arg
        match_bool, filter_type, select_string_arg, number = filter_num_classifier(qspan_string_arg, module="select")
        if match_bool and not any([x in question_tokens for x in blocklist]):  # removing over 65 years style questions
            change = 1
            select_node = qdmr_node
            select_node.string_arg = select_string_arg
            filternum_predicate_name = "FILTER_NUM_{}".format(filter_type).lower()
            filternum_node = Node(predicate=filternum_predicate_name, string_arg=number)
            filternum_node.add_child(select_node)
            qdmr_node = filternum_node

    new_children = []
    for child in qdmr_node.children:
        new_child, x = select_to_filternum(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change


def filter_to_filternum(qdmr_node: Node, question: str):
    """ Recursively, convert FILTER(SET, QSPAN) node into
        FILTER_NUM_CONDITION(SET, GET_Q_NUM) if the QSPAN matches any of the pre-defined regexes
        CONDITION is one of LT, GT, LT_EQ, GT_EQ
    """
    change = 0
    if qdmr_node.predicate == "filter_passage":
        qspan_string_arg = qdmr_node.string_arg
        matches, filter_type, number = filter_num_classifier(qspan_string_arg, module="filter")
        if matches:
            change = 1
            filternum_predicate_name = "FILTER_NUM_{}".format(filter_type).lower()
            qdmr_node.predicate = filternum_predicate_name
            qdmr_node.string_arg = number

    new_children = []
    for child in qdmr_node.children:
        new_child, x = filter_to_filternum(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change



min_superlatives = ["shortest", "nearest"]
max_superlatives = ["longest", "farthest"]
football_events = ["touchdown", "field", "interception", "score", "Touchdown", "TD", "touch", "rushing", "catch",
                   "scoring", "return"]
superlative_football_phrases = []
for e in football_events:
    for s in max_superlatives + min_superlatives:
        superlative_football_phrases.append(f"{s} {e}")


def add_required_minmax(qdmr_node: Node, question: str):
    change = 0
    if qdmr_node.predicate == "select_passage":
        match = False
        min_max = None
        matched_phrase = None
        qspan_string_arg = qdmr_node.string_arg
        for phrase in superlative_football_phrases:
            if phrase in qspan_string_arg:
                min_max = "min" if any([x in phrase for x in min_superlatives]) else "max"
                match = True
                matched_phrase = phrase
                break

        if match:
            change = 1
            event_phrase = " ".join(matched_phrase.split(" ")[1:])      # removing superlative
            qspan_string_arg = qspan_string_arg.replace(matched_phrase, event_phrase)
            select_node = qdmr_node
            select_node.string_arg = qspan_string_arg
            min_max_node_predicate = "select_{}_num".format(min_max)
            min_max_node = Node(predicate=min_max_node_predicate)
            min_max_node.add_child(select_node)
            qdmr_node = min_max_node

    new_children = []
    for child in qdmr_node.children:
        new_child, x = add_required_minmax(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)

    return qdmr_node, change




min_football = []
for s in min_superlatives:
    for e in football_events:
        min_football.append(f"{s} {e}")

def remove_vacuous_minmax(qdmr_node: Node, question: str):
    change = 0
    if qdmr_node.predicate == "select_min_num" and any([x in question for x in football_events]) and \
            "shortest" not in question and "first" in question and qdmr_node.children[0].predicate == "select_passage":
        # min predicate; and football question; and shortest not in question
        select_node = qdmr_node.children[0]
        select_string_arg = select_node.string_arg
        if "field goals" in select_string_arg:
            select_string_arg = select_string_arg.replace("field goals", "first field goal")
        if "touchdown" in select_string_arg:
            if "touchdowns" in select_string_arg:
                select_string_arg = select_string_arg.replace("touchdowns", "first touchdown")
            elif "touchdown passes" in select_string_arg:
                select_string_arg = select_string_arg.replace("touchdown passes", "first touchdown pass")
            elif "touchdown runs" in select_string_arg:
                select_string_arg = select_string_arg.replace("touchdown runs", "first touchdown run")
            else:
                select_string_arg = select_string_arg.replace("touchdown", "first touchdown")
        if "possessions" in select_string_arg:
            select_string_arg = select_string_arg.replace("possessions", "first possession")
        if "scored points" in select_string_arg:
            select_string_arg = select_string_arg.replace("scored points", "first scored points")

        select_node.string_arg = select_string_arg
        qdmr_node = select_node
        change = 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = remove_vacuous_minmax(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)
    return qdmr_node, change


def get_cog(attention: List[int]):
    cog = sum([i * x for (i, x) in enumerate(attention)])   # Weighted-COG  -- this seems to work better
    # cog = max([i if x == 1 else 0 for (i, x) in enumerate(attention)])   # Last token attended COG
    return cog


lesser_comparison_tokens = ["shorter", "lower", "less", "fewer", "few"]
def fix_numdiff_arg_order(qdmr_node: Node, question: str):
    change = 0
    program_lisp = nested_expression_to_lisp(qdmr_node.get_nested_expression())
    if program_lisp == "(passagenumber_difference (select_num select_passage) (select_num select_passage))":
        question_tokens = tokenize(question)
        if any([x in question_tokens for x in lesser_comparison_tokens]):
            # We need to ensure that COG of select1 > select2
            select1_node = qdmr_node.children[0].children[0]
            select2_node = qdmr_node.children[1].children[0]
            select_qattn1 = select1_node.supervision["question_attention_supervision"]
            select_qattn2 = select2_node.supervision["question_attention_supervision"]
            cog1 = get_cog(select_qattn1)
            cog2 = get_cog(select_qattn2)

            if cog1 < cog2:
                qdmr_node.children[0].children = []
                qdmr_node.children[0].add_child(select2_node)
                qdmr_node.children[1].children = []
                qdmr_node.children[1].add_child(select1_node)
                change = 1
    return qdmr_node, change


def remove_filter_module(qdmr_node: Node, question: str):
    change = 0
    if qdmr_node.predicate == "filter_passage":
        if qdmr_node.children[0].predicate == "select_passage":
            select_node = qdmr_node.children[0]
            select_node.string_arg += " " + qdmr_node.string_arg
            qdmr_node = select_node
            change = 1
        else:
            # No select after this, completely remove this node.
            qdmr_node = qdmr_node.children[0]
            change = 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = remove_filter_module(child, question)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)
    return qdmr_node, change


def get_postprocessed_dataset(dataset: Dict) -> Dict:
    """ Filter dataset to remove "select_passagespan_answer(select_passage)" questions.
    """
    filtered_data = {}
    total_qa = 0
    num_converted_qa = 0

    qtype_to_function = {
        "filter_to_filternum": filter_to_filternum,
        "select_to_filternum": select_to_filternum,
        "add_required_minmax": add_required_minmax,
        "remove_vacuous_minmax": remove_vacuous_minmax,
        "fix_numdiff_arg_order": fix_numdiff_arg_order,
        "remove_filter_module": remove_filter_module,
    }

    qtype2conversion = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            total_qa += 1
            if constants.program_supervision not in qa:
                continue

            else:
                program_node = node_from_dict(qa[constants.program_supervision])
                post_processed_node = copy.deepcopy(program_node)
                for qtype, processing_function in qtype_to_function.items():
                    post_processed_node, change = processing_function(post_processed_node, question)
                    if change:
                        qtype2conversion[qtype] += 1
                        # if processing_function == remove_filter_module:
                        #     print()
                        #     print(question)
                        #     print(program_node.get_nested_expression_with_strings())
                        #     print(post_processed_node.get_nested_expression_with_strings())

                qa["preprocess_program_supervision"] = program_node.to_dict()
                qa[constants.program_supervision] = post_processed_node.to_dict()


    print()
    print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
    print(f"QType 2 conversion count: {qtype2conversion}")
    return dataset


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json"]

    for filename in FILES_TO_FILTER:
        print(filename)
        input_json = os.path.join(args.input_dir, filename)
        print(f"Input json: {input_json}")

        postprocessed_dataset = get_postprocessed_dataset(dataset=read_drop_dataset(input_json))

        output_json = os.path.join(args.output_dir, filename)
        print(f"OutFile: {output_json}")

        print(f"Writing post-processed data to : {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(postprocessed_dataset, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)

