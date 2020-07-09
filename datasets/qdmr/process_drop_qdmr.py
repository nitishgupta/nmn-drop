from typing import List, Dict, Tuple, Union
import json
import os
import argparse

from datasets.drop import constants

from semqa.utils.qdmr_utils import Node, QDMRExample, nested_expression_to_lisp, nested_expression_to_tree, \
    read_qdmr_json_to_examples, read_drop_dataset, convert_answer, convert_nestedexpr_to_tuple, node_from_dict, \
    get_domainlang_function2returntype_mapping

from allennlp.data.tokenizers import SpacyTokenizer, Token

from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object

nmndrop_language: DropLanguage = get_empty_language_object()

function2returntype_mapping = get_domainlang_function2returntype_mapping(nmndrop_language)


"""
Takes drop-programs produced by (github.com/nitishgupta/qdmr/blob/master/parse_dataset/drop_grammar_program.py) which 
are parsable by the language here: github.com/nitishgupta/qdmr/blob/master/qdmr/domain_languages/drop_language.py

This script converts them into programs that can be compiled by the DROPLanguageV2 described in this codebase:
(semqa.domain_languages.drop_language_v2.DropLanguageV2)

Additionally, we would like to add these programs as supervision to pre-processed DROP data.

There are few things that definitely need to be taken care of -- 
    1. QDMR-DROP language contains GET_QUESTION_SPAN which is implicit here (in the question-attention version at least)
        This node would need to be removed and its string_arg would need to be transferred to its parent node.
    
    2. Programs whose root-node output a passage-span (passage-attention) will need to be wrapped with predicate 
        `select_passagespan_answer` to output the logits for span start/end.
    
    3. mapping from predicates in QDMR-DROP language to the language here 
     
"""

# Mapping from predicates in QDMR-DROP language (in break repo) to the predicates in DROP language in NMN (this repo)
QDMR_DROP_to_NMN_DROP_predicate_mapping = {
    'AGGREGATE_avg': None,
    'AGGREGATE_count': "aggregate_count",
    'AGGREGATE_max': "select_max_num",
    'AGGREGATE_min': "select_min_num",
    'AGGREGATE_sum': None,
    'ARITHMETIC_difference': "passagenumber_difference",    # TODO(nitish): This is not correct. e.g. diff(count, count)
    'ARITHMETIC_divison': None,
    'ARITHMETIC_multiplication': None,
    'ARITHMETIC_sum': "passagenumber_addition",             # TODO(nitish): This is not correct. e.g. add(count, count)
    'BOOLEAN': None,
    'COMPARATIVE': None,
    'COMPARISON_DATE_max': "compare_date_gt",
    'COMPARISON_DATE_min': "compare_date_lt",
    'COMPARISON_count_max': None,
    'COMPARISON_count_min': None,
    'COMPARISON_max': "compare_num_gt",
    'COMPARISON_min': "compare_num_lt",
    'COMPARISON_sum_max': None,
    'COMPARISON_sum_min': None,
    'COMPARISON_true': None,
    'CONDITION': None,
    'DISCARD': None,
    'FILTER': "filter_passage",
    # 'FILTER_NUM_EQ': "filter_num_eq",
    # 'FILTER_NUM_GT': "filter_num_gt",
    # 'FILTER_NUM_GT_EQ': "filter_num_gt_eq",
    # 'FILTER_NUM_LT': "filter_num_lt",
    # 'FILTER_NUM_LT_EQ': "filter_num_lt_eq",
    # 'GET_QUESTION_NUMBER': "get_question_number",
    'GET_QUESTION_SPAN': None,
    'GROUP_count': None,
    'GROUP_sum': None,
    'INTERSECTION': None,
    'PARTIAL_GROUP_count': None,
    'PARTIAL_GROUP_sum': None,
    'PARTIAL_PROJECT': None,
    'PARTIAL_SELECT_NUM': None,
    'PARTIAL_SELECT_SINGLE_NUM': None,
    'PROJECT': "project_passage",
    'SELECT': "select_passage",
    'SELECT_IMPLICIT_NUM': "select_implicit_num",
    'SELECT_NUM': "select_num",
    'SELECT_NUM_SPAN': None,
    'SUPERLATIVE_max': None,
    'SUPERLATIVE_min': None,
    'UNION': None,
    'Year_Diff_Single_Event': "year_difference_single_event",
    'Year_Diff_Two_Events': "year_difference_two_events",
}

qdmr_predicates_with_passagespan_output = {"AGGREGATE_max", "AGGREGATE_min", "FILTER", "COMPARISON_DATE_max",
                                           "COMPARISON_DATE_min", "COMPARISON_count_max", "COMPARISON_count_min",
                                           "COMPARISON_max", "COMPARISON_min", "COMPARISON_sum_max",
                                           "COMPARISON_sum_min", "DISCARD", "INTERSECTION", "UNION", "SUPERLATIVE_max",
                                           "SUPERLATIVE_min", "FILTER_NUM_EQ", "FILTER_NUM_GT", "FILTER_NUM_GT_EQ",
                                           "FILTER_NUM_LT", "FILTER_NUM_LT_EQ", "COMPARATIVE", "SELECT", "PROJECT"}


nmndrop_predicates_with_passageattention_output = {"compare_date_gt", "compare_date_lt", "compare_num_gt",
                                                   "compare_num_lt", "filter_num_eq", "filter_num_gt",
                                                   "filter_num_gt_eq", "filter_num_lt", "filter_num_lt_eq",
                                                   "filter_passage", "project_passage", "select_max_num",
                                                   "select_min_num", "select_passage"}


# this is the outermost predicate in nmn-drop language to convert passage-attention to span start/end logits
NMN_DROP_select_passagespan_answer_predicate = "select_passagespan_answer"

QDMR_DROP_get_question_span_predicate = "GET_QUESTION_SPAN"

spacy_tokenizer = SpacyTokenizer()


def tokenize(text: str) -> List[str]:
    tokens: List[Token] = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def add_question_attention_supervision(node: Node, question_tokens: List[str]) -> Node:
    if node.string_arg is not None:
        arg_tokens = set(tokenize(node.string_arg))
        if "REF" in arg_tokens:
            arg_tokens.remove("REF")
        if "#" in arg_tokens:
            arg_tokens.remove("#")
        question_attention: List[int] = [1 if t in arg_tokens else 0 for t in question_tokens]
        node.supervision["question_attention_supervision"] = question_attention

    processed_children = []
    for child in node.children:
        processed_children.append(add_question_attention_supervision(child, question_tokens))

    node.children = []
    for child in processed_children:
        node.add_child(child)

    return node


def get_cog(attention: List[int]):
    cog = sum([i * x for (i, x) in enumerate(attention)])   # Weighted-COG  -- this seems to work better
    # cog = max([i if x == 1 else 0 for (i, x) in enumerate(attention)])   # Last token attended COG
    return cog


def switch_year_diff_two_events_arguments(question: str, program_node: Node, count_dict: Dict):
    question = question.lower()
    nested_expr = program_node.get_nested_expression()
    nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)
    if nested_expr_tuple == ('year_difference_two_events', 'select_passage', 'select_passage'):
        count_dict["year-two-diff"] = count_dict.get("year-two-diff", 0) + 1
        select1_node = program_node.children[0]
        select2_node = program_node.children[1]
        qattn1 = select1_node.supervision["question_attention_supervision"]
        qattn2 = select2_node.supervision["question_attention_supervision"]
        cog1 = get_cog(qattn1)
        cog2 = get_cog(qattn2)
        if any([x in question for x in ["how many years passed",
                                        "how many years were between",
                                        "how many years was it"]]):
            count_dict["years-passed-between"] = count_dict.get("years-passed-between", 0) + 1
            # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
            if cog1 < cog2:
                # Switch select-node order in this case
                program_node.children = []
                program_node.add_child(select2_node)
                program_node.add_child(select1_node)
                count_dict["switched"] = count_dict.get("switched", 0) + 1
        elif "after" in question:
            count_dict["year-two-diff-after"] = count_dict.get("year-two-diff-after", 0) + 1
            if question[0:3] == "how":
                # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
                if cog1 < cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
            else:
                # We want cog1 < cog2, i.e., the first select1 to select the event mentioned earlier
                if cog1 > cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
        elif "how many years before" in question:
            count_dict["year-two-diff-before"] = count_dict.get("year-two-diff-before", 0) + 1
            if question[0:3] == "how":
                # We want cog1 <>> cog2, i.e., the first select1 to select the event mentioned earlier
                if cog1 > cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
            else:
                # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
                if cog1 < cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
        else:
            count_dict["remaining"] = count_dict.get("remaining", 0) + 1
            # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
            if cog1 < cog2:
                # Switch select-node order in this case
                program_node.children = []
                program_node.add_child(select2_node)
                program_node.add_child(select1_node)
                count_dict["switched"] = count_dict.get("switched", 0) + 1
    return program_node


def remove_get_question_span_node(qdmr_node: Node) -> Node:
    """This function removes the GET_QUESTION_SPAN node from the program and makes the string_arg of this predicate
    as the string_arg of the parent node. This is done since NMN-DROP (question-attention version) does not have an
    explicit predicate to select discrete question span as input to a module. Instead the node which needs
    question-string-arg uses the question attention from decoding at that time-step.
    The now resulting string-arg in a predicate provides guidance to the predicate for what kind of question-attention
    it should aim for.

    E.g.
    ['FILTER_NUM_GT', ['SELECT', 'GET_QUESTION_SPAN(field goals that Adam Vinatieri kick)'], 'GET_QUESTION_NUMBER(40)']
    becomes
    ['FILTER_NUM_GT', 'SELECT(field goals that Adam Vinatieri kick)', 'GET_QUESTION_NUMBER(40)']
    """
    # First remove get_question_span node as a child
    children = qdmr_node.children
    pruned_children = []
    for child in children:
        if child.predicate == QDMR_DROP_get_question_span_predicate:
            qdmr_node.string_arg = child.string_arg
        else:
            pruned_children.append(child)
    # Recursively remove get_question_span node from all children's subtree
    pruned_children = [remove_get_question_span_node(c) for c in pruned_children]
    qdmr_node.children = []
    for child in pruned_children:
        qdmr_node.add_child(child)
    return qdmr_node


def wrap_passage_set_root_w_span_ans_predicate(nmn_drop_program: Node) -> Node:
    """If the QDMR-DROP program returns Set[Passage] then wrap it with NMN-DROP's select_passagespan_ans predicate.

    This predicate is needed to convert passage-attention to logits for span start/end as output
    """
    if nmn_drop_program.predicate in nmndrop_predicates_with_passageattention_output:
        select_spanans_node = Node(predicate=NMN_DROP_select_passagespan_answer_predicate)
        select_spanans_node.add_child(nmn_drop_program)
        nmn_drop_program = select_spanans_node
    return nmn_drop_program


def map_qdmr_to_nmn_predicates(qdmr_program: Node) -> Tuple[Union[Node, None], bool]:
    """Converts a QDMR-DROP program into NMN-DROP program by mapping predicates from QDMR-language to NMN-language."""
    if qdmr_program.predicate not in QDMR_DROP_to_NMN_DROP_predicate_mapping:
        print("QDMR-DROP predicate not found: {}".format(qdmr_program.predicate))

    mapping_present = True
    nmn_drop_predicate = QDMR_DROP_to_NMN_DROP_predicate_mapping.get(qdmr_program.predicate, None)
    if nmn_drop_predicate is None:
        mapping_present = False
        return None, mapping_present

    qdmr_program.predicate = nmn_drop_predicate
    new_children = []
    children_mapping_present = True
    for c in qdmr_program.children:
        new_child, child_mapping_present = map_qdmr_to_nmn_predicates(c)
        children_mapping_present = children_mapping_present and child_mapping_present
        new_children.append(new_child)

    qdmr_program.children = []
    if children_mapping_present:
        for c in new_children:
            qdmr_program.add_child(c)

    mapping_present = mapping_present and children_mapping_present
    if mapping_present:
        return qdmr_program, mapping_present
    else:
        return None, mapping_present


def convert_qdmr_program_to_nmn(qdmr_program: Node,
                                question: str,
                                drop_question_tokens: List[str]) -> Tuple[Union[Node, None], bool, bool]:
    """Convert QDMR-DROP program into NMN-DROP program.

    Does it in two steps --
    (a) removes get_question_span node from qdmr-program and adds its output as the parent node's string_arg,
    (b) map qdmr-predicates to nmn-predicates,
    (c) wrap passage-span producing programs by module that converts passage-attention into passage-span logits,

    Finally checks if the nmn-program is executable -- it might not be executable since the mapping is incorrect or
    we need to add modules in nmn-language that type-check with qdmr-language.

    """

    qdmr_program = remove_get_question_span_node(qdmr_program)

    mapping_present = True
    executable = True
    # mapping_present is False if for any node a mapping does not exist
    nmn_drop_program, mapping_present = map_qdmr_to_nmn_predicates(qdmr_program)
    if not mapping_present:
        executable = False
        return None, mapping_present, executable

    # wrap program in NMN's passage-attention to passage-span-ans module
    nmn_drop_program: Node = wrap_passage_set_root_w_span_ans_predicate(nmn_drop_program)

    nmn_drop_prog_lisp = nested_expression_to_lisp(nmn_drop_program.get_nested_expression())
    try:
        nmndrop_language.logical_form_to_action_sequence(nmn_drop_prog_lisp)
    except:
        executable = False
        print(f"{nmn_drop_program.get_nested_expression_with_strings()}  --  non-executable")

    if executable:
        nmn_drop_program = add_question_attention_supervision(nmn_drop_program, drop_question_tokens)
        switch_year_diff_two_events_arguments(question=question, program_node=nmn_drop_program, count_dict={})

    return nmn_drop_program, mapping_present, executable


def qdmrid_to_dropid(qdmr_query_id) -> Tuple[str, str]:
    """Convert QDMR question_id into DROP's passage_id and query_id. """
    # E.g. QDMR id -- DROP_train_history_1_2a9a05d2-6fb0-4e99-8751-cd3199b0a80f
    qdmr_splits = qdmr_query_id.split("_")
    split = qdmr_splits[1]
    passage_type, passage_num = qdmr_splits[2], qdmr_splits[3]
    passage_id = f"{passage_type}_{passage_num}"
    query_id = qdmr_splits[4]
    return passage_id, query_id


def get_drop_question_and_tokens(drop_data, passage_id, query_id) -> Tuple[str, List[str]]:
    passage_info = drop_data[passage_id]
    for qa in passage_info[constants.qa_pairs]:
        if qa[constants.query_id] == query_id:
            return qa[constants.question], qa[constants.question_tokens]


def get_drop_programs(qdmr_examples: List[QDMRExample], drop_data) -> Dict[str, Dict[str, Node]]:
    num_w_qdmr_drop, num_w_nmn_drop, num_w_nmn_executable = 0, 0, 0

    drop_passageid2qid2program = {}

    for qdmr_example in qdmr_examples:
        program_node = qdmr_example.program_tree
        qdmr_query_id = qdmr_example.query_id
        drop_passage_id, drop_query_id = qdmrid_to_dropid(qdmr_query_id)
        drop_question, drop_question_tokens = get_drop_question_and_tokens(drop_data, drop_passage_id, drop_query_id)

        if program_node:
            num_w_qdmr_drop += 1
            nmn_drop_prog, mapping_present, executable = convert_qdmr_program_to_nmn(program_node,
                                                                                     drop_question,
                                                                                     drop_question_tokens)
            num_w_nmn_drop += 1 if mapping_present else 0
            num_w_nmn_executable += 1 if executable else 0

            if executable:
                if drop_passage_id not in drop_passageid2qid2program:
                    drop_passageid2qid2program[drop_passage_id] = {}
                drop_passageid2qid2program[drop_passage_id][drop_query_id] = nmn_drop_prog

    print("Total ques with qdmr-drop programs: {}".format(num_w_qdmr_drop))
    print("Total ques with nmn-drop programs: {}".format(num_w_nmn_drop))
    print("Total ques with nmn-drop executable program: {}".format(num_w_nmn_executable))

    return drop_passageid2qid2program


def prune_dropdata_wprograms(drop_data, drop_passageid2qid2program):
    """Prune drop data to questions that contain executable nmn-program supervision."""
    pruned_drop_data = {}
    for passage_id, qid2program in drop_passageid2qid2program.items():
        passage_info = drop_data[passage_id]
        new_qa_pairs = []
        qa_pairs = passage_info[constants.qa_pairs]
        for qa in qa_pairs:
            query_id = qa[constants.query_id]
            if query_id not in qid2program:
                continue
            program_supervision: Node = qid2program[query_id]
            root_predicate = program_supervision.predicate
            qa[constants.program_supervision] = program_supervision.to_dict()
            new_qa_pairs.append(qa)
        passage_info[constants.qa_pairs] = new_qa_pairs
        pruned_drop_data[passage_id] = passage_info
    return pruned_drop_data


def main(args):
    qdmr_json_path = args.qdmr_json
    qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(qdmr_json_path)
    drop_data = read_drop_dataset(args.drop_json)

    drop_passageid2qid2program = get_drop_programs(qdmr_examples, drop_data)

    pruned_drop_data = prune_dropdata_wprograms(drop_data, drop_passageid2qid2program)
    print(f"Total passages: {len(pruned_drop_data)}")

    output_json = args.output_json
    output_dir = os.path.split(output_json)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Writing drop data with program-supervision: {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(pruned_drop_data, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdmr_json")
    parser.add_argument("--drop_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)
