import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from analysis.qdmr.program_diagnostics import is_potential_filter_num
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, get_postorder_function_list
from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object
from allennlp.data.tokenizers import SpacyTokenizer

from datasets.drop import constants

"""  """

spacy_tokenizer = SpacyTokenizer()

nmndrop_language: DropLanguage = get_empty_language_object()


def tokenize(text):
    tokens = spacy_tokenizer.tokenize(text)
    tokens = [t.text for t in tokens]
    tokens = [x for t in tokens for x in t.split("-")]
    return tokens


min_superlatives = ["shortest", "nearest"]
max_superlatives = ["longest", "farthest"]
football_events = ["touchdown", "field", "interception", "score", "Touchdown", "TD", "touch", "rushing", "catch",
                   "scoring", "return"]
superlative_football_phrases = []
for e in football_events:
    for s in max_superlatives + min_superlatives:
        superlative_football_phrases.append(f"{s} {e}")


def num_minmax_question(program_node: Node, question: str, question_tokens: List[str]):
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    max_lisp = "(select_num (select_max_num select_passage))"
    min_lisp = "(select_num (select_min_num select_passage))"

    if program_lisp == min_lisp:
        prog_type = "min"
    elif program_lisp == max_lisp:
        prog_type = "max"
    else:
        prog_type = None

    if prog_type is None:
        return None

    if prog_type is "min" and "shortest" not in question_tokens:
        return None

    if prog_type is "max" and "longest" not in question_tokens:
        return None

    # Most likely the question should now be of the type "How many yards was the longest rushing touchdown ?"
    # shared_program_lisp is of the opposite superlative as the program_lisp
    if prog_type == "min":
        aux_question = question.replace("shortest", "longest")
        aux_question_tokens = [x if x != "shortest" else "longest" for x in question_tokens]
        aux_program_lisp = max_lisp
        superlative_index = aux_question_tokens.index("longest")
    else:
        aux_question = question.replace("longest", "shortest")
        aux_question_tokens = [x if x != "longest" else "shortest" for x in question_tokens]
        aux_program_lisp = min_lisp
        superlative_index = aux_question_tokens.index("shortest")

    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)

    original_select_node = program_node.children[0].children[0]
    original_ques_attn = original_select_node.supervision.get("question_attention_supervision", None)

    select_node = aux_program_node.children[0].children[0]
    if original_ques_attn is not None:
        # We only switched superlatives; the attention should remain the same
        question_attention_supervision = original_ques_attn
    else:
        # Attend all tokens apart from "How many yards was", "?" and "superlative"
        question_attention_supervision = [0] * len(aux_question_tokens)
        question_attention_supervision[4:-1] = [1] * (len(aux_question_tokens) - 4 - 1)  # `how many yards was` and `?`
        question_attention_supervision[superlative_index] = 0
    select_node.supervision["question_attention_supervision"] = question_attention_supervision

    # We can hard-code this to the index for the `select_passage` for both original and shared program
    origprog_postorder_node_idx = 0
    sharedprog_postorder_node_idx = 0

    return_dict = {
        constants.question: aux_question,
        constants.question_tokens: aux_question_tokens,
        constants.program_supervision: aux_program_node.to_dict(),
        "orig_program_lisp": program_lisp,
        "orig_question": question,
        "origprog_postorder_node_idx": origprog_postorder_node_idx,
        "sharedprog_postorder_node_idx": sharedprog_postorder_node_idx,
    }

    return return_dict


def get_postprocessed_dataset(dataset: Dict, prune_dataset: bool) -> Tuple[Dict, str]:
    """ Filter dataset to remove "select_passagespan_answer(select_passage)" questions.

    Args:
    -----
    prune_dataset: `bool`
        If True, only keep questions for which we augment a shared-substructure question
    """

    filtered_data = {}
    total_qa = 0
    num_substruct_qa = 0

    # Different types of questions for which we can augment shared-substructure questions
    qtype_to_function = {
        "num_minmax_question": num_minmax_question,
    }

    qtype2conversion = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        qa_w_sharedsub = []
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            question_tokens = qa[constants.question_tokens]
            total_qa += 1
            if constants.program_supervision not in qa:
                continue

            else:
                program_node = node_from_dict(qa[constants.program_supervision])

                post_processed_node = copy.deepcopy(program_node)
                for qtype, processing_function in qtype_to_function.items():
                    return_dict = processing_function(post_processed_node, question, question_tokens)
                    if return_dict is not None:
                        qtype2conversion[qtype] += 1
                        # if processing_function == num_minmax_question:
                        #     print()
                        #     print(question)
                        #     print(program_node.to_dict())
                        #     print(return_dict)
                        #     print(get_postorder_function_list(node_from_dict(return_dict[constants.program_supervision])))

                        qa[constants.shared_substructure_annotations] = [return_dict]
                        if prune_dataset:
                            qa_w_sharedsub.append(qa)

                        num_substruct_qa += 1

        if prune_dataset and len(qa_w_sharedsub) > 0:
            passage_info[constants.qa_pairs] = qa_w_sharedsub
            filtered_data[passage_id] = passage_info

    print()
    print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
    print(f"QType 2 conversion count: {qtype2conversion}")

    to_return_dataset = filtered_data if prune_dataset else dataset

    stats = f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}" + "\n"
    stats += f"QType 2 conversion count: {qtype2conversion}" + "\n"

    return to_return_dataset, stats


# def remove_uncompilable_programs(dataset):
#     filtered_data = {}
#     total_qa, num_filtered_qa = 0, 0
#     for passage_id, passage_info in dataset.items():
#         filtered_qas = []
#         for qa in passage_info[constants.qa_pairs]:
#             total_qa += 1
#             if constants.program_supervision not in qa:
#                 continue
#             else:
#                 program_node = node_from_dict(qa[constants.program_supervision])
#                 program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
#                 try:
#                     nmndrop_language.logical_form_to_action_sequence(program_lisp)
#                     filtered_qas.append(qa)
#                     num_filtered_qa += 1
#                 except:
#                     continue
#         if filtered_qas:
#             passage_info[constants.qa_pairs] = filtered_qas
#             filtered_data[passage_id] = passage_info
#
#     print()
#     print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
#     print(f"Number of filtered passages: {len(filtered_data)}\nNumber of input questions: {num_filtered_qa}")
#     return filtered_data


# def add_question_attention_supervision(node: Node, question_lemmas: List[str]) -> Node:
#     if node.string_arg is not None:
#         arg_tokens = spacy_tokenizer.tokenize(node.string_arg)
#         arg_lemmas = []
#         for t in arg_tokens:
#             try:
#                 arg_lemmas.append(t.lemma_)
#             except:
#                 arg_lemmas.append('')
#
#         if "REF" in arg_lemmas:
#             arg_lemmas.remove("REF")
#         if "#" in arg_lemmas:
#             arg_lemmas.remove("#")
#         question_attention: List[int] = [1 if t in arg_lemmas else 0 for t in question_lemmas]
#         node.supervision["question_attention_supervision"] = question_attention
#
#     processed_children = []
#     for child in node.children:
#         processed_children.append(add_question_attention_supervision(child, question_lemmas))
#
#     node.children = []
#     for child in processed_children:
#         node.add_child(child)
#
#     return node
#
#
# def update_question_attention(dataset: Dict):
#     for passage_id, passage_info in dataset.items():
#         for qa in passage_info[constants.qa_pairs]:
#             question = qa[constants.question]
#             question_tokens = qa[constants.question_tokens]
#             question_lemmas = []
#             for t in question_tokens:
#                 tts = spacy_tokenizer.tokenize(t)
#                 if tts:
#                     question_lemmas.append(tts[0].lemma_)
#                 else:
#                     question_lemmas.append('')
#
#             if constants.program_supervision not in qa:
#                 continue
#             else:
#                 program_node = node_from_dict(qa[constants.program_supervision])
#                 program_node = add_question_attention_supervision(program_node, question_lemmas)
#                 qa[constants.program_supervision] = program_node.to_dict()
#
#     return dataset


def main(args):
    input_dir = args.input_dir

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prune_dataset: bool = args.prune_dataset

    FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json"]
    stats = ""

    for filename in FILES_TO_FILTER:
        stats += "Stats for : {}\n".format(filename)

        input_json = os.path.join(input_dir, filename)
        print("Reading data from: {}".format(input_json))
        dataset = read_drop_dataset(input_json)
        dataset_w_sharedsub, file_stats = get_postprocessed_dataset(dataset=dataset, prune_dataset=prune_dataset)

        stats += file_stats + "\n"

        output_json = os.path.join(args.output_dir, filename)
        print(f"OutFile: {output_json}")

        print(f"Writing data w/ shared-substructures to: {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(dataset_w_sharedsub, outf, indent=4)

    stats_file = os.path.join(output_dir, "stats.txt")
    print(f"\nWriting stats to: {stats_file}")
    with open(stats_file, "w") as outf:
        outf.write(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument('--prune-dataset', dest='prune_dataset', action='store_true')

    args = parser.parse_args()

    main(args)

