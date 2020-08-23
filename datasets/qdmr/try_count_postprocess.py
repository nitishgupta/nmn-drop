import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp
from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object
from allennlp.data.tokenizers import SpacyTokenizer

from datasets.drop import constants

""" Trying a simple post-process to convert history count --> project(count). """

nmndrop_language: DropLanguage = get_empty_language_object()


football_events = ["touchdown", "field", "interception", "score", "Touchdown", "TD", "touch", "rushing", "catch",
                   "scoring", "return"]


def get_postprocessed_dataset(dataset: Dict) -> Dict:
    """ Filter dataset to remove "select_passagespan_answer(select_passage)" questions.
    """
    total_qa = 0
    qtype2conversion = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            qid = qa[constants.query_id]
            question_tokens = qa[constants.question_tokens]
            total_qa += 1
            if constants.program_supervision not in qa:
                continue
            else:
                program_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
                if program_lisp != "(aggregate_count select_passage)":
                    continue
                if "history" not in passage_id:
                    continue

                if any([x in question_tokens for x in football_events]):
                    continue

                post_processed_node = copy.deepcopy(program_node)
                select_node = post_processed_node.children[0]
                project_node = Node(predicate="project_passage")
                project_node.add_child(select_node)
                post_processed_node.children = []
                post_processed_node.add_child(project_node)

                print(qid)

                qa["preprocess_program_supervision"] = program_node.to_dict()
                qa[constants.program_supervision] = post_processed_node.to_dict()
                qtype2conversion["count"] += 1

    print()
    print("No questions / programs are removed at this stage")
    print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
    print(f"QType 2 conversion count: {qtype2conversion}")
    return dataset


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json"]
    # FILES_TO_FILTER = ["drop_dataset_dev.json"]

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

