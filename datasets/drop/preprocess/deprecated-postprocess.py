import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node

from datasets.drop import constants


def remove_filter_module(qdmr_node: Node, question: str):
    change = 0
    if qdmr_node.predicate == "filter_passage":
        if qdmr_node.children[0].predicate == "select_passage":
            select_node = qdmr_node.children[0]
            if select_node.string_arg is not None and qdmr_node.string_arg is not None:
                select_node.string_arg += " " + qdmr_node.string_arg
            select_node.parent = qdmr_node.parent
            filter_supervision_dict: Dict = qdmr_node.supervision
            select_supervision_dict: Dict = select_node.supervision
            filter_qattn = filter_supervision_dict.get("question_attention_supervision", None)
            select_qattn = select_supervision_dict.get("question_attention_supervision", None)
            if filter_qattn is not None:
                if select_qattn is None:
                    select_node.supervision["question_attention_supervision"] = filter_qattn
                else:
                    new_select_qattn = [min(x+y, 1) for x,y in zip(filter_qattn, select_qattn)]
                    select_supervision_dict["question_attention_supervision"] = new_select_qattn

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
    total_qa = 0

    qtype_to_function = {
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
