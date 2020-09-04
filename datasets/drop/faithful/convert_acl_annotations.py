from typing import List, Dict, Tuple
import json
import argparse

from semqa.domain_languages.drop_language import Output
from semqa.utils.qdmr_utils import node_from_dict, lisp_to_nested_expression, nested_expression_to_tree, \
    read_json_dataset



def convert_annotations(dataset):
    """Convert ACL annotations to newer format of using Node for programs and drop_language.Output for module outputs.

    In ACL annotations, each qa_pair contains an additional key, "module_output_annotations" which is a List containing
    ("module_name", List[Span]) tuples. Each span is token-offset tuple (start, end) with exclusive-end.
    The key "qtype": str contains the question-type that needs to be converted to a lisp -> Node program.

    Example, qtype = "max_find_qtype" --> (select_number (select_max_num select_passage))
    The annotations for this program are "find-events" (select_passage), "find-nums" (number_input in select_max_num),
    and "after-max-event"
    """









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl_json")
    parser.add_argument("--drop_dataset")
    parser.add_argument("--output_json")
    args = parser.parse_args()



