from typing import List, Dict
import json
from collections import defaultdict


def read_demo_ouput(jsonl_filepath):
    with open(jsonl_filepath) as f:
        for line in f:
            demo_output = json.loads(line)
            question = demo_output["question"]
            program_lisp = demo_output["program_lisp"]
            program_nested_expression = demo_output["program_nested_expression"]

            print("question: {}".format(question))
            print("program lisp: {}".format(program_lisp))
            print("program_nested_expression: {}".format(program_nested_expression))
            program_execution = demo_output["program_execution"]
            print("program_execution")
            for module_dict in program_execution:
                for module_name, module_output_dict in module_dict.items():
                    print("  {}".format(module_name))
                    for output_type, attention in module_output_dict.items():
                        print("    {}".format(output_type))



if __name__=="__main__":
    predictions_jsonl = "test/demo_output.jsonl"

    read_demo_ouput(predictions_jsonl)