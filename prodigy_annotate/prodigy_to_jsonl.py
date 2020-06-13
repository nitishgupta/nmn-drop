from typing import List, Dict
import os
import argparse
import itertools
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, nested_expression_to_lisp, write_jsonl, \
    convert_nestedexpr_to_tuple, convert_answer, read_jsonl

from datasets.drop import constants
from semqa.domain_languages.drop_language_v2 import Date


def convert_to_tsv(jsondict: Dict):
    qid = jsondict["query_id"]
    question = jsondict["question"]
    passage = jsondict["passage"]
    gold_program = jsondict["prodigy_lisp"]
    annotated_program = jsondict["program_annotation"]
    remarks = jsondict.get("remarks", "")       # Key not present if remarks not entered

    header = f"Query Id\tQuestion\tPassage\tGold-program\tProdigy-program\tRemarks"
    outstr = f"{qid}\t{question}\t{passage}\t{gold_program}\t{annotated_program}\t{remarks}"
    return outstr, header


def get_prodigy_annotations(prodigy_annotation_jsonl):
    annotation_dicts: List[Dict] = read_jsonl(args.prodigy_annotation_jsonl)
    # These are the keys --
    # annotation_dict = {
    #     "question": question,
    #     "passage": passage,
    #     "query_id": query_id,
    #     "nested_expr": nested_expr,
    #     "nested_tuple": nested_tuple,
    #     "lisp": lisp,
    #     "prodigy_lisp": prodigy_lisp,
    #     "answer_annotation": answer_annotation,
    #     "answer_list": answers,
    #     "answer_passage_spans": answer_passage_spans,
    #     "passage_number_values": passage_number_values,
    #     "passage_date_values": passage_date_values,
    #     "year_differences": year_differences,
    #     "program": Annotated program from prodigy
    #     "remarks": Remarks annotated from prodigy
    # }

    for annotation_dict in annotation_dicts:
        if "program" in annotation_dict:
            # Change "program" key to "program_annotation"
            annotation_dict["program_annotation"] = annotation_dict.pop("program")

    _, tsv_header = convert_to_tsv(annotation_dicts[0])
    tsv_strings: List[str] = [tsv_header]
    for annotation_dict in annotation_dicts:
        tsv_output, _ = convert_to_tsv(annotation_dict)
        tsv_strings.append(tsv_output)
    tsv_output_string = "\n".join(tsv_strings)

    return annotation_dicts, tsv_output_string


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prodigy_annotation_jsonl")
    args = parser.parse_args()

    prodigy_annotation_tsv = os.path.splitext(args.prodigy_annotation_jsonl)[0] + ".tsv"

    annotation_dicts, tsv_output_string = get_prodigy_annotations(args.prodigy_annotation_jsonl)

    print("TSV writen to: {}".format(prodigy_annotation_tsv))
    with open(prodigy_annotation_tsv, 'w') as outf:
        outf.write(tsv_output_string)

    write_jsonl(args.prodigy_annotation_jsonl, annotation_dicts)
