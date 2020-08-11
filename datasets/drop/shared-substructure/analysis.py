import os
import json
import random
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict
from utils.util import sortDictByValue
from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object
from allennlp.data.tokenizers import SpacyTokenizer

from datasets.drop import constants

random.seed(42)

"""
Write DROP data to TSV with paragraph and question ids to identify common substructures within the same paragraph

"""

spacy_tokenizer = SpacyTokenizer()

nmndrop_language: DropLanguage = get_empty_language_object()


def tokenize(text):
    tokens = spacy_tokenizer.tokenize(text)
    tokens = [t.text for t in tokens]
    tokens = [x for t in tokens for x in t.split("-")]
    return tokens



def get_tsv_strings(drop_dataset: Dict, output_tsv_path: str, num_of_paras: int = 100):
    paraid2questions = {}
    paraids2count = {}

    for paraid, para_info in drop_dataset.items():
        paraid2questions[paraid] = []
        paraids2count[paraid] = 0
        for qa in para_info[constants.qa_pairs]:
            if constants.program_supervision not in qa or not qa[constants.program_supervision]:
                continue
            qid = qa[constants.query_id]
            question = qa[constants.question]
            program_node = node_from_dict(qa[constants.program_supervision])
            program = program_node.get_nested_expression_with_strings()
            paraid2questions[paraid].append((qid, question, program))
            paraids2count[paraid] += 1

    sorted_paraid2count = sortDictByValue(paraids2count, decreasing=True)

    num_q = 0
    para_num = 0
    para_written = 0
    tsv_string = "ParaId\tQuery-Id\tQ-num\tQuestion\tProgram\n"
    while para_written < num_of_paras and para_num < len(paraid2questions):
        paraid = sorted_paraid2count[para_num][0]
        questions = paraid2questions[paraid]
        para_num += 1
        if len(questions) < 2:
            continue
        para_written += 1
        for qnum, (qid, question, program) in enumerate(questions):
            tsv_string += f"{paraid}\t{qid}\t{qnum}\t{question}\t{program}\n"
            num_q += 1
    print("total number of paras / questions: {} / {}".format(num_of_paras, num_q))

    print("writing output to: {}".format(output_tsv_path))
    with open(output_tsv_path, 'w') as f:
        f.write(tsv_string)


def main(args):
    input_json = args.input_json
    input_dir = os.path.split(input_json)[0]
    output_tsv_path = os.path.join(input_dir, args.output_tsv_name)

    print("Readining dataset: {}".format(input_json))
    drop_dataset = read_drop_dataset(input_json)

    print("Accumulating data and writing TSV")  
    get_tsv_strings(drop_dataset=drop_dataset, output_tsv_path=output_tsv_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_tsv_name")

    args = parser.parse_args()

    main(args)

