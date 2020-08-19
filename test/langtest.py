from typing import Dict, List
import json
from semqa.utils.qdmr_utils import node_from_dict, read_drop_dataset, nested_expression_to_lisp, \
    get_inorder_function_list, function_to_action_string_alignment
from datasets.drop import constants
from semqa.models.qgen.constants import SPAN_START_TOKEN, SPAN_END_TOKEN, ALL_SPECIAL_TOKENS
from utils.util import _KnuthMorrisPratt

file_path = "/shared/nitishg/data/squad/squad-train-v1.1_drop.json"

mask_token = "<mask>"

total_q, nonskipped_q = 0, 0
multiple_answers = 0

with open(file_path) as dataset_file:
    dataset = json.load(dataset_file)

for passage_id, passage_info in dataset.items():
    passage = passage_info[constants.passage]

    for qa in passage_info[constants.qa_pairs]:
        total_q += 1
        question_id = qa[constants.query_id]
        question = qa[constants.question]
        question_tokens = qa[constants.question_tokens]
        question_charidxs = qa[constants.question_charidxs]

        answer_dict = qa["answer"]
        answer_text = answer_dict["spans"][0]  # SQuAD only has a single span answer
        if not answer_text:
            continue

        answer_start_charoffsets = list(_KnuthMorrisPratt(passage, answer_text))
        if not answer_start_charoffsets:
            continue

        if len(answer_start_charoffsets) > 1:
            multiple_answers += 1

        program_supervision: Dict = qa.get(constants.program_supervision, None)
        if program_supervision is None:
            continue

        program_node = node_from_dict(program_supervision)
        # Since this is a SQuAD program we know it is a project(select)
        # (start, end) _inclusive_ token indices
        project_start_end_indices = program_node.extras.get("project_start_end_indices", None)
        select_start_end_indices = program_node.extras.get("select_start_end_indices", None)

        if project_start_end_indices is None or select_start_end_indices is None:
            continue

        # Start char-index of the project's start token
        mask_start_charoffset = question_charidxs[project_start_end_indices[0]]
        mask_end_charoffset = question_charidxs[project_start_end_indices[1]] + \
                              len(question_tokens[project_start_end_indices[1]]) - 1

        masked_question = question[0:mask_start_charoffset] + mask_token + \
                          question[mask_end_charoffset + 1:]

        # print(question)
        # print(masked_question)

        nonskipped_q  += 1


print(total_q)
print(nonskipped_q)
print(multiple_answers)