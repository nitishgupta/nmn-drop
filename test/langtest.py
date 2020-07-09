from semqa.domain_languages.drop_language import get_empty_language_object
from semqa.utils.qdmr_utils import node_from_dict, read_drop_dataset, nested_expression_to_lisp, \
    get_inorder_function_list, function_to_action_string_alignment
from datasets.drop import constants

drop_dataset_json = "/shared/nitishg/data/drop-w-qdmr/drop_wqdmr_programs/drop_dataset_dev.json"
dataset = read_drop_dataset(drop_dataset_json)
dl = get_empty_language_object()

for pid, pinfo in dataset.items():
    qas = pinfo[constants.qa_pairs]
    for qa in qas:
        question = qa[constants.question]
        program = node_from_dict(qa[constants.program_supervision])
        inorder_func_list = get_inorder_function_list(program)
        nested_exp = program.get_nested_expression()
        print(nested_exp)
        print(inorder_func_list)
        action_strings = dl.logical_form_to_action_sequence(nested_expression_to_lisp(nested_exp))
        print(action_strings)
        func2actionidx = function_to_action_string_alignment(program, action_strings)
        print(func2actionidx)

        print()

        break