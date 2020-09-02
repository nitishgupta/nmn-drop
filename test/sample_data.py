import json
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, convert_answer

dev1_json = "/shared/nitishg/data/drop/iclr21/iclr_qdmr-v1-noexc/drop_dataset_dev.json"
dev2_json = "/shared/nitishg/data/drop/iclr21/iclr_qdmr-v1/drop_dataset_dev.json"

train_json = "/shared/nitishg/data/drop/iclr21/iclr_qdmr-v1-noexc/drop_dataset_train-FGS-Diff-DC.json"

def read_json_dataset(input_json: str):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


# dev1 = read_json_dataset(dev1_json)
# dev2 = read_json_dataset(dev2_json)

# def get_qids(dataset):
#     qids = set()
#     for paraid, parainfo in dataset.items():
#         for qapair in parainfo["qa_pairs"]:
#             qid = qapair["query_id"]
#             qids.add(qid)
#     return qids

def get_diff_paired(dataset):
    count = 0
    for paraid, parainfo in dataset.items():
        # if "nfl" in paraid:
        #     continue

        for qa in parainfo["qa_pairs"]:
            if "program_supervision" not in qa:
                continue

            program_sup = node_from_dict(qa["program_supervision"])
            if program_sup.predicate != "passagenumber_difference":
                continue

            # print(parainfo["passage"])
            # print("--------------------------------------------------------------------------")
            print(qa["question"])
            print(program_sup.get_nested_expression_with_strings())
            count += 1
            print()

            # if "shared_substructure_annotations" not in qa:
            #     continue
            # if len(qa["shared_substructure_annotations"]) < 2:
            #     continue
            #
            # for paired_qa in qa["shared_substructure_annotations"]:
            #     print(paired_qa["question"])
            #     print(paired_qa["answer"]["number"])

            print()

    print(count)

#
# qids1 = get_qids(dev1)
# qids2 = get_qids(dev2)
# import pdb
# pdb.set_trace()

train_data = read_json_dataset(train_json)
get_diff_paired(train_data)
