from typing import List
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    filter_qids_w_logicalforms


count_project_qids = ["93957280-c977-46cf-a2c4-362de4399d4a",
                    "950bdee5-ce90-475f-8bc8-cdefd1817660",
                    "bd01ac93-f741-44d0-bc63-ea17da8d2cfe",
                    "331ac863-2cfe-4187-b14e-c64c36dc633c",
                    "32bacaab-b4c4-4af9-8e69-bb0fcf2f22d2",
                    "7d691a21-5c46-4e91-9af2-4438367754a8",
                    "02254a93-0cac-4576-b192-5d46b9b683e9",
                    "b1514967-60df-485a-bceb-1813f3c19380",
                    "ccaa6255-bc55-4ee9-bc4e-daa657ceddb6",
                    "64918f8e-6a66-4579-984d-b978aee999a6",
                    "8d1d1067-668e-47c9-9c68-4047b59b178b",
                    "1236769b-479b-4adc-bf2c-cbbc25733cc9",
                    "71446d50-bab4-42a8-b8b5-ba3e4cf99035",
                    "8e69e590-5608-4850-8023-a60d5beef76e",
                    "4f0bf230-1adf-4973-bacb-41c3200e06f3",
                    "f59a4f59-01ca-47e6-842a-7f91d81402c4",
                    "7d68bc90-8b3d-4fb4-89a8-6a0f125e1286",
                    "1840100d-60a4-4b48-818a-d8133400c381",
                    "b32c5736-a96f-4669-b4a1-398e7c18fdf6",
                    "4c3efdfd-2782-4b9c-8322-8e11bd3c165b",
                    "da95efc1-0a9d-4232-9af5-68dc6e3e69df",
                    "d7b01be5-e94c-42b1-b83a-5cadc9e86b38",
                    "165de714-e8ba-4aa4-b8cc-329fbb6df82d",
                    "8dbd6237-2843-43d1-8752-6dc71a8382f5",
                    "797e963c-52f2-4f8e-ba70-5a382cda0d85",
                    "bd1b14d6-4264-4c02-837d-f34f227bb7ae",
                    "2bcbb94d-8866-49b9-a964-0ef6e8d998c2",
                    "2684ade2-5687-43a9-b238-c63a8c94d2a4",
                    "303ab451-cf3d-4824-af72-bd8b25f356a1",
                    "0ac9c4ec-9ef4-429b-970a-9af43bdb672b",
                    "989273e3-6d57-46d1-bbac-08932bce8dd2",
                    "aec757e7-1807-4b63-b8de-2ec49edbed44",
                    "cf455cf1-0221-45c3-88ed-b0c768ed82c5",
                    "b61cb679-02a8-4de0-8adf-5a79aa0485f4",
                    "7f3056c5-d276-4b3d-933e-0de7ef9e4edd",
                    "5af72816-96bd-4970-8052-1cc83ba6fda0",
                    "60e0ffa6-d4c1-4c26-aa08-23347dcb847c",
                    "4acd12c4-6394-40a2-a929-2ea6f0a7f340",
                    "c306ad76-1573-4adb-95d6-482d6e8c844a"]




def count_select(preds: List[NMNPredictionInstance], lf):
    selected_instances = []
    for i in preds:
        if i.top_logical_form == lf:
            selected_instances.append(i)
            if lf == "(aggregate_count (project_passage select_passage))":
                print("{} {}".format(i.question, i.f1_score))

    print("Total instances : {}".format(len(selected_instances)))
    if selected_instances:
        print("F1: {}".format(avg_f1(selected_instances)))



old_preds_file = "/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss/drop_parser_bert/Qattn_true/" \
            "EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/" \
            "S_42_PreBIO/predictions/qdmr-v1_dev_predictions.jsonl"

new_preds_file = "/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss-cnt/drop_parser_bert/Qattn_true/" \
            "EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/S_42_PreBIO/" \
            "predictions/qdmr-v1_dev_predictions.jsonl"

old_preds: List[NMNPredictionInstance] = read_nmn_prediction_file(old_preds_file)
new_preds: List[NMNPredictionInstance] = read_nmn_prediction_file(new_preds_file)

selected_old_instances = [i for i in old_preds if i.query_id in count_project_qids]
selected_new_instances = [i for i in new_preds if i.query_id in count_project_qids]

oldf1 = avg_f1(selected_old_instances)
newf1 = avg_f1(selected_new_instances)

print(oldf1)
print(newf1)

count_select(old_preds, "(aggregate_count select_passage)")
count_select(old_preds, "(aggregate_count (project_passage select_passage))")



count_select(new_preds, "(aggregate_count select_passage)")
count_select(new_preds, "(aggregate_count (project_passage select_passage))")


import pdb
pdb.set_trace()
