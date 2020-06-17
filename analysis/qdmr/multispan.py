from semqa.utils.qdmr_utils import read_drop_dataset, convert_answer
from datasets.drop import constants


def is_ans_multispan(qa):
    answer_annotation = qa[constants.answer]
    ans_type, answerlist = convert_answer(answer_annotation)

    ans_multispan = False
    ans_span = False
    if ans_type == "spans":
        ans_span = True
        if len(answerlist) > 1:
            ans_multispan = True
    return ans_span, ans_multispan


def num_multispan_ans_in_dataset(drop_json="/shared/nitishg/data/drop-w-qdmr/qdmr-filter-v2/drop_dataset_train.json"):
    dataset = read_drop_dataset(drop_json)

    total, num_spanans, num_multispan = 0, 0, 0
    for pid, pinfo in dataset.items():
        for qa in pinfo[constants.qa_pairs]:
            total += 1
            ans_span, ans_multispan = is_ans_multispan(qa)
            num_spanans += int(ans_span)
            num_multispan += int(ans_multispan)

    print("Total: {} SpanAns: {}  multi-span ans: {}".format(total, num_spanans, num_multispan))


num_multispan_ans_in_dataset()



