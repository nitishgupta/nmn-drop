import json
import argparse
from collections import defaultdict

from datasets.drop import constants


def answerTypeAnalysis(input_json: str) -> None:
    """ Perform some analysis on answer types. """

    print("Reading input json: {}".format(input_json))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, "r") as f:
        dataset = json.load(f)

    print("Number of docs: {}".format(len(dataset)))

    anstypes_count = defaultdict(int)
    num_pspan_ans = 0
    num_qspan_ans = 0
    num_bothspan_ans = 0
    num_nospan_ans = 0
    num_qa = 0
    spanans_numspandist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        qa_pairs = passage_info[constants.qa_pairs]
        for qa in qa_pairs:
            num_qa += 1
            ans_type = qa[constants.answer_type]
            anstypes_count[ans_type] += 1

            answer_dict = qa[constants.answer]

            if ans_type == constants.NUM_TYPE:
                print(qa)

            if ans_type == constants.SPAN_TYPE:
                answer_span_texts = answer_dict["spans"]
                spanans_numspandist[len(answer_span_texts)] += 1
                q_ans_spans = qa[constants.answer_question_spans]
                p_ans_spans = qa[constants.answer_passage_spans]
                if q_ans_spans and p_ans_spans:
                    num_bothspan_ans += 1
                elif q_ans_spans:
                    num_qspan_ans += 1
                elif p_ans_spans:
                    num_pspan_ans += 1
                else:
                    num_nospan_ans += 1

    print(f"Num of QA:{num_qa}")
    print(f"Answer Types: {anstypes_count}")
    print("Among span answers:")
    print(f"PSpans: {num_pspan_ans}. QSpans:{num_qspan_ans}, BothSpans:{num_bothspan_ans}, NoSpans:{num_nospan_ans}")
    print(f"SpanType ans - number of correct spans: {spanans_numspandist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    args = parser.parse_args()

    answerTypeAnalysis(input_json=args.input_json)
