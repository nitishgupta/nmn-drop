from typing import List, Dict, Tuple, Union

import argparse
from collections import defaultdict

from allennlp.data import Token
from allennlp.data.fields import TextField, ListField


from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset
from semqa.data.fields import LabelsField
from semqa.data.dataset_readers.utils import BIOAnswerGenerator

labels = {'O': 0, 'I': 1}
bio_answer_generator = BIOAnswerGenerator(ignore_question=True, flexibility_threshold=1000, labels=labels)

def get_bio_tagging_spans(qa: Dict, spacy_passage_tokens: List[Token], passage_field: TextField):
    answer_annotation: Dict = qa[constants.answer]
    passage_len = len(spacy_passage_tokens)

    # packed_gold_spans_list = Generator for List[Tuple[List[Span]]] -- outer list is the num of possible taggings,
    # where each tagging is a Tuple (size = num of answer-texts) containing a list of spans for each answer-text
    (_, _, _, all_spans, packed_gold_spans_list, spans_dict, has_answer) = bio_answer_generator.get_bio_labels(
        answer_annotation=answer_annotation,
        passage_tokens=spacy_passage_tokens,
        max_passage_len=passage_len,
        passage_field=passage_field)

    if not has_answer or packed_gold_spans_list is None:
        return None

    all_taggings_spans = []     # Contains list of diffent taggings, each tagging is a list of spans
    bio_label_list = []
    for packed_gold_spans in packed_gold_spans_list:
        spans_for_single_tagging: List[Tuple[int, int]] = [s for sublist in packed_gold_spans for s in sublist]
        all_taggings_spans.append(spans_for_single_tagging)
        bio_label = LabelsField(bio_answer_generator._create_sequence_labels(spans_for_single_tagging, passage_len))
        bio_label_list.append(bio_label)

        # print(packed_gold_spans_list)

    # This field would contain all taggings for a given answer_annotation
    bio_labels_field = ListField(bio_label_list)

    return all_taggings_spans, bio_labels_field, spans_dict


def get_spans_count_data(drop_dataset: Dict):
    total_examples, total_num_taggings = 0, 0
    total_num_ans_texts, total_num_grounded_spans = 0, 0

    answertextcount2freq = defaultdict(int)
    spanspertextcount2freq = defaultdict(int)
    spanlen2freq = defaultdict(int)

    for passage_id, passage_info in drop_dataset.items():
        qa_pairs = passage_info[constants.qa_pairs]

        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        passage_charidxs: List[int] = passage_info[constants.passage_charidxs]

        spacy_passage_tokens: List[Token] = [Token(text=t, idx=idx)
                                             for t, idx in zip(passage_tokens, passage_charidxs)]

        passage_field = TextField(tokens=spacy_passage_tokens, token_indexers=None)

        for qa in qa_pairs:
            returns = get_bio_tagging_spans(qa, spacy_passage_tokens, passage_field)
            if returns is None:
                continue

            total_examples += 1

            # List of spans for different possible taggings; choose a subset of these taggings to make the `gold`
            # passage attention and the count value
            all_taggings_spans: List[List[Tuple[int, int]]] = returns[0]
            total_num_taggings += len(all_taggings_spans)

            # This ListField[LabelsField] contains all possible taggings and would be used as
            # weak supervision for marginalization
            bio_labels_field: ListField = returns[1]

            # Dict: answer_text -> List[Tuple[int, int]] -- could be useful to get stats on answer_text and spans
            spans_dict = returns[2]
            total_num_ans_texts += len(spans_dict)
            for _, spans in spans_dict.items():
                total_num_grounded_spans += len(spans)
                spanspertextcount2freq[len(spans)] += 1
                for (start, end) in spans:
                    spanlen2freq[end - start + 1] += 1

            answertextcount2freq[len(spans_dict)] += 1


    avg_taggings_per_example = float(total_num_taggings)/float(total_examples)
    avg_num_ans_text = float(total_num_ans_texts)/float(total_examples)
    avg_groundings_per_ans_text = float(total_num_grounded_spans)/float(total_num_ans_texts)

    print(f"Total Examples: {total_examples}   Total num of taggings: {total_num_taggings}")
    print(f"avg_taggings_per_example: {avg_taggings_per_example}")
    print(f"avg_num_ans_text: {avg_num_ans_text}  avg_groundings_per_ans_text: {avg_groundings_per_ans_text}")
    print(f"answertextcount2freq: {answertextcount2freq}")
    print(f"spanspertextcount2freq: {spanspertextcount2freq}")
    print(f"spanlen2freq : {spanlen2freq}")

    print()


def main(args):
    drop_json = args.drop_json

    drop_json = "/shared/nitishg/data/drop-w-qdmr/preprocess/drop_dataset_train.json"

    drop_dataset = read_drop_dataset(drop_json)

    get_spans_count_data(drop_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_json")
    args = parser.parse_args()

    main(args)







