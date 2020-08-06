from typing import List, Dict, Tuple, Union

import random
import logging
import numpy as np
from typing import List
from overrides import overrides
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import MetadataField, LabelField, ArrayField, SequenceField, TextField, ListField
from allennlp.data import Token

from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset
from semqa.data.fields import LabelsField
from semqa.data.dataset_readers.utils import extract_answer_info_from_annotation, BIOAnswerGenerator

labels = {'O': 0, 'I': 1}
bio_answer_generator = BIOAnswerGenerator(ignore_question=True, flexibility_threshold=1000, labels=labels)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: Add more number here
WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

Span = Tuple[int, int]

def get_bio_tagging_spans(qa: Dict,
                          spacy_passage_tokens: List[Token],
                          passage_field: TextField) -> Tuple[List[List[Span]], ListField, Dict[str, List[Span]], bool]:
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

    all_taggings_spans = []     # Contains list of different taggings, each tagging is a list of spans
    bio_label_list = []
    for packed_gold_spans in packed_gold_spans_list:
        spans_for_single_tagging: List[Tuple[int, int]] = [s for sublist in packed_gold_spans for s in sublist]
        all_taggings_spans.append(spans_for_single_tagging)
        bio_label = LabelsField(bio_answer_generator._create_sequence_labels(spans_for_single_tagging, passage_len))
        bio_label_list.append(bio_label)

        # print(packed_gold_spans_list)

    # This field would contain all taggings for a given answer_annotation
    bio_labels_field = ListField(bio_label_list)

    return all_taggings_spans, bio_labels_field, spans_dict, has_answer



def get_spans_count_data(drop_dataset: Dict):

    drop_answer_spans_data = []

    total_num_ques, total_has_answer = 0, 0


    for passage_id, passage_info in drop_dataset.items():
        qa_pairs = passage_info[constants.qa_pairs]

        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        passage_charidxs: List[int] = passage_info[constants.passage_charidxs]
        passage_len = len(passage_tokens)

        spacy_passage_tokens: List[Token] = [Token(text=t, idx=idx)
                                             for t, idx in zip(passage_tokens, passage_charidxs)]

        passage_field = TextField(tokens=spacy_passage_tokens, token_indexers=None)

        for qa in qa_pairs:
            total_num_ques += 1
            returns = get_bio_tagging_spans(qa, spacy_passage_tokens, passage_field)
            if returns is None:
                continue

            # List of spans for different possible taggings; choose a subset of these taggings to make the `gold`
            # passage attention and the count value
            all_taggings_spans: List[List[Tuple[int, int]]] = returns[0]

            # This ListField[LabelsField] contains all possible taggings and would be used as
            # weak supervision for marginalization
            bio_labels_field: ListField = returns[1]

            # Dict: answer_text -> List[Tuple[int, int]] -- could be useful to get stats on answer_text and spans
            spans_dict = returns[2]

            has_answer = returns[3]
            total_has_answer += int(has_answer)

            return_tuple = (passage_len, all_taggings_spans, bio_labels_field, has_answer)

            drop_answer_spans_data.append(return_tuple)

    logger.info(f"Num DROP questions: {total_num_ques}  has_answer: {total_has_answer}")
    return drop_answer_spans_data


def get_empty_attention(passage_len):
    return np.abs(np.random.normal(0.0, 0.1, passage_len))


def get_span_attention(start, end):
    return np.abs(np.random.normal(1.0, 0.2, end - start + 1))


@DatasetReader.register("passage_attn2bio_reader")
class PAttn2BioCountReader(DatasetReader):
    def __init__(
        self,
        count_samples_per_bucket_count: int = 100,
        joint_count: bool = True,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)

        self.joint_count = joint_count
        self.count_samples_per_bucket_count = count_samples_per_bucket_count

    @overrides
    def _read(self, file_path: str):
        drop_dataset = read_drop_dataset(file_path)
        spanlen2freq = defaultdict(int)
        count2freq = defaultdict(int)

        drop_answer_spans_data = get_spans_count_data(drop_dataset)
        logger.info("DROP span-answer data for: {}".format(len(drop_answer_spans_data)))

        count2instance = defaultdict(list)
        num_instances = 0
        for (passage_len, all_taggings_spans, bios_label_field, _) in drop_answer_spans_data:
            # all_taggings_spans: List[List[Span]] -- all possible taggings, each being a list of spans
            # bios_label_field: ListField[LabelField] -- BIO labels for all possible taggings for use in marginalization

            # all_taggings_spans contains all possible taggings; we create gold passage_attention
            #  for a small subset of these to mimic actual gold answers as they may appear in the data
            tagging_indices = list(range(len(all_taggings_spans)))
            # Select 30% of tagging sequeces (or atleast 1) from all available
            num_taggings_to_select = max(1, int(0.3 * len(tagging_indices)))
            random.shuffle(tagging_indices)

            selected_tagging_indices = tagging_indices[:num_taggings_to_select]

            bios_label_mask = np.array([1] * len(all_taggings_spans), dtype=np.int32)
            bios_label_mask_field = ArrayField(bios_label_mask, padding_value=0, dtype=np.int32)

            for i in selected_tagging_indices:
                spans = all_taggings_spans[i]
                count_value = len(spans)
                attention = get_empty_attention(passage_len)
                for (start, end) in spans:  # end is _inclusive_
                    attention[start:end+1] += get_span_attention(start, end)
                    spanlen2freq[end - start + 1] += 1
                attention_sum = sum(attention)
                attention = attention / attention_sum
                count2freq[count_value] += 1

                fields = {}
                fields["passage_attention"] = ArrayField(np.array(attention), padding_value=-1)
                fields["all_taggings_spans"] = MetadataField(all_taggings_spans)
                fields["passage_lengths"] = MetadataField(passage_len)
                fields["bios_label_field"] = bios_label_field
                fields["bios_label_mask"] = bios_label_mask_field
                fields["gold_spans"] = MetadataField(spans)
                fields["count_answer"] = LabelField(count_value, skip_indexing=True)

                instance = Instance(fields=fields)
                count2instance[count_value].append(instance)
                num_instances += 1

        if self.joint_count:
            logger.info(f"Instances before adding 0-count instances: {num_instances}")
            num_of_zero_count_instances_to_add = int(0.1 * num_instances)
            for i in range(num_of_zero_count_instances_to_add):
                all_taggings_spans = [[]]
                spans = []
                passage_len = random.randint(300, 500)
                attention = get_empty_attention(passage_len)
                bios_label_field = ListField([bio_answer_generator._get_empty_answer(passage_len)])
                bios_label_mask = np.array([1] * len(all_taggings_spans), dtype=np.int32)
                bios_label_mask_field = ArrayField(bios_label_mask, padding_value=0, dtype=np.int32)
                count_value = 0

                fields = {}
                fields["passage_attention"] = ArrayField(np.array(attention), padding_value=-1)
                fields["all_taggings_spans"] = MetadataField(all_taggings_spans)
                fields["passage_lengths"] = MetadataField(passage_len)
                fields["bios_label_field"] = bios_label_field
                fields["bios_label_mask"] = bios_label_mask_field
                fields["gold_spans"] = MetadataField(spans)
                fields["count_answer"] = LabelField(count_value, skip_indexing=True)

                spanlen2freq[0] += 1
                count2freq[count_value] += 1
                instance = Instance(fields=fields)
                count2instance[count_value].append(instance)
                num_instances += 1

            logger.info(f"Instances after adding 0-count instances: {num_instances}")
            logger.info(f"spanlen2freq: {spanlen2freq}")
            logger.info(f"count2freq: {count2freq}")

        instances: List[Instance] = []
        num_instances = 0
        count2freq = {}
        for count, count_instances in count2instance.items():
            random.shuffle(count_instances)
            count_instances = count_instances[0:4000]
            num_instances += len(count_instances)
            instances.extend(count_instances)
            count2freq[count] = len(count_instances)

        logger.info(f"Selected instances after pruning: {len(instances)}")
        logger.info(f"count2freq: {count2freq}")

        if self.joint_count:
            logger.info("Adding synthetic count instances")
            syn_count_instances = self.make_instances_from_count(self.count_samples_per_bucket_count)
            instances.extend(syn_count_instances)
            logger.info(f"Instances after adding synthetic count instances: {len(instances)}")
            logger.info("Total num of instances made: {}".format(len(instances)))

        return instances


    def make_instances_from_count(self, count_samples_per_bucket_count) -> List[Instance]:
        count_data_dicts: List[Dict] = self.make_count_data(count_samples_per_bucket_count)

        logger.info(f"Total synthetic count instances: {len(count_data_dicts)}")

        instances = []
        num_bio_instances = 0

        for count_example in count_data_dicts:
            passage_attention = count_example["attention"]
            count_value = count_example["count_value"]
            spans = count_example["spans"]
            passage_len = len(passage_attention)

            if random.random() < 0.3 and count_value > 0:
                # Make BIO instance from 20% of synthetic count data
                all_taggings_spans = [spans]    # Only one possible tagging
                bios: List[int] = bio_answer_generator._create_sequence_labels(spans, passage_len)
                bios_label_field = ListField([LabelsField(bios)])
                bios_label_mask = np.array([1] * len(all_taggings_spans), dtype=np.int32)
                bios_label_mask_field = ArrayField(bios_label_mask, padding_value=0, dtype=np.int32)
                num_bio_instances += 1
            else:
                # Don't make BIO instance from this; create a masked instance
                all_taggings_spans = [[]]
                bios_label_field = ListField([bio_answer_generator._get_empty_answer(passage_len)])
                bios_label_mask = np.array([0] * len(all_taggings_spans), dtype=np.int32)
                bios_label_mask_field = ArrayField(bios_label_mask, padding_value=0, dtype=np.int32)

            fields = {}
            fields["passage_attention"] = ArrayField(np.array(passage_attention), padding_value=-1)
            fields["all_taggings_spans"] = MetadataField(all_taggings_spans)
            fields["passage_lengths"] = MetadataField(passage_len)
            fields["bios_label_field"] = bios_label_field
            fields["bios_label_mask"] = bios_label_mask_field
            fields["gold_spans"] = MetadataField(spans)
            fields["count_answer"] = LabelField(count_value, skip_indexing=True)

            instances.append(Instance(fields=fields))

        logger.info(f"Num of BIO instances within synthetic count: {num_bio_instances}")

        return instances


    def make_count_data(
        self,
        count_samples_per_bucket_count: int,
        min_passage_length=300,
        max_passage_length=600,
        min_span_length=1,
        max_span_length=10,
        max_count_value: int = 8,
    ):
        # For each 100 length bucket, and count value, generate 1000 examples in train mode, and 100 in val mode
        num_instances_per_bucket_per_count = count_samples_per_bucket_count

        # List of min and max passage
        minmax_passagelen_tuples = self._get_length_buckets(min_passage_length, max_passage_length)

        data_dicts = []

        print(f"Making Data ... ")

        lenbucket_count_dict = defaultdict()
        print(f"Passage Length Buckets: {minmax_passagelen_tuples}")

        for count_value in range(0, max_count_value + 1):
            print(f"Count Value: {count_value}")
            for min_plen, max_plen in minmax_passagelen_tuples:
                instances_for_bucket = 0
                for i in range(num_instances_per_bucket_per_count):
                    spans, attention = self.make_count_instance(
                        min_passage_length=min_plen,
                        max_passage_length=max_plen,
                        min_span_length=min_span_length,
                        max_span_length=max_span_length,
                        count_value=count_value,
                    )
                    if attention is None:
                        continue
                    if count_value not in lenbucket_count_dict:
                        lenbucket_count_dict[count_value] = defaultdict(int)
                    lenbucket_count_dict[count_value][(min_plen, max_plen)] += 1
                    data_dicts.append({"attention": attention, "spans": spans, "count_value": count_value})
                    instances_for_bucket += 1
                print(f"{min_plen}, {max_plen} :: {instances_for_bucket}")
            print("\n")

        return data_dicts


    def make_count_instance(
        self,
        min_passage_length: int,
        max_passage_length: int,
        min_span_length: int,
        max_span_length: int,
        count_value: int,
    ):

        passage_length = random.randint(min_passage_length, max_passage_length)
        # Mean: 0, Std: 0.2, Size: PassageLength
        attention = get_empty_attention(passage_length)

        if count_value > 0:
            span_lengths = [random.randint(min_span_length, max_span_length) for _ in range(count_value)]
            # Sample n=count_value spans of the same length. Ends are exclusive
            # sampled_spans = self.sample_spans(passage_length, count_value, span_length)
            sampled_spans = self.sample_spansfor_variablelength(passage_length, count_value, span_lengths)
            if sampled_spans is None:
                return None, None
            sampled_spans = [(x, y-1) for (x, y) in sampled_spans]   # making end _inclusive_

            fixed_spans = []
            for (start, end) in sampled_spans:
                if end >= len(attention):
                    end = max(start, len(attention) - 1)
                fixed_spans.append((start, end))

            for (start, end) in fixed_spans:  # end is _exclusive_
                attention[start:end + 1] += get_span_attention(start, end)
        else:
            fixed_spans = []


        attention_sum = sum(attention)
        attention = attention / attention_sum

        return fixed_spans, attention

    def sample_spansfor_variablelength(self, seqlen, num_spans, span_lengths: List[int]):
        def isSpanOverlap(s1, s2):
            start1, end1 = s1[0], s1[1]
            start2, end2 = s2[0], s2[1]
            return max(start1, start2) <= min(end1, end2)

        sum_lengths = sum(span_lengths)
        num_heads = seqlen - (sum_lengths + len(span_lengths) - 1)

        if num_heads < 0:
            return None

        res = set()
        for _, spanlen in enumerate(span_lengths):
            s = random.randint(0, seqlen - spanlen)
            e = s + spanlen - 1
            while any(isSpanOverlap((s, e), span) for span in res):
                s = random.randint(0, seqlen - spanlen)
                e = s + spanlen - 1
            res.add((s, e))

        return res

    def _get_length_buckets(self, min_passage_length, max_passage_length):
        min_length_buckets = [min_passage_length]
        max_length_buckets = []

        # Add start, end + 100 until end <= max_passage_length
        i = 1
        while True:
            potential_max_len = i * 100 + min_passage_length
            if potential_max_len <= max_passage_length:
                max_length_buckets.append(potential_max_len)
                min_length_buckets.append(max_length_buckets[-1])  # Last end is next's start

                i += 1
            else:
                break
        if len(max_length_buckets) == 0 or max_length_buckets[-1] != max_passage_length:  # This was left out
            max_length_buckets.append(max_passage_length)

        if min_length_buckets[-1] == max_passage_length:
            min_length_buckets = min_length_buckets[:-1]

        return list(zip(min_length_buckets, max_length_buckets))
