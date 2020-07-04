import random
import logging
import numpy as np
from typing import List
from overrides import overrides
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import MetadataField, LabelField, ArrayField

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


@DatasetReader.register("passage_attn2count_reader")
class PAttn2CountReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        min_passage_length=200,
        max_passage_length=400,
        min_span_length=5,
        max_span_length=15,
        samples_per_bucket_count=1000,
    ) -> None:
        super().__init__(lazy)

        self._min_passage_length = min_passage_length
        self._max_passage_length = max_passage_length
        self._min_span_length = min_span_length
        self._max_span_length = max_span_length
        self.samples_per_bucket_count = samples_per_bucket_count
        self.num_instances = 0

    @overrides
    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation
        logger.info(
            f"Making training examples with:\n"
            f"min_passage_len: {self._min_passage_length}\n"
            f"max_passage_length: {self._max_passage_length}\n"
            f"min / max span_len: {self._min_span_length} / {self._max_span_length}\n"
        )

        instances: List[Instance] = []

        data_dicts = self.make_data(
            min_passage_length=self._min_passage_length,
            max_passage_length=self._max_passage_length,
            min_span_length=self._min_span_length,
            max_span_length=self._max_span_length,
            max_count_value=7,
            samples_per_bucket_count=self.samples_per_bucket_count,
        )

        for d in data_dicts:
            attention = d["attention"]
            count_value = d["count_value"]
            passage_length = len(attention)

            fields = {}
            fields["passage_attention"] = ArrayField(np.array(attention), padding_value=-1)
            fields["passage_lengths"] = MetadataField(passage_length)
            fields["count_answer"] = LabelField(count_value, skip_indexing=True)

            instances.append(Instance(fields))
            self.num_instances += 1

        print(f"SamplesPerBucketCount: {self.samples_per_bucket_count} TotalInstances: {self.num_instances}")
        print(f"Data made!")

        return instances

    def make_data(
        self,
        min_passage_length,
        max_passage_length,
        min_span_length,
        max_span_length,
        samples_per_bucket_count: int,
        max_count_value: int = 7,
    ):
        # For each 100 length bucket, and count value, generate 1000 examples in train mode, and 100 in val mode
        num_instances_per_bucket_per_count = samples_per_bucket_count

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
                    attention = self.make_instance(
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
                    data_dicts.append({"attention": attention, "count_value": count_value})
                    instances_for_bucket += 1
                print(f"{min_plen}, {max_plen} :: {instances_for_bucket}")
            print("\n")
        return data_dicts

    def sample_spansfor_variablelength(self, seqlen, num_spans, span_lengths: List[int]):
        sum_lengths = sum(span_lengths)
        # We need a gap of atleast 1 token between two spans. Number of heads is computed based on longer spans (+1)
        # and offset is also up by +1
        # Range of Number of possible span starts
        num_heads = seqlen - (sum_lengths - num_spans + num_spans)
        if num_heads < num_spans:
            return None
        indices = range(seqlen - (sum_lengths - num_spans))
        result = []
        offset = 0
        # Randomly sample n=num_spans heads
        for i, idx in enumerate(sorted(random.sample(indices, num_spans))):
            # These heads are 0-indexed, to this we add the offset we've covered in the seq
            idx += offset
            span_length = span_lengths[i]
            result.append((idx, idx + span_length))
            offset += span_length - 1 + 1
        return result

    def make_instance(
        self,
        min_passage_length: int,
        max_passage_length: int,
        min_span_length: int,
        max_span_length: int,
        count_value: int,
    ):

        passage_length = random.randint(min_passage_length, max_passage_length)
        # Mean: 0, Std: 0.2, Size: PassageLength
        attention = np.abs(np.random.normal(0.0, 0.1, passage_length))

        if count_value > 0:
            span_lengths = [random.randint(min_span_length, max_span_length) for _ in range(count_value)]
            # Sample n=count_value spans of the same length. Ends are exclusive
            # sampled_spans = self.sample_spans(passage_length, count_value, span_length)
            sampled_spans = self.sample_spansfor_variablelength(passage_length, count_value, span_lengths)
            if sampled_spans is None:
                return None

            for (start, end) in sampled_spans:
                attention[start:end] += 1.0

        attention_sum = sum(attention)
        attention = attention / attention_sum

        return attention

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
