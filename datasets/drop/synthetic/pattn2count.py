from typing import List
import numpy as np
import random
from collections import defaultdict
import json

random.seed(100)
np.random.seed(100)


def sample_spansfor_variablelength(seqlen, num_spans, span_lengths: List[int]):
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


def make_instance(min_passage_length: int, max_passage_length: int,
                  min_span_length: int, max_span_length: int, count_value: int):

    passage_length = random.randint(min_passage_length, max_passage_length)
    # Mean: 0, Std: 0.2, Size: PassageLength
    attention = np.abs(np.random.normal(0.0, 0.1, passage_length))

    if count_value > 0:
        span_lengths = [random.randint(min_span_length, max_span_length) for _ in range(count_value)]
        # Sample n=count_value spans of the same length. Ends are exclusive
        # sampled_spans = self.sample_spans(passage_length, count_value, span_length)
        sampled_spans = sample_spansfor_variablelength(passage_length, count_value, span_lengths)
        if sampled_spans is None:
            return None

        for (start, end) in sampled_spans:
            attention[start:end] += 1.0

    attention_sum = sum(attention)
    attention = attention / attention_sum

    return attention

def _get_length_buckets(min_passage_length, max_passage_length):
    if min_passage_length == max_passage_length:
        return [(min_passage_length, max_passage_length)]

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


def make_data(min_passage_length, max_passage_length, min_span_length, max_span_length,
              samples_per_bucket_count: int, max_count_value: int = 7):
    # For each 100 length bucket, and count value, generate 1000 examples in train mode, and 100 in val mode
    num_instances_per_bucket_per_count = samples_per_bucket_count

    # List of min and max passage
    minmax_passagelen_tuples = _get_length_buckets(min_passage_length, max_passage_length)
    data_dicts = []

    lenbucket_count_dict = defaultdict()

    for count_value in range(0, max_count_value + 1):
        print(f"Count Value: {count_value}")
        for min_plen, max_plen in minmax_passagelen_tuples:
            instances_for_bucket = 0
            for i in range(num_instances_per_bucket_per_count):
                attention = make_instance(min_passage_length=min_plen, max_passage_length=max_plen,
                                          min_span_length=min_span_length, max_span_length=max_span_length,
                                          count_value=count_value)
                if attention is None:
                    continue
                if count_value not in lenbucket_count_dict:
                    lenbucket_count_dict[count_value] = defaultdict(int)
                lenbucket_count_dict[count_value][(min_plen, max_plen)] += 1
                attention = attention.tolist()
                data_dicts.append({'attention': attention, 'count_value': count_value})
                instances_for_bucket += 1
            print(f"{min_plen}, {max_plen} :: {instances_for_bucket}")
        print('\n')

    print(lenbucket_count_dict)
    return data_dicts


def write_data_to_file(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__=='__main__':
    train_data = make_data(min_passage_length=100, max_passage_length=600, min_span_length=5,
                           max_span_length=15, max_count_value=7, samples_per_bucket_count=2000)

    dev_data = make_data(min_passage_length=100, max_passage_length=600, min_span_length=5,
                         max_span_length=15, max_count_value=7, samples_per_bucket_count=500)

    train_data_path = "./resources/data/drop_s/synthetic/pattn2count/train.json"
    dev_data_path = "./resources/data/drop_s/synthetic/pattn2count/dev.json"


    write_data_to_file(train_data, train_data_path)
    write_data_to_file(dev_data, dev_data_path)








