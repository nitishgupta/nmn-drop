import json
import random
import logging
import itertools
import numpy as np
from typing import Dict, List, Union, Tuple, Any
from collections import defaultdict
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension.util import make_reading_comprehension_instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS
from allennlp.data.fields import Field, TextField, MetadataField, LabelField, ListField, \
    SequenceLabelField, SpanField, IndexField, ProductionRuleField, ArrayField

from semqa.domain_languages.drop_old.drop_language import DropLanguage, Date, get_empty_language_object
from collections import defaultdict

from datasets.drop import constants

# from reading_comprehension.utils import split_tokens_by_hyphen

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: Add more number here
WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("passage_attn2count_reader")
class DROPReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 min_passage_length=200,
                 max_passage_length=400,
                 max_span_length=10,
                 num_training_samples=2000,
                 normalized=True,
                 withnoise=True)-> None:
        super().__init__(lazy)

        self._min_passage_length = min_passage_length
        self._max_passage_length = max_passage_length
        self._max_span_length = max_span_length
        self._num_training_samples = num_training_samples
        self._normalized = normalized
        self._withnoise = withnoise
    '''
    @overrides
    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Making {self._num_training_samples} training examples with:\n"
                    f"max_passage_length: {self._max_passage_length}\n"
                    f"min_passage_len: {self._min_passage_length}\n"
                    f"max_span_len:{self._max_span_length}\n")

        instances: List[Instance] = []
        for i in range(self._num_training_samples):
            fields: Dict[str, Field] = {}

            passage_length = random.randint(self._min_passage_length, self._max_passage_length)
            attention = [0.0 for _ in range(passage_length)]

            span_length = random.randint(1, self._max_span_length)

            # Inclusive start and end positions
            start_position = random.randint(0, passage_length - span_length)
            end_position = start_position + span_length - 1

            attention[start_position:end_position + 1] = [1.0] * span_length

            if self._withnoise:
                attention = [x + abs(random.gauss(0, 0.001)) for x in attention]

            if self._normalized:
                attention_sum = sum(attention)
                attention = [float(x)/attention_sum for x in attention]

            passage_span_fields = ArrayField(np.array([[start_position, end_position]]), padding_value=-1)

            fields["passage_attention"] = ArrayField(np.array(attention), padding_value=0.0)

            fields["passage_lengths"] = MetadataField(passage_length)

            fields["answer_as_passage_spans"] = passage_span_fields

            instances.append(Instance(fields))

            print("Making data")

        return instances
    '''

    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation

        instances: List[Instance] = []
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info(f"Reading the dataset from: {file_path}")

        count_dist = defaultdict(int)
        masked = 0

        for passage_id, passage_info in dataset.items():
            passage_text = passage_info[constants.tokenized_passage]
            passage_length = len(passage_text.split(' '))
            passage_tokens = passage_text.split(' ')
            # print(passage_text)
            # print()

            for question_answer in passage_info[constants.qa_pairs]:
                fields = {}

                # TODO(nitish): Only using first span as answer

                start_position = 5
                end_position = 10


                attention, count_answer, mask = self.make_count_instance(passage_tokens)
                if mask != 0:
                    count_dist[count_answer] += 1
                else:
                    masked += 1


                if self._withnoise:
                    attention = [x + abs(random.gauss(0, 0.001)) for x in attention]

                if self._normalized:
                    attention_sum = sum(attention)
                    attention = [float(x) / attention_sum for x in attention]

                count_answer_vec = [0] * 10
                count_answer_vec[count_answer] = 1

                fields["passage_attention"] = ArrayField(np.array(attention), padding_value=0.0)

                fields["passage_lengths"] = MetadataField(passage_length)

                fields["answer_as_count"] = ArrayField(np.array(count_answer_vec))

                fields["count_mask"] = ArrayField(np.array(mask))

                instances.append(Instance(fields))

        print(count_dist)

        return instances



    def make_count_instance(self, passage_tokens: List[str]):
        ''' output an attention, count_answer, mask. Mask is when we don;t find relevant spans '''

        # We would like to count these spans
        relevant_spans = ['TD pass', 'TD run', 'touchdown pass', 'field goal', 'touchdown run']
        num_relevant_spans = len(relevant_spans)

        attention = [0.0] * len(passage_tokens)

        # With 10% prob select no span
        count_zero_prob = random.random()
        if count_zero_prob < 0.1:
            return (attention, 0, 1)


        # Choose a particular type of span from relevant ones and find it's starting positions
        tries = 0
        starting_positions_in_passage = []
        while len(starting_positions_in_passage) == 0 and tries < 5:
            choosen_span = random.randint(0, num_relevant_spans - 1)
            span_tokens = relevant_spans[choosen_span].split(' ')
            starting_positions_in_passage = self.contains(span_tokens, passage_tokens)
            tries += 1

        # even after 5 tries, span to count not found. Return masked attention
        if len(starting_positions_in_passage) == 0:
            return attention, 0, 0

        # # TO save from infinite loop
        # count_zero_prob = random.random()
        # if count_zero_prob < 0.1:
        #     return attention, 0

        if len(starting_positions_in_passage) == 1:
            count = len(starting_positions_in_passage)
            starting_position = starting_positions_in_passage[0]
            attention[starting_position] = 1.0
            attention[starting_position + 1] = 1.0

        else:
            num_of_spans_found = len(starting_positions_in_passage)
            # Choose a subset of the starting_positions
            random.shuffle(starting_positions_in_passage)
            num_spans = random.randint(2, num_of_spans_found)
            num_spans = min(num_spans, 9)

            count = num_spans

            spread_len = random.randint(1, 3)

            chosen_starting_positions = starting_positions_in_passage[0:num_spans]
            for starting_position in chosen_starting_positions:
                attention[starting_position] = 1.0
                attention[starting_position + 1] = 1.0
                for i in range(1, spread_len+1):
                    prev_idx = starting_position - i
                    if prev_idx >= 0:
                        attention[prev_idx] = 0.5
                    next_idx = starting_position + 1 + i
                    if next_idx < len(passage_tokens):
                        attention[next_idx] = 0.5

        return attention, count, 1

    def contains(self, small, big):
        starting_positions = []
        for i in range(len(big) - len(small) + 1):
            start = True
            for j in range(len(small)):
                if big[i + j] != small[j]:
                    start = False
                    break
            if start:
                starting_positions.append(i)
        return starting_positions
