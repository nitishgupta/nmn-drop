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

import utils.util as myutil

from datasets.drop import constants

# from reading_comprehension.utils import split_tokens_by_hyphen

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: Add more number here
WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("numdist2count_reader")
class NUmDist2CountReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 min_dist_length=6,
                 max_dist_length=14,
                 max_count=7,
                 num_training_samples=2000,
                 noise_std=0.05,
                 normalized=True,
                 withnoise=True)-> None:
        super().__init__(lazy)

        self._min_dist_length = min_dist_length
        self._max_dist_length = max_dist_length
        self._max_count = max_count
        self._num_training_samples = num_training_samples
        self._normalized = normalized
        self._withnoise = withnoise
        self._noise_std = noise_std
        self.instances_made = 0

    @overrides
    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Making {self._num_training_samples} training examples with:\n"
                    f"max_pdist_length: {self._max_dist_length}\n"
                    f"min_dist_length: {self._min_dist_length}\n"
                    f"max_count:{self._max_count}\n")

        instances: List[Instance] = []
        for i in range(self._num_training_samples):
            fields: Dict[str, Field] = {}

            dist_length = random.randint(self._min_dist_length, self._max_dist_length)
            count_value = random.randint(1, min(self._max_count, dist_length))

            number_distribution = [0.0] * dist_length
            if count_value > 0:
                indices = random.sample(range(dist_length), count_value)
                # Add 1.0 to all sampled indices
                for i in indices:
                    number_distribution[i] += 1.0

            if self._withnoise:
                std_dev = random.uniform(0.01, 0.1)
                number_distribution = [x + abs(random.gauss(0, std_dev)) for x in number_distribution]

            if self._normalized:
                attention_sum = sum(number_distribution)
                number_distribution = [float(x)/attention_sum for x in number_distribution]

            number_distribution = myutil.round_all(number_distribution, 3)

            print(f"{number_distribution}   {count_value}")

            fields["number_dist"] = ArrayField(np.array(number_distribution), padding_value=-1)

            fields["count_answer"] = LabelField(count_value, skip_indexing=True)

            instances.append(Instance(fields))
            self.instances_made += 1

        print(f"Instances made: {self.instances_made}")
        return instances
