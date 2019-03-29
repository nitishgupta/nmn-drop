import json
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

from semqa.domain_languages.drop.drop_language import DropLanguage, Date, get_empty_language_object

from datasets.drop import constants

# from reading_comprehension.utils import split_tokens_by_hyphen

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: Add more number here
WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("drop_reader")
class DROPReader(DatasetReader):
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 relaxed_span_match: bool = True,
                 do_augmentation: bool = True,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 passage_length_limit_for_evaluation: int = None,
                 question_length_limit_for_evaluation: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._relaxed_span_match = relaxed_span_match
        self._do_augmentation = do_augmentation
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.passage_length_limit_for_eval = passage_length_limit_for_evaluation or passage_length_limit
        self.question_length_limit_for_eval = question_length_limit_for_evaluation or question_length_limit

    @overrides
    def _read(self, file_path: str):
        # pylint: disable=logging-fstring-interpolation
        # if `file_path` is a URL, redirect to the cache
        is_train = "train" in str(file_path)
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        instances, skip_count = [], 0
        max_passage_len = self.passage_length_limit if is_train else self.passage_length_limit_for_eval
        max_question_len = self.question_length_limit if is_train else self.question_length_limit_for_eval
        for passage_id, passage_info in dataset.items():
            original_passage_text = passage_info[constants.original_passage]
            passage_text = passage_info[constants.passage]
            passage_charidxs = passage_info[constants.passage_charidxs]
            passage_len = passage_info["num_tokens"]
            p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = \
                passage_info[constants.passage_date_mens]
            p_date_entidxs: List[int] = passage_info[constants.passage_date_entidx]
            p_date_normvals: List[Tuple[int, int, int]] = passage_info[constants.passage_date_normalized_values]

            p_num_mens: List[Tuple[str, int, int]] = passage_info[constants.passage_num_mens]
            p_num_entidxs: List[int] = passage_info[constants.passage_num_entidx]
            p_num_normvals: List[int] = passage_info[constants.passage_num_normalized_values]


            for question_answer in passage_info[constants.qa_pairs]:
                question_id = question_answer[constants.query_id]
                original_ques_text = question_answer[constants.original_question]
                question_text = question_answer[constants.question]
                question_charidxs = question_answer[constants.question_charidxs]
                question_len = question_answer["num_tokens"]

                answer_type = question_answer[constants.answer_type]
                answer_passage_spans = question_answer[constants.answer_passage_spans]
                answer_question_spans = question_answer[constants.answer_question_spans]
                answer_annotation = question_answer["answer"] if "answer" in question_answer else None
                instance = self.text_to_instance(question_text,
                                                 original_ques_text,
                                                 question_charidxs,
                                                 question_len,
                                                 passage_text,
                                                 original_passage_text,
                                                 passage_charidxs,
                                                 passage_len,
                                                 p_date_mens,
                                                 p_date_entidxs,
                                                 p_date_normvals,
                                                 p_num_mens,
                                                 p_num_entidxs,
                                                 p_num_normvals,
                                                 answer_type,
                                                 answer_passage_spans,
                                                 answer_question_spans,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotation,
                                                 max_passage_len,
                                                 max_question_len,
                                                 drop_invalid=is_train)

                if instance is not None:
                    yield instance

        #         if instance is not None:
        #             instances.append(instance)
        #         else:
        #             skip_count += 1
        # logger.info(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        # return instances

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         original_ques_text: str,
                         question_charidxs: List[int],
                         question_len: int,
                         passage_text: str,
                         original_passage_text: str,
                         passage_charidxs: List[int],
                         passage_len: int,
                         p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
                         p_date_entidxs: List[int],
                         p_date_normvals: List[Tuple[int, int, int]],
                         p_num_mens: List[Tuple[str, int, int]],
                         p_num_entidxs: List[int],
                         p_num_normvals: List[int],
                         answer_type: str,
                         answer_passage_spans: List[Tuple[int, int]],
                         answer_question_spans: List[Tuple[int, int]],
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotation: Dict[str, Union[str, Dict, List]] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None,
                         drop_invalid: bool = False) -> Union[Instance, None]:

        language = get_empty_language_object()

        # DropLanguage(None, None, None, None, None, None, None)

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        # language_field = MetadataField(language)

        # pylint: disable=arguments-differ
        passage_tokens = [Token(text=t, idx=t_charidx)
                          for t, t_charidx in zip(passage_text.split(' '), passage_charidxs)]

        question_tokens = [Token(text=t, idx=t_charidx)
                           for t, t_charidx in zip(question_text.split(' '), question_charidxs)]

        if max_passage_len is not None:
            passage_tokens = passage_tokens[: max_passage_len]
        if max_question_len is not None:
            question_tokens = question_tokens[: max_question_len]


        passage_number_indices = [tokenidx for (_, tokenidx, _) in p_num_mens]
        passage_number_entidxs = p_num_entidxs
        passage_number_values = p_num_normvals

        passage_date_spanidxs = [(x,y) for (_, (x,y), _) in p_date_mens]
        passage_date_entidxs = p_date_entidxs
        passage_date_values = p_date_normvals

        # TODO(nitish): Only span answer supported. Extend to other types
        answer_texts = answer_annotation["spans"]
        answer_texts_for_evaluation = [' '.join(answer_texts)]

        answer_info = {"answer_type": answer_type,
                       "answer_texts": answer_texts_for_evaluation,
                       "answer_passage_spans": answer_passage_spans,
                       "answer_question_spans": answer_question_spans}
        additional_metadata = {
            "original_passage": original_passage_text,
            "original_question": original_ques_text,
            "question_num_tokens": question_len,
            "passage_num_tokens": passage_len,
            # "original_numbers": numbers_in_passage,
            "passage_id": passage_id,
            "question_id": question_id,
            "answer_annotation": answer_annotation
            # "candidate_additions": candidate_additions,
            # "candidate_subtractions": candidate_subtractions
        }

    #     return self.make_augmented_instance(question_tokens=question_tokens,
    #                                         passage_tokens=passage_tokens,
    #                                         token_indexers=self._token_indexers,
    #                                         passage_number_indices=passage_number_indices,
    #                                         passage_number_entidxs=passage_number_entidxs,
    #                                         passage_number_values=passage_number_values,
    #                                         passage_date_spanidxs=passage_date_spanidxs,
    #                                         passage_date_entidxs=passage_date_entidxs,
    #                                         passage_date_values=passage_date_values,
    #                                         passage_text=original_passage_text,
    #                                         answer_info=answer_info,
    #                                         action_field=action_field,
    #                                         additional_metadata=additional_metadata)
    #
    # @staticmethod
    # def make_augmented_instance(question_tokens: List[Token],
    #                             passage_tokens: List[Token],
    #                             token_indexers: Dict[str, TokenIndexer],
    #                             passage_number_indices: List[int],
    #                             passage_number_entidxs: List[int],
    #                             passage_number_values: List[int],
    #                             passage_date_spanidxs: List[Tuple[int, int]],
    #                             passage_date_entidxs: List[int],
    #                             passage_date_values: List[Tuple[int, int, int]],
    #                             passage_text: str,
    #                             answer_info: Dict[str, Any] = None,
    #                             action_field: ListField[ProductionRuleField] = None,
    #                             # language_field: MetadataField = None,
    #                             number_tokens: List[Token]=None,
    #                             number_indices: List[int]=None,
    #                             additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}

        fields["actions"] = action_field
        # fields["languages"] = language_field
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        fields["passage"] = TextField(passage_tokens, self._token_indexers)
        fields["question"] = TextField(question_tokens, self._token_indexers)


        # List of passage_len containing number_entidx for each token (-1 otherwise)
        passage_number_idx2entidx = [-1 for _ in range(len(passage_tokens))]
        for passage_num_idx, number_idx in zip(passage_number_indices, passage_number_entidxs):
            passage_number_idx2entidx[passage_num_idx] = number_idx

        fields["passageidx2numberidx"] = ArrayField(np.array(passage_number_idx2entidx), padding_value=-1)
        fields["passage_number_values"] = MetadataField(passage_number_values)

        # if len(passage_date_spanidxs) == 0:
        #     print("SKIPPING")
            # return None

        passage_date_idx2dateidx = [-1 for _ in range(len(passage_tokens))]
        for passage_date_span, date_idx in zip(passage_date_spanidxs, passage_date_entidxs):
            (s, e) = passage_date_span
            passage_date_idx2dateidx[s:e+1] = [date_idx] * (e + 1 - s)
        fields["passageidx2dateidx"] = ArrayField(np.array(passage_date_idx2dateidx), padding_value=-1)
        passage_date_objs = [Date(day=d, month=m, year=y) for (d, m, y) in passage_date_values]
        if len(passage_date_objs) == 0:
            passage_date_objs.append(Date(day=-1, month=-1, year=-1))
        fields["passage_date_values"] = MetadataField(passage_date_objs)

        # passage_number_index_fields = [IndexField(index, fields["passage"]) for index in passage_number_indices]
        # if not passage_number_index_fields:
        #     passage_number_index_fields = [IndexField(-1, fields["passage"])]
        # fields["passage_number_indices"] = ListField(passage_number_index_fields)
        # fields["passage_number_entidxs"] = ArrayField(np.array(passage_number_entidxs), padding_value=-1)
        # fields["passage_number_values"] = MetadataField(passage_number_values)

        # passage_date_spanfields = [SpanField(s, e, fields["passage"]) for (s, e) in passage_date_spanidxs]
        # if not passage_date_spanfields:
        #     passage_date_spanfields = [SpanField(-1, -1, fields["passage"])]
        # fields["passage_date_spans"] = ListField(passage_date_spanfields)
        # fields["passage_date_entidxs"] = ArrayField(np.array(passage_date_entidxs), padding_value=-1)
        # fields["passage_date_values"] = MetadataField(passage_date_values)

        # This field is actually not required in the model,
        # it is used to create the `answer_as_plus_minus_combinations` field, which is a `SequenceLabelField`.
        # We cannot use `number_indices` field for creating that, because the `ListField` will not be empty
        # when we want to create a new empty field. That will lead to error.
        # fields["numbers_in_passage"] = TextField(number_tokens, token_indexers)

        metadata = additional_metadata

        metadata.update({"passage_token_offsets": passage_offsets,
                         "question_token_offsets": question_offsets,
                         "question_tokens": [token.text for token in question_tokens],
                         "passage_tokens": [token.text for token in passage_tokens],
                         # "number_tokens": [token.text for token in number_tokens],
                         # "number_indices": number_indices
                        })

        if answer_info:
            metadata["answer_type"] = answer_info["answer_type"]
            metadata["answer_texts"] = answer_info["answer_texts"]

            fields["answer_types"] = MetadataField(answer_info["answer_type"])

            passage_span_fields = \
                [SpanField(span[0], span[1], fields["passage"]) for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                return None
                passage_span_fields.append(SpanField(-1, -1, fields["passage"]))

            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields = \
                [SpanField(span[0], span[1], fields["question"]) for span in answer_info["answer_question_spans"]]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, fields["question"]))

            fields["answer_as_question_spans"] = ListField(question_span_fields)

            ''' Not dealing with arithmetic at the moment '''
            # plus_minus_combinations_fields = []
            # for plus_minus_combination in answer_info["plus_minus_combinations"]:
            #     plus_minus_combinations_fields.append(
            #             SequenceLabelField(plus_minus_combination, fields["numbers_in_passage"]))
            # if not plus_minus_combinations_fields:
            #     plus_minus_combinations_fields.append(
            #             SequenceLabelField([0] * len(fields["numbers_in_passage"]), fields["numbers_in_passage"]))
            # fields["answer_as_plus_minus_combinations"] = ListField(plus_minus_combinations_fields)
            #
            # count_fields = [LabelField(count_label, skip_indexing=True) for count_label in answer_info["counts"]]
            # if not count_fields:
            #     count_fields.append(LabelField(-1, skip_indexing=True))
            # fields["answer_as_counts"] = ListField(count_fields)

        # TODO(nitish): Only using questions which have PassageSpan as answers
        if not answer_info["answer_passage_spans"]:
            # print("Not dealing with empty passage answers")
            return None

        # TODO(nitish): Only using questions which have a single 'or'
        if len(' '.join(metadata["question_tokens"]).split(' or ')) != 2:
            print("Only date comparison with single OR")
            return None

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def convert_string_to_int(string: str):
        no_comma_string = string.replace(",", "")
        if no_comma_string in WORD_NUMBER_MAP:
            number = WORD_NUMBER_MAP[no_comma_string]
        else:
            try:
                number = int(no_comma_string)
            except ValueError:
                number = None
        return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_plus_minus_combinations(numbers: List[int],
                                           targets: List[int],
                                           max_length_of_combinations: int = 2) -> List[List[int]]:
        valid_combinations = []
        for combination_length in range(2, max_length_of_combinations + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=combination_length))
            for combination in itertools.combinations(enumerate(numbers), combination_length):
                indices = [it[0] for it in combination]
                values = [it[1] for it in combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    if eval_value in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_combinations.append(labels_for_numbers)
        return valid_combinations

    @staticmethod
    def find_valid_count(count_numbers: List[int],
                         targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices

    @staticmethod
    def convert_answer(answer_annotation: Dict[str, Union[str, Dict, List]]) -> Tuple[str, List]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def get_candidate_additions(numbers_in_passage: List[int],
                                number_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        candidate_additions = defaultdict(list)

        for number_1, index_1 in zip(numbers_in_passage, number_indices):
            for number_2, index_2 in zip(numbers_in_passage, number_indices):
                result = number_1 + number_2
                candidate_additions[result].append((index_1, index_2))
        return candidate_additions

    @staticmethod
    def get_candidate_subtractions(numbers_in_passage: List[int],
                                   number_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        candidate_subtractions = defaultdict(list)

        for number_1, index_1 in zip(numbers_in_passage, number_indices):
            for number_2, index_2 in zip(numbers_in_passage, number_indices):
                result = number_1 - number_2
                candidate_subtractions[result].append((index_1, index_2))
        return candidate_subtractions
