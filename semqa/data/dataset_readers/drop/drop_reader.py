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
                 question_length_limit_for_evaluation: int = None,
                 only_strongly_supervised: bool = False,
                 skip_instances=False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._relaxed_span_match = relaxed_span_match
        self._do_augmentation = do_augmentation
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.passage_length_limit_for_eval = passage_length_limit_for_evaluation or passage_length_limit
        self.question_length_limit_for_eval = question_length_limit_for_evaluation or question_length_limit
        self.only_strongly_supervised = only_strongly_supervised
        self.skip_instances = skip_instances
        self.skipped_instances = 0

    @overrides
    def _read(self, file_path: str):
        self.skipped_instances = 0
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
            original_passage_text = passage_info[constants.cleaned_passage]
            passage_text = passage_info[constants.tokenized_passage]
            passage_charidxs = passage_info[constants.passage_charidxs]
            p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = \
                passage_info[constants.passage_date_mens]
            p_date_entidxs: List[int] = passage_info[constants.passage_date_entidx]
            p_date_normvals: List[Tuple[int, int, int]] = passage_info[constants.passage_date_normalized_values]

            p_num_mens: List[Tuple[str, int, int]] = passage_info[constants.passage_num_mens]
            p_num_entidxs: List[int] = passage_info[constants.passage_num_entidx]
            p_num_normvals: List[int] = passage_info[constants.passage_num_normalized_values]


            for question_answer in passage_info[constants.qa_pairs]:
                question_id = question_answer[constants.query_id]
                original_ques_text = question_answer[constants.cleaned_question]
                question_text = question_answer[constants.tokenized_question]
                question_charidxs = question_answer[constants.question_charidxs]

                answer_type = question_answer[constants.answer_type]
                answer_passage_spans = question_answer[constants.answer_passage_spans]
                answer_question_spans = question_answer[constants.answer_question_spans]
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                # answer_annotation = question_answer["answer"] if "answer" in question_answer else None

                strongly_supervised = False
                if constants.strongly_supervised in question_answer:
                    strongly_supervised = question_answer[constants.strongly_supervised]

                qtype = "UNK"
                if constants.qtype in question_answer:
                    qtype = question_answer[constants.qtype]

                ques_attn_supervision = None
                if constants.ques_attention_supervision in question_answer:
                    ques_attn_supervision = question_answer[constants.ques_attention_supervision]

                datecomp_ques_event_date_groundings = None
                if constants.datecomp_ques_event_date_groundings in question_answer:
                    datecomp_ques_event_date_groundings = question_answer[constants.datecomp_ques_event_date_groundings]

                numcomp_qspan_num_groundings = None
                if constants.numcomp_qspan_num_groundings in question_answer:
                    numcomp_qspan_num_groundings = question_answer[constants.numcomp_qspan_num_groundings]

                instance = self.text_to_instance(question_text,
                                                 original_ques_text,
                                                 question_charidxs,
                                                 passage_text,
                                                 original_passage_text,
                                                 passage_charidxs,
                                                 p_date_mens,
                                                 p_date_entidxs,
                                                 p_date_normvals,
                                                 p_num_mens,
                                                 p_num_entidxs,
                                                 p_num_normvals,
                                                 strongly_supervised,
                                                 qtype,
                                                 ques_attn_supervision,
                                                 answer_type,
                                                 answer_passage_spans,
                                                 answer_question_spans,
                                                 datecomp_ques_event_date_groundings,
                                                 numcomp_qspan_num_groundings,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 max_passage_len,
                                                 max_question_len,
                                                 drop_invalid=is_train)

                if self.only_strongly_supervised:
                    if not strongly_supervised:
                        instance = None

                if instance is not None:
                    yield instance

        #         if instance is not None:
        #             instances.append(instance)
        #         else:
        #             skip_count += 1
        # logger.info(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        # return instances

    @overrides
    def text_to_instance(self,
                         question_text: str,
                         original_ques_text: str,
                         question_charidxs: List[int],
                         passage_text: str,
                         original_passage_text: str,
                         passage_charidxs: List[int],
                         p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
                         p_date_entidxs: List[int],
                         p_date_normvals: List[Tuple[int, int, int]],
                         p_num_mens: List[Tuple[str, int, int]],
                         p_num_entidxs: List[int],
                         p_num_normvals: List[int],
                         strongly_supervised: bool,
                         qtype: str,
                         ques_attn_supervision: Tuple[List[float]],
                         answer_type: str,
                         answer_passage_spans: List[Tuple[int, int]],
                         answer_question_spans: List[Tuple[int, int]],
                         datecomp_ques_event_date_groundings: Tuple[List[int], List[int]] = None,
                         numcomp_qspan_num_groundings: Tuple[List[int], List[int]] = None,
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict[str, Union[str, Dict, List]]] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None,
                         drop_invalid: bool = False) -> Union[Instance, None]:

        language = get_empty_language_object()

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        # pylint: disable=arguments-differ
        passage_tokens = [Token(text=t, idx=t_charidx)
                          for t, t_charidx in zip(passage_text.split(' '), passage_charidxs)]

        question_tokens = [Token(text=t, idx=t_charidx)
                           for t, t_charidx in zip(question_text.split(' '), question_charidxs)]

        if max_passage_len is not None:
            passage_tokens = passage_tokens[: max_passage_len]
        if max_question_len is not None:
            question_tokens = question_tokens[: max_question_len]

        metadata = {
            "original_passage": original_passage_text,
            "original_question": original_ques_text,
            # "original_numbers": numbers_in_passage,
            "passage_id": passage_id,
            "question_id": question_id,
            # "candidate_additions": candidate_additions,
            # "candidate_subtractions": candidate_subtractions
        }

        fields = {}

        fields["actions"] = action_field
        # fields["languages"] = language_field
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        fields["passage"] = TextField(passage_tokens, self._token_indexers)
        fields["question"] = TextField(question_tokens, self._token_indexers)

        ##  Passage Number
        # The normalized values in processed dataset are floats even if the passage had ints. Converting them back ..
        p_num_normvals = [int(x) if int(x) == x else x for x in p_num_normvals]
        passage_number_entidxs = p_num_entidxs      # same length as p_num_mens, containing num_grounding for the mens
        passage_number_values = p_num_normvals
        passage_number_indices = [tokenidx for (_, tokenidx, _) in p_num_mens]

        (passage_number_values,
         passage_number_entidxs) = self.sort_passage_numbers(passage_number_values=passage_number_values,
                                                             passage_number_entidxs=passage_number_entidxs)

        # List of passage_len containing number_entidx for each token (-1 otherwise)
        passage_number_idx2entidx = [-1 for _ in range(len(passage_tokens))]
        if passage_number_entidxs:
            for passage_num_idx, number_idx in zip(passage_number_indices, passage_number_entidxs):
                passage_number_idx2entidx[passage_num_idx] = number_idx
        else:
            # No numbers found in the passage - making a fake number at the 0th token
            passage_number_idx2entidx[0] = 0
            passage_number_values = [-1]
        fields["passageidx2numberidx"] = ArrayField(np.array(passage_number_idx2entidx), padding_value=-1)
        fields["passage_number_values"] = MetadataField(passage_number_values)

        ##  Passage Dates
        passage_date_entidxs = p_date_entidxs
        passage_date_values = p_date_normvals
        passage_date_spanidxs = [(x, y) for (_, (x, y), _) in p_date_mens]
        passage_date_idx2dateidx = [-1 for _ in range(len(passage_tokens))]
        if passage_date_values:
            for passage_date_span, date_idx in zip(passage_date_spanidxs, passage_date_entidxs):
                (s, e) = passage_date_span
                passage_date_idx2dateidx[s:e+1] = [date_idx] * (e + 1 - s)
            passage_date_objs = [Date(day=d, month=m, year=y) for (d, m, y) in passage_date_values]
        else:
            passage_date_idx2dateidx[0] = 0
            passage_date_objs = [Date(day=-1, month=-1, year=-1)]
        fields["passageidx2dateidx"] = ArrayField(np.array(passage_date_idx2dateidx), padding_value=-1)
        fields["passage_date_values"] = MetadataField(passage_date_objs)
        passage_date_strvals = [str(d) for d in passage_date_objs]

        # year_differences: List[int]
        year_differences, year_differences_mat = self.get_year_difference_candidates(passage_date_objs)
        fields["year_differences"] = MetadataField(year_differences)
        fields["year_differences_mat"] = MetadataField(year_differences_mat)

        metadata.update({"passage_token_offsets": passage_offsets,
                         "question_token_offsets": question_offsets,
                         "question_tokens": [token.text for token in question_tokens],
                         "passage_tokens": [token.text for token in passage_tokens],
                         "passage_date_values": passage_date_strvals,
                         "passage_number_values": passage_number_values,
                         "passage_year_diffs": year_differences
                         # "number_tokens": [token.text for token in number_tokens],
                         # "number_indices": number_indices
                        })


        # FIELDS FOR STRONG-SUPERVISION
        fields["strongly_supervised"] = MetadataField(strongly_supervised)
        fields["qtypes"] = MetadataField(qtype)   # String for strong supervision

        # Question Attention Supervision
        if strongly_supervised and ques_attn_supervision:
            if qtype in [constants.DATECOMP_QTYPE, constants.NUMCOMP_QTYPE]:
                # QAttn supervision, is a n-tuple of question attentions
                ques_attn_supervision = (ques_attn_supervision[1], ques_attn_supervision[0])
            fields["qattn_supervision"] = ArrayField(np.array(ques_attn_supervision), padding_value=0)
        else:
            qlen = len(question_tokens)
            empty_question_attention = [0.0] * qlen
            empty_question_attention_tuple = [empty_question_attention]
            fields["qattn_supervision"] = ArrayField(np.array(empty_question_attention_tuple), padding_value=0)

        # Date-comparison - Date Grounding Supervision
        if strongly_supervised and datecomp_ques_event_date_groundings:
            # TODO(nitish): Reverse this in pre-processing
            datecomp_ques_event_date_groundings_reversed = (datecomp_ques_event_date_groundings[1],
                                                            datecomp_ques_event_date_groundings[0])
            fields["datecomp_ques_event_date_groundings"] = MetadataField(datecomp_ques_event_date_groundings_reversed)
        else:
            empty_date_grounding = [0.0] * len(passage_date_objs)
            empty_date_grounding_tuple = (empty_date_grounding, empty_date_grounding)
            fields["datecomp_ques_event_date_groundings"] = MetadataField(empty_date_grounding_tuple)

        # Number Comparison - Passage Number Grounding Supervision
        if strongly_supervised and numcomp_qspan_num_groundings:
            # TODO(nitish): Reverse this in pre-processing
            numcomp_qspan_num_groundings_reversed = (numcomp_qspan_num_groundings[1],
                                                     numcomp_qspan_num_groundings[0])
            fields["numcomp_qspan_num_groundings"] = MetadataField(numcomp_qspan_num_groundings_reversed)
        else:
            empty_passagenum_grounding = [0.0] * len(passage_number_values)
            empty_passagenum_grounding_tuple = (empty_passagenum_grounding, empty_passagenum_grounding)
            fields["numcomp_qspan_num_groundings"] = MetadataField(empty_passagenum_grounding_tuple)

        # Get gold action_seqs for strongly_supervised questions
        action2idx_map = {rule: i for i, rule in enumerate(language.all_possible_productions())}

        # Tuple[List[List[int]], List[List[int]]]
        (gold_action_seqs,
         gold_actionseq_masks,
         instance_w_goldprog) = self.get_gold_action_seqs(qtype=qtype,
                                                          questions_tokens=question_text.split(' '),
                                                          language=language,
                                                          action2idx_map=action2idx_map)
        fields["gold_action_seqs"] = MetadataField((gold_action_seqs, gold_actionseq_masks))
        fields["instance_w_goldprog"] = MetadataField(instance_w_goldprog)


        ########     ANSWER FIELDS      ###################

        if answer_annotations:
            metadata.update({"answer_annotations": answer_annotations})

            # Using the first one for training (really, there's only one)
            answer_annotation = answer_annotations[0]
            answer_type = "UNK"
            if answer_annotation["spans"]:
                answer_type = "spans"
            elif answer_annotation["number"]:
                answer_type = "number"
            else:
                raise NotImplementedError

            # This list contains the possible-start-types for programs that can yield the correct answer
            # For example, if the answer is a number but also in passage, this will contain two keys
            # If the answer is a number, we'll find which kind and that program-start-type will be added here
            answer_program_start_types = []

            # We've pre-parsed the span types to passage / question spans

            # Passage-span answer
            if answer_passage_spans:
                answer_program_start_types.append("passage_span")
                passage_span_fields = \
                    [SpanField(span[0], span[1], fields["passage"]) for span in answer_passage_spans]
                metadata.update({'answer_passage_spans': answer_passage_spans})
            else:
                passage_span_fields = [SpanField(-1, -1, fields["passage"])]
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            # if answer_question_spans:
            #     answer_program_start_types.append("question_span")
            #     question_span_fields = \
            #         [SpanField(span[0], span[1], fields["question"]) for span in answer_question_spans]
            #     metadata.update({'answer_question_spans': answer_question_spans})
            # else:
            #     question_span_fields = [SpanField(-1, -1, fields["question"])]
            # fields["answer_as_question_spans"] = ListField(question_span_fields)

            # Question-span answer
            question_span_fields = [SpanField(-1, -1, fields["question"])]
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            # Number answers
            ans_as_passage_number = [0] * len(passage_number_values)
            ans_as_year_difference = [0] * len(year_differences)
            if answer_type == "number":
                answer_text = answer_annotation["number"]
                answer_number = float(answer_text)
                # Number lists below are mix of floats and ints. Type of answer_number doesn't matter

                # Passage-number answer
                if answer_number in passage_number_values:
                    answer_program_start_types.append("passage_number")
                    answer_number = int(answer_number) if int(answer_number) == answer_number else answer_number
                    ans_as_passage_number_idx = passage_number_values.index(answer_number)
                    ans_as_passage_number[ans_as_passage_number_idx] = 1

                # Year-difference answer
                if answer_number in year_differences:
                    answer_number = int(answer_number) if int(answer_number) == answer_number else answer_number
                    answer_program_start_types.append("year_difference")
                    ans_as_year_difference_idx = year_differences.index(answer_number)
                    ans_as_year_difference[ans_as_year_difference_idx] = 1

            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_year_difference"] = MetadataField(ans_as_year_difference)

            fields["answer_program_start_types"] = MetadataField(answer_program_start_types)

            if len(answer_program_start_types) == 0:
                print(original_ques_text)
                print(original_passage_text)
                print(answer_annotation)
                print(f"PassageNumVals:{passage_number_values}")
                print(f"PassageDates:{passage_date_strvals}")
                print(f"YearDiffs:{year_differences}")


            if self.skip_instances:
                if len(answer_program_start_types) == 0:
                    self.skipped_instances += 1
                    print(f"Skipped instances: {self.skipped_instances}")
                    # print("\nNo answer grounding")
                    # print(original_ques_text)
                    # print(original_passage_text)
                    # print(answer_annotation)
                    # print(answer_passage_spans)
                    # print(answer_question_spans)
                    return None


        # TODO(nitish): Only using questions which have PassageSpan as answers
        '''
        if not answer_passage_spans:
            # print("Not dealing with empty passage answers")
            return None
        '''

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def sort_passage_numbers(passage_number_values, passage_number_entidxs):
        """ It will be easier to do computations in the model if the number values are sorted in increasing order.
            Here we sort passage_number_values and correspondingly change the values in passage_number_entidxs

            passage_number_indices: List[int] are the token_idxs of numbers in the passage
            passage_number_entidxs: List[int] For each index above, mapping to the actual number in passage_number_values

            Therefore if we re-order passage_number_values, we need to change the values in passage_number_entidxs
            accordingly. For example:
            passage_number_values = [20, 10, 30]
            passage_number_entidxs = [2, 1, 0, 0, 1]

            After sorting,
            passage_number_values = [10, 20, 30]
            passage_number_entidxs = [2, 0, 1, 1, 0]
        """

        # [ (new_idx, value) ]
        new_idx_values_tuples = sorted(enumerate(passage_number_values), key=lambda x: x[1])

        sorted_passage_number_values = [x[1] for x in new_idx_values_tuples]
        reordered_oldidxs = [x[0] for x in new_idx_values_tuples]

        oldidx2newidx = {}
        for new_idx, oldidx in enumerate(reordered_oldidxs):
            oldidx2newidx[oldidx] = new_idx

        sorted_passage_number_entidxs = [oldidx2newidx[x] for x in passage_number_entidxs]

        return sorted_passage_number_values, sorted_passage_number_entidxs


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
    def get_year_difference_candidates(passage_date_objs: List[Date]) -> Tuple[List[int], np.array]:
        """ List of integers indicating all-possible year differences between the passage-dates
            If year difference is not defined (year = -1) or negative, we don't consider such date-combinations

            Returns the following:

            Returns:
            ---------
            year_differences:
                List[int] These are the possible year differences.
            year_difference_mat: Binary np.array of shape (D, D, y_d)
                Entry (i, j, k) == 1 denotes that D[i] - D[j] == year_differences[k]
        """
        num_date_objs = len(passage_date_objs)
        # Adding zero-first since it'll definitely be added and makes sanity-checking easy
        year_differences: List[int] = [0]

        # If any year is -1, we consider the year difference to be 0
        # If the year difference is negative, we consider the difference to be 0
        for (date1, date2) in itertools.product(passage_date_objs, repeat=2):
            year_diff = date1.year_diff(date2)
            if year_diff >= 0:
                if year_diff not in year_differences:
                    year_differences.append(year_diff)

        num_of_year_differences = len(year_differences)
        # Making year_difference_mat
        year_difference_mat = np.zeros(shape=(num_date_objs, num_date_objs, num_of_year_differences), dtype=int)
        for ((date_idx1, date1), (date_idx2, date2)) in itertools.product(enumerate(passage_date_objs), repeat=2):
            year_diff = date1.year_diff(date2)
            if year_diff >= 0:
                year_diff_idx = year_differences.index(year_diff)   # We know this will not fail
                year_difference_mat[date_idx1, date_idx2, year_diff_idx] = 1

        return year_differences, year_difference_mat



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

    def get_gold_action_seqs(self,
                             qtype: str,
                             questions_tokens: List[str],
                             language: DropLanguage,
                             action2idx_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]], bool]:

        qtype_to_lffunc = {constants.DATECOMP_QTYPE: self.get_gold_logicalforms_datecomp,
                           constants.NUMCOMP_QTYPE: self.get_gold_logicalforms_numcomp}

        gold_actionseq_idxs: List[List[int]] = []
        gold_actionseq_mask: List[List[int]] = []
        instance_w_goldprog: bool = False

        if qtype in qtype_to_lffunc:
            gold_logical_forms: List[str] = qtype_to_lffunc[qtype](questions_tokens, language)
            assert len(gold_logical_forms) >= 1, f"No logical forms found for: {questions_tokens}"
            for logical_form in gold_logical_forms:
                gold_actions: List[str] = language.logical_form_to_action_sequence(logical_form)
                actionseq_idxs: List[int] = [action2idx_map[a] for a in gold_actions]
                actionseq_mask: List[int] = [1 for _ in range(len(actionseq_idxs))]
                gold_actionseq_idxs.append(actionseq_idxs)
                gold_actionseq_mask.append(actionseq_mask)
            instance_w_goldprog = True
        else:
            gold_actionseq_idxs.append([0])
            gold_actionseq_mask.append([0])
            instance_w_goldprog = False

        return gold_actionseq_idxs, gold_actionseq_mask, instance_w_goldprog

    @staticmethod
    def get_gold_logicalforms_datecomp(question_tokens: List[str], language: DropLanguage) -> List[str]:
        # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
        psa_start = "(find_passageSpanAnswer ("
        qsa_start = "(find_questionSpanAnswer ("
        # lf1 = "(find_passageSpanAnswer ("

        lf2 = " find_PassageAttention find_PassageAttention))"
        greater_than = "compare_date_greater_than"
        lesser_than = "compare_date_lesser_than"

        # Correct if Attn1 is first event
        lesser_tokens = ['first', 'earlier', 'forst', 'firts']
        greater_tokens = ['later', 'last', 'second']

        operator_action = None

        for t in lesser_tokens:
            if t in question_tokens:
                operator_action = lesser_than
                break

        for t in greater_tokens:
            if t in question_tokens:
                operator_action = greater_than
                break

        if operator_action is None:
            operator_action = greater_than

        gold_logical_forms = []
        if '@start@ -> PassageSpanAnswer' in language.all_possible_productions():
            gold_logical_forms.append(f"{psa_start}{operator_action}{lf2}")
        if '@start@ -> QuestionSpanAnswer' in language.all_possible_productions():
            gold_logical_forms.append(f"{qsa_start}{operator_action}{lf2}")

        return gold_logical_forms

    @staticmethod
    def get_gold_logicalforms_numcomp(question_tokens: List[str], language: DropLanguage) -> List[str]:
        # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
        psa_start = "(find_passageSpanAnswer ("
        qsa_start = "(find_questionSpanAnswer ("

        lf2 = " find_PassageAttention find_PassageAttention))"
        greater_than = "compare_num_greater_than"
        lesser_than = "compare_num_lesser_than"

        # Correct if Attn1 is first event
        greater_tokens = ['larger', 'more', 'largest', 'bigger', 'higher', 'highest', 'most', 'greater']
        lesser_tokens = ['smaller', 'fewer', 'lowest', 'smallest', 'less', 'least', 'fewest', 'lower']

        operator_action = None

        for t in lesser_tokens:
            if t in question_tokens:
                operator_action = lesser_than
                break
                # return [f"{psa_start}{lesser_than}{lf2}", f"{qsa_start}{lesser_than}{lf2}"]
        if operator_action is None:
            for t in greater_tokens:
                if t in question_tokens:
                    operator_action = greater_than
                    break
                    # return [f"{psa_start}{greater_than}{lf2}", f"{qsa_start}{greater_than}{lf2}"]

        if operator_action is None:
            operator_action = greater_than

        gold_logical_forms = []
        if '@start@ -> PassageSpanAnswer' in language.all_possible_productions():
            gold_logical_forms.append(f"{psa_start}{operator_action}{lf2}")
        if '@start@ -> QuestionSpanAnswer' in language.all_possible_productions():
            gold_logical_forms.append(f"{qsa_start}{operator_action}{lf2}")

        return gold_logical_forms


