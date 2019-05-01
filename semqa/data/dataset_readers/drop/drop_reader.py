import json
import logging
import random
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
                 only_strongly_supervised: bool = False,
                 skip_instances=False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._relaxed_span_match = relaxed_span_match
        self._do_augmentation = do_augmentation
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.only_strongly_supervised = only_strongly_supervised
        self.skip_instances = skip_instances
        self.skipped_instances = 0

    @overrides
    def _read(self, file_path: str):
        self.skipped_instances = 0
        # pylint: disable=logging-fstring-interpolation
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        instances, skip_count = [], 0
        max_passage_len = self.passage_length_limit
        max_question_len = self.question_length_limit
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


            for qa in passage_info[constants.qa_pairs]:
                question_id = qa[constants.query_id]
                original_ques_text = qa[constants.cleaned_question]
                question_text = qa[constants.tokenized_question]
                question_charidxs = qa[constants.question_charidxs]

                answer_type = qa[constants.answer_type]
                answer_passage_spans = qa[constants.answer_passage_spans]
                answer_question_spans = qa[constants.answer_question_spans]
                answer_annotations = []
                if "answer" in qa:
                    answer_annotations.append(qa["answer"])
                if "validated_answers" in qa:
                    answer_annotations += qa["validated_answers"]
                # answer_annotation = question_answer["answer"] if "answer" in question_answer else None

                qtype = "UNK"
                if constants.qtype in qa and qa[constants.qtype] is not None:
                    qtype = qa[constants.qtype]
                program_supervised = False
                if constants.program_supervised in qa:
                    program_supervised = qa[constants.program_supervised]

                # If qtype is known and program_supervised = False OR
                # If qtype is unknown and program_supervision is True --- There's a problem, Houston!
                if (program_supervised and qtype == "UNK") or (qtype != "UNK" and program_supervised is False):
                    print(original_ques_text)
                    print(f"Qtype: {qtype}")
                    print(f"Program supervised: {program_supervised}")
                    exit()

                ques_attn_supervision = None
                qattn_supervised = False
                if constants.qattn_supervised in qa:
                    qattn_supervised = qa[constants.qattn_supervised]
                    if qattn_supervised is True:
                        ques_attn_supervision = qa[constants.ques_attention_supervision]

                date_grounding_supervision = None
                num_grounding_supervision = None
                execution_supervised = False
                if constants.exection_supervised in qa:
                    execution_supervised = qa[constants.exection_supervised]
                    if qa[constants.exection_supervised] is True:
                        # There can be multiple types of execution_supervision
                        if constants.qspan_dategrounding_supervision in qa:
                            date_grounding_supervision = qa[constants.qspan_dategrounding_supervision]
                        if constants.qspan_numgrounding_supervision in qa:
                            num_grounding_supervision = qa[constants.qspan_numgrounding_supervision]

                strongly_supervised = program_supervised and qattn_supervised and execution_supervised

                if qattn_supervised is True:
                    assert program_supervised is True and qtype is not "UNK"
                if execution_supervised is True:
                    assert qattn_supervised is True

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
                                                 qtype,
                                                 program_supervised,
                                                 qattn_supervised,
                                                 execution_supervised,
                                                 strongly_supervised,
                                                 ques_attn_supervision,
                                                 date_grounding_supervision,
                                                 num_grounding_supervision,
                                                 answer_type,
                                                 answer_passage_spans,
                                                 answer_question_spans,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 max_passage_len,
                                                 max_question_len)

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
                         qtype: str,
                         program_supervised: bool,
                         qattn_supervised: bool,
                         execution_supervised: bool,
                         strongly_supervised: bool,
                         ques_attn_supervision: Tuple[List[float]],
                         date_grounding_supervision: Tuple[List[int], List[int]],
                         num_grounding_supervision: Tuple[List[int], List[int]],
                         answer_type: str,
                         answer_passage_spans: List[Tuple[int, int]],
                         answer_question_spans: List[Tuple[int, int]],
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict[str, Union[str, Dict, List]]] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None) -> Union[Instance, None]:

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
            (p_date_mens, p_date_entidxs, p_date_normvals,
             p_num_mens, p_num_entidxs, p_num_normvals,
             answer_passage_spans,
             date_grounding_supervision,
             num_grounding_supervision) = self.prune_for_passage_len(max_passage_len,
                                                                     p_date_mens, p_date_entidxs, p_date_normvals,
                                                                     p_num_mens, p_num_entidxs, p_num_normvals,
                                                                     answer_passage_spans,
                                                                     date_grounding_supervision,
                                                                     num_grounding_supervision)
        if max_question_len is not None:
            question_tokens = question_tokens[: max_question_len]
            (answer_question_spans,
             ques_attn_supervision) = self.prune_for_question_len(max_question_len, answer_question_spans,
                                                                  ques_attn_supervision)

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

        # List of passage_len containing number_entidx for each token (-1 otherwise)
        passage_number_idx2entidx = [-1 for _ in range(len(passage_tokens))]
        if passage_number_entidxs:
            for passage_num_tokenidx, number_idx in zip(passage_number_indices, passage_number_entidxs):
                passage_number_idx2entidx[passage_num_tokenidx] = number_idx
        else:
            # No numbers found in the passage - making a fake number at the 0th token
            passage_number_idx2entidx[0] = 0
        if not passage_number_values:
            passage_number_values.append(-1)
        fields["passageidx2numberidx"] = ArrayField(np.array(passage_number_idx2entidx), padding_value=-1)
        fields["passage_number_values"] = MetadataField(passage_number_values)

        ##  Passage Dates
        passage_date_entidxs = p_date_entidxs
        passage_date_values = p_date_normvals
        passage_date_spanidxs = [(x, y) for (_, (x, y), _) in p_date_mens]

        passage_date_idx2dateidx = [-1 for _ in range(len(passage_tokens))]
        if passage_date_spanidxs:
            for passage_date_span, date_idx in zip(passage_date_spanidxs, passage_date_entidxs):
                (s, e) = passage_date_span
                passage_date_idx2dateidx[s:e+1] = [date_idx] * (e + 1 - s)
        else:
            passage_date_idx2dateidx[0] = 0
        if passage_date_values:
            passage_date_objs = [Date(day=d, month=m, year=y) for (d, m, y) in passage_date_values]
        else:
            passage_date_objs = [Date(day=-1, month=-1, year=-1)]
        fields["passageidx2dateidx"] = ArrayField(np.array(passage_date_idx2dateidx), padding_value=-1)
        fields["passage_date_values"] = MetadataField(passage_date_objs)
        passage_date_strvals = [str(d) for d in passage_date_objs]

        # year_differences: List[int]
        year_differences, year_differences_mat = self.get_year_difference_candidates(passage_date_objs)
        fields["year_differences"] = MetadataField(year_differences)
        fields["year_differences_mat"] = MetadataField(year_differences_mat)

        passage_number_differences, passage_number_diff_mat = self.get_passagenumber_difference_candidates(
                                                                                            passage_number_values)
        fields["passagenumber_difference_values"] = MetadataField(passage_number_differences)
        fields["passagenumber_differences_mat"] = MetadataField(passage_number_diff_mat)

        count_values = list(range(10))
        fields["count_values"] = MetadataField(count_values)

        metadata.update({"passage_token_offsets": passage_offsets,
                         "question_token_offsets": question_offsets,
                         "question_tokens": [token.text for token in question_tokens],
                         "passage_tokens": [token.text for token in passage_tokens],
                         "passage_date_values": passage_date_strvals,
                         "passage_number_values": passage_number_values,
                         "passage_year_diffs": year_differences,
                         "passagenum_diffs": passage_number_differences,
                         "count_values": count_values
                         # "number_tokens": [token.text for token in number_tokens],
                         # "number_indices": number_indices
                        })


        # FIELDS FOR STRONG-SUPERVISION
        fields["strongly_supervised"] = MetadataField(strongly_supervised)
        fields["program_supervised"] = MetadataField(program_supervised)
        fields["qattn_supervised"] = MetadataField(qattn_supervised)
        fields["execution_supervised"] = MetadataField(execution_supervised)
        fields["qtypes"] = MetadataField(qtype)

        # Question Attention Supervision
        if ques_attn_supervision:
            fields["qattn_supervision"] = ArrayField(np.array(ques_attn_supervision), padding_value=0)
        else:
            qlen = len(question_tokens)
            empty_question_attention = [0.0] * qlen
            empty_question_attention_tuple = [empty_question_attention]
            fields["qattn_supervision"] = ArrayField(np.array(empty_question_attention_tuple), padding_value=0)

        # Date-comparison - Date Grounding Supervision
        if date_grounding_supervision:
            fields["datecomp_ques_event_date_groundings"] = MetadataField(date_grounding_supervision)
        else:
            empty_date_grounding = [0.0] * len(passage_date_objs)
            empty_date_grounding_tuple = (empty_date_grounding, empty_date_grounding)
            fields["datecomp_ques_event_date_groundings"] = MetadataField(empty_date_grounding_tuple)

        # Number Comparison - Passage Number Grounding Supervision
        if num_grounding_supervision:
            fields["numcomp_qspan_num_groundings"] = MetadataField(num_grounding_supervision)
        else:
            empty_passagenum_grounding = [0.0] * len(passage_number_values)
            empty_passagenum_grounding_tuple = (empty_passagenum_grounding, empty_passagenum_grounding)
            fields["numcomp_qspan_num_groundings"] = MetadataField(empty_passagenum_grounding_tuple)

        # Get gold action_seqs for strongly_supervised questions
        action2idx_map = {rule: i for i, rule in enumerate(language.all_possible_productions())}

        # Tuple[List[List[int]], List[List[int]]]
        (gold_action_seqs,
         gold_actionseq_masks,
         program_supervised) = self.get_gold_action_seqs(program_supervised=program_supervised,
                                                         qtype=qtype,
                                                         question_tokens=question_text.split(' '),
                                                         language=language,
                                                         action2idx_map=action2idx_map)
        fields["program_supervised"] = MetadataField(program_supervised)
        fields["gold_action_seqs"] = MetadataField((gold_action_seqs, gold_actionseq_masks))

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
            passage_span_fields = []
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
            answer_as_passagenum_difference = [0] * len(passage_number_differences)
            answer_as_count = [0] * len(count_values)
            if answer_type == "number":
                answer_text = answer_annotation["number"]
                answer_number = float(answer_text)
                answer_number = int(answer_number) if int(answer_number) == answer_number else answer_number
                # Number lists below are mix of floats and ints. Type of answer_number doesn't matter

                # Passage-number answer
                if answer_number in passage_number_values:
                    answer_program_start_types.append("passage_number")
                    ans_as_passage_number_idx = passage_number_values.index(answer_number)
                    ans_as_passage_number[ans_as_passage_number_idx] = 1

                # Year-difference answer
                if answer_number in year_differences:
                    answer_program_start_types.append("year_difference")
                    ans_as_year_difference_idx = year_differences.index(answer_number)
                    ans_as_year_difference[ans_as_year_difference_idx] = 1

                # PassageNum-difference Answer
                if answer_number in passage_number_differences:
                    answer_program_start_types.append("passagenum_diff")
                    ans_as_passagenum_diff_idx = passage_number_differences.index(answer_number)
                    answer_as_passagenum_difference[ans_as_passagenum_diff_idx] = 1

                # Count answer
                if answer_number in count_values:
                    answer_program_start_types.append("count_number")
                    answer_count_idx = count_values.index(answer_number)
                    answer_as_count[answer_count_idx] = 1


            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_year_difference"] = MetadataField(ans_as_year_difference)
            fields["answer_as_passagenum_difference"] = MetadataField(answer_as_passagenum_difference)
            fields["answer_as_count"] = MetadataField(answer_as_count)

            fields["answer_program_start_types"] = MetadataField(answer_program_start_types)

            # if len(answer_program_start_types) == 0:
            #     print(original_ques_text)
            #     print(original_passage_text)
            #     print(answer_annotation)
            #     print(f"PassageNumVals:{passage_number_values}")
            #     print(f"PassageDates:{passage_date_strvals}")
            #     print(f"PassageNumDiffs: {passage_number_differences}")
            #     print(f"YearDiffs:{year_differences}")

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

        '''
        attention, count_answer, mask = self.make_count_instance(passage_text.split(' '))
        attention = [x + abs(random.gauss(0, 0.001)) for x in attention]
        attention_sum = sum(attention)
        attention = [float(x) / attention_sum for x in attention]
        count_answer_vec = [0] * 10
        count_answer_vec[count_answer] = 1
        fields["aux_passage_attention"] = ArrayField(np.array(attention), padding_value=0.0)
        fields["aux_answer_as_count"] = ArrayField(np.array(count_answer_vec))
        fields["aux_count_mask"] = ArrayField(np.array(mask))
        '''

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


    def prune_for_passage_len(self,
                              max_passage_len: int,
                              p_date_mens, p_date_entidxs, p_date_normvals,
                              p_num_mens, p_num_entidxs, p_num_normvals,
                              answer_passage_spans,
                              date_grounding_supervision,
                              num_grounding_supervision):

        """ Prunes the passage and related data for a maximum length

            For the given max_passage_len, we first need to find out the pruned date and number mentions
            Since these might remove some dates and numbers from the passage, we need to find the
            pruned list of p_date_normvals (p_date_entidxs with the new date_entidxs)
            pruned list of p_num_normvals (p_num_entidxs with new num_entidxs) -- make sure the numbers are still sorted

            answer_passage_spans - only spans that are contained in the pruned passage

            date_grounding_supervision, num_grounding_supervision -- both these are the length of original dates/nums
            we need to find the new value by pruning and mapping old ent idxs to new ones.
        """
        pruned_date_mens = []       # New passage date mens
        pruned_old_dateidxs = []
        for date_men, date_idx in zip(p_date_mens, p_date_entidxs):
            _, (x,y), _ = date_men
            if y < max_passage_len:
                pruned_date_mens.append(date_men)
                pruned_old_dateidxs.append(date_idx)

        new_date_values = []        # New passage date values
        new2old_dateidx = {}
        old2new_dateidx = {}
        for old_date_idx, date_value in enumerate(p_date_normvals):
            # Atleast one mention of this old_date_idx remains
            if old_date_idx in pruned_old_dateidxs:
                new_date_idx = len(new_date_values)
                new2old_dateidx[new_date_idx] = old_date_idx
                old2new_dateidx[old_date_idx] = new_date_idx
                new_date_values.append(date_value)

        new_date_entidxs = [old2new_dateidx[x] for x in pruned_old_dateidxs]      # New passage date entidxs

        if date_grounding_supervision is not None:
            new_dategrounding_supervision = []
            for date_grounding in date_grounding_supervision:
                new_grounding = [date_grounding[new2old_dateidx[newidx]] for newidx in range(len(new_date_values))]
                new_dategrounding_supervision.append(new_grounding)
        else:
            new_dategrounding_supervision = None

        # Pruning numbers
        pruned_num_mens, pruned_old_numidxs = [], []
        for num_men, num_idx in zip(p_num_mens, p_num_entidxs):
            _, tokenidx, _ = num_men
            if tokenidx < max_passage_len:
                pruned_num_mens.append(num_men)
                pruned_old_numidxs.append(num_idx)
        new_num_values = []
        old2new_numidx, new2old_numidx = {}, {}
        for old_num_idx, num_value in enumerate(p_num_normvals):
            if old_num_idx in pruned_old_numidxs:
                new_num_idx = len(new_num_values)
                old2new_numidx[old_num_idx] = new_num_idx
                new2old_numidx[new_num_idx] = old_num_idx
                new_num_values.append(num_value)
        new_num_idxs = [old2new_numidx[x] for x in pruned_old_numidxs]

        if num_grounding_supervision is not None:
            new_numgrounding_supervision = []
            for num_grounding in num_grounding_supervision:
                new_grounding = [num_grounding[new2old_numidx[newidx]] for newidx in range(len(new_num_values))]
                new_numgrounding_supervision.append(new_grounding)
        else:
            new_numgrounding_supervision = None

        new_answer_passage_spans = [span for span in answer_passage_spans if span[1] < max_passage_len]

        return (pruned_date_mens, new_date_entidxs, new_date_values,
                pruned_num_mens, new_num_idxs, new_num_values,
                new_answer_passage_spans,
                new_dategrounding_supervision, new_numgrounding_supervision)

    def prune_for_question_len(self, max_question_len, answer_question_spans, ques_attn_supervision):
        new_answer_question_spans = [span for span in answer_question_spans if span[1] < max_question_len]

        if ques_attn_supervision is not None:
            new_qattn_supervision = [qattn[0:max_question_len] for qattn in ques_attn_supervision]
        else:
            new_qattn_supervision = None

        return (new_answer_question_spans, new_qattn_supervision)

    '''
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
    '''


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
    def get_passagenumber_difference_candidates(passage_num_values: List[float]) -> Tuple[List[float], np.array]:
        """ List of numbers indicating all-possible non-negative subtractions between the passage-numbers

            Parameters:
            -----------
            passage_num_values: List[float]
                Sorted list of passage numbers
            Returns:
            --------
            passage_number_differences:
                List[float] These are the possible differences between passage numbers.
            passagenumber_difference_mat: Binary np.array of shape (PN, PN, d)
                Entry (i, j, k) == 1 denotes that PN[i] - PN[j] == passage_number_differences[k]
        """
        num_passage_numbers = len(passage_num_values)
        # Adding zero-first since it'll definitely be added and makes sanity-checking easy
        passage_number_differences: List[int] = [0]

        for (num1, num2) in itertools.product(passage_num_values, repeat=2):
            number_diff = num1 - num2
            if number_diff >= 0:
                if number_diff not in passage_number_differences:
                    passage_number_differences.append(number_diff)

        num_of_passagenum_differences = len(passage_number_differences)
        # Making year_difference_mat
        passage_number_diff_mat = np.zeros(shape=(num_passage_numbers, num_passage_numbers,
                                                  num_of_passagenum_differences),
                                       dtype=int)
        for ((num_idx1, num1), (num_idx2, num2)) in itertools.product(enumerate(passage_num_values), repeat=2):
            number_diff = num1 - num2
            if number_diff >= 0:
                num_diff_idx = passage_number_differences.index(number_diff)  # We know this will not fail
                passage_number_diff_mat[num_idx1, num_idx2, num_diff_idx] = 1

        return passage_number_differences, passage_number_diff_mat



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
                             program_supervised: bool,
                             qtype: str,
                             question_tokens: List[str],
                             language: DropLanguage,
                             action2idx_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]], bool]:

        qtype_to_lffunc = {constants.DATECOMP_QTYPE: self.datecomp_logicalforms,
                           constants.NUMCOMP_QTYPE: self.numcomp_logicalforms,
                           constants.YARDS_longest_qtype: self.yardslongest_logicalforms,
                           constants.YARDS_shortest_qtype: self.yardsshortest_logicalforms,
                           constants.YARDS_findnum_qtype: self.findnum_logicalforms,
                           constants.DIFF_MAXMIN_qtype: self.numdiff_logicalforms,
                           constants.DIFF_MAXNUM_qtype: self.numdiff_logicalforms,
                           constants.DIFF_MAXMAX_qtype: self.numdiff_logicalforms,
                           constants.DIFF_NUMMAX_qtype: self.numdiff_logicalforms,
                           constants.DIFF_NUMMIN_qtype: self.numdiff_logicalforms,
                           constants.DIFF_NUMNUM_qtype: self.numdiff_logicalforms,
                           constants.DIFF_MINMAX_qtype: self.numdiff_logicalforms,
                           constants.DIFF_MINNUM_qtype: self.numdiff_logicalforms,
                           constants.DIFF_MINMIN_qtype: self.numdiff_logicalforms,
                           constants.COUNT_qtype: self.count_logicalforms}

        gold_actionseq_idxs: List[List[int]] = []
        gold_actionseq_mask: List[List[int]] = []

        if not program_supervised:
            gold_actionseq_idxs.append([0])
            gold_actionseq_mask.append([0])
            return gold_actionseq_idxs, gold_actionseq_mask, program_supervised

        if qtype in qtype_to_lffunc:
            gold_logical_forms: List[str] = qtype_to_lffunc[qtype](question_tokens=question_tokens,
                                                                   language=language,
                                                                   qtype=qtype)
            assert len(gold_logical_forms) >= 1, f"No logical forms found for: {question_tokens}"
            for logical_form in gold_logical_forms:
                gold_actions: List[str] = language.logical_form_to_action_sequence(logical_form)
                actionseq_idxs: List[int] = [action2idx_map[a] for a in gold_actions]
                actionseq_mask: List[int] = [1 for _ in range(len(actionseq_idxs))]
                gold_actionseq_idxs.append(actionseq_idxs)
                gold_actionseq_mask.append(actionseq_mask)
        else:
            program_supervised = False
            gold_actionseq_idxs.append([0])
            gold_actionseq_mask.append([0])
            logger.error(f"Tried get gold logical form for: {qtype}")

        return gold_actionseq_idxs, gold_actionseq_mask, program_supervised

    @staticmethod
    def findnum_logicalforms(**kwargs):
        gold_lf = "(find_PassageNumber find_PassageAttention)"
        return [gold_lf]

    @staticmethod
    def count_logicalforms(**kwargs):
        gold_lf = "(passageAttn2Count find_PassageAttention)"
        return [gold_lf]


    @staticmethod
    def numdiff_logicalforms(**kwargs):
        qtype = kwargs['qtype']
        # Qtype of form: diff_maxmin_qtype
        numtypes = qtype.split('_')[1]
        first_num = numtypes[0:3] # first 3 chars
        second_num = numtypes[3:6]  # last 3 chars

        max_num_program = "(max_PassageNumber (find_PassageNumber find_PassageAttention))"
        min_num_program = "(min_PassageNumber (find_PassageNumber find_PassageAttention))"
        find_num_program = "(find_PassageNumber find_PassageAttention)"

        if first_num == 'max':
            first_num_prog = max_num_program
        elif first_num == 'min':
            first_num_prog = min_num_program
        elif first_num == 'num':
            first_num_prog = find_num_program
        else:
            raise NotImplementedError

        if second_num == 'max':
            second_num_prog = max_num_program
        elif second_num == 'min':
            second_num_prog = min_num_program
        elif second_num == 'num':
            second_num_prog = find_num_program
        else:
            raise NotImplementedError

        # "(passagenumber_difference first_num_prog second_num_program)"
        gold_lf = f"(passagenumber_difference {first_num_prog} {second_num_prog})"

        return [gold_lf]


    @staticmethod
    def yardsshortest_logicalforms(**kwargs):
        gold_lf = "(min_PassageNumber (find_PassageNumber find_PassageAttention))"
        return [gold_lf]

    @staticmethod
    def yardslongest_logicalforms(**kwargs):
        gold_lf = "(max_PassageNumber (find_PassageNumber find_PassageAttention))"
        return [gold_lf]


    @staticmethod
    def datecomp_logicalforms(**kwargs) -> List[str]:
        question_tokens: List[str] = kwargs['question_tokens']
        language: DropLanguage = kwargs['language']
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
    def numcomp_logicalforms(**kwargs) -> List[str]:
        question_tokens: List[str] = kwargs['question_tokens']
        language: DropLanguage = kwargs['language']
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



