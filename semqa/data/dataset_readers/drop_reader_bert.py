import json
import logging
import itertools
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Set, Any
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from transformers.tokenization_auto import AutoTokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, MetadataField, ListField, SpanField, ArrayField

from allennlp_semparse.fields import ProductionRuleField

from semqa.domain_languages.drop_language import DropLanguage, Date, get_empty_language_object
from semqa.utils.qdmr_utils import Node, node_from_dict, nested_expression_to_lisp, \
    get_domainlang_function2returntype_mapping, get_inorder_function_list, function_to_action_string_alignment
from datasets.drop import constants
from utils.util import round_all

from semqa.data.dataset_readers.utils import find_valid_spans, index_text_to_tokens, \
    extract_answer_info_from_annotation, split_tokens_by_hyphen, BIOAnswerGenerator, get_single_answer_span_fields


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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


def tokenize_bert(bert_tokenizer: PretrainedTransformerTokenizer, tokens: List[str]):
    """Word-piece tokenize input tokens.

    Returns:
        wordpiece_tokens: List[str] word-pieces
        tokenidx2wpidxs: List[List[int]] For each original-tokenidx, a list of indices for its corresponding wps
        wpidx2tokenidx: List[int] Same length as wordpiece_tokens; index of original

    """
    tokenidx2wpidxs: List[List[int]] = []
    wpidx2tokenidx: List[int] = []
    wordpiece_tokens = []
    for tokenidx, token in enumerate(tokens):
        wp_idx = len(wordpiece_tokens)
        wps = bert_tokenizer.tokenize(token)    # this can be empty, if token = " "
        wordpiece_tokens.extend([t.text for t in wps])
        wpidx2tokenidx.extend([tokenidx] * len(wps))
        wp_idxs = [wp_idx + i for i, _ in enumerate(wps)]   # This can be empty,
        tokenidx2wpidxs.append(wp_idxs)     # and hence, entries in this can be empty
    assert len(wordpiece_tokens) == len(wpidx2tokenidx)
    assert len(tokens) == len(tokenidx2wpidxs)

    return wordpiece_tokens, tokenidx2wpidxs, wpidx2tokenidx


def process_spans(max_token_len: int, spans: List[Tuple[int, int]] = None):
    if spans is not None:
        pruned_spans = [span for span in spans if span[1] < max_token_len]
        return pruned_spans
    else:
        return None

def spans_into_listofspanfields(spans: List[Tuple[int, int]], tokenidx2wpidx: List[List[int]], seq_wps_len: int,
                                field: TextField, end_exclusive=False) -> Tuple[ListField, bool]:
    """Convert a list of spans (start_token_idx, end_token_idx) (both inclusive) into a list of span-fields, where
       the spans are indexed by word-pieces.

       seq_wps_len is the len of the sequence in word-pieces into which we are indexing.

    """
    if spans is None:
        return ListField([SpanField(-1, -1, field)]), False

    span_fields = []
    spans_found = False
    for (start_token_idx, end_token_idx) in spans:
        end_token_idx = end_token_idx if not end_exclusive else end_token_idx - 1
        if start_token_idx < len(tokenidx2wpidx):
            if not tokenidx2wpidx[start_token_idx]:  # This token didn't get any wordpieces
                start_token_idx += 1
            start_wp_idx = tokenidx2wpidx[start_token_idx][0]
            if start_wp_idx < seq_wps_len and end_token_idx < len(tokenidx2wpidx):
                if not tokenidx2wpidx[end_token_idx]:   # This token didn't get any wordpieces
                    end_token_idx -= 1
                end_wp_idx = min(tokenidx2wpidx[end_token_idx][-1], seq_wps_len - 1)
                span_fields.append(SpanField(start_wp_idx, end_wp_idx, field))
                spans_found = True
    if not span_fields:
        span_fields.append(SpanField(-1, -1, field))

    return ListField(span_fields), spans_found


def process_date_mentions(max_token_len: int, date_mens, date_entidxs, date_normvals):
    pruned_date_mens = []  # New passage date mens
    pruned_old_dateidxs = []
    for date_men, date_idx in zip(date_mens, date_entidxs):
        _, (x, y), _ = date_men
        if y < max_token_len:
            pruned_date_mens.append(date_men)
            pruned_old_dateidxs.append(date_idx)

    new_date_values = []  # New passage date values
    new2old_dateidx = {}
    old2new_dateidx = {}
    for old_date_idx, date_value in enumerate(date_normvals):
        # Atleast one mention of this old_date_idx remains
        if old_date_idx in pruned_old_dateidxs:
            new_date_idx = len(new_date_values)
            new2old_dateidx[new_date_idx] = old_date_idx
            old2new_dateidx[old_date_idx] = new_date_idx
            new_date_values.append(date_value)

    new_date_entidxs = [old2new_dateidx[x] for x in pruned_old_dateidxs]  # New passage date entidxs

    return pruned_date_mens, new_date_entidxs, new_date_values, old2new_dateidx, new2old_dateidx


def process_num_mentions(max_token_len: int, num_mens, num_entidxs, num_normvals):
    """ Pruned number mentions so that mentions that appear withing token_len remain.

    Since mentions are arranged so that underlying-values are sorted in ascending-order, the resulting mentions
    also need to follow this order. It can be done if mentions are processed in order and removed if they appear
    after truncation.
    """
    # Pruning numbers
    pruned_num_mens, pruned_old_numidxs = [], []
    for num_men, num_idx in zip(num_mens, num_entidxs):
        _, tokenidx, _ = num_men
        if tokenidx < max_token_len:
            pruned_num_mens.append(num_men)
            pruned_old_numidxs.append(num_idx)
    new_num_values = []
    old2new_numidx, new2old_numidx = {}, {}
    for old_num_idx, num_value in enumerate(num_normvals):
        if old_num_idx in pruned_old_numidxs:
            new_num_idx = len(new_num_values)
            old2new_numidx[old_num_idx] = new_num_idx
            new2old_numidx[new_num_idx] = old_num_idx
            new_num_values.append(num_value)
    new_num_idxs = [old2new_numidx[x] for x in pruned_old_numidxs]

    return pruned_num_mens, new_num_idxs, new_num_values, old2new_numidx, new2old_numidx


@DatasetReader.register("drop_reader_bert")
class DROPReader(DatasetReader):
    def __init__(
            self,
            lazy: bool = True,
            tokenizer: PretrainedTransformerTokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            relaxed_span_match: bool = True,
            max_question_wps: int = 50,
            max_transformer_length: int = 512,
            bio_tagging: bool = False,
            bio_label_scheme: str = "IO",
            shared_substructure: bool = False,
            only_strongly_supervised: bool = False,
            skip_instances: bool = False,
            skip_if_progtype_mismatch_anstype: bool = False,
            convert_spananswer_to_num=False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer: PretrainedTransformerTokenizer = tokenizer
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers
        self._relaxed_span_match = relaxed_span_match
        self.max_question_wps = max_question_wps
        self.max_transformer_length = max_transformer_length
        self.only_strongly_supervised = only_strongly_supervised
        self.convert_spananswer_to_num = convert_spananswer_to_num

        self.skip_instances: bool = skip_instances
        self.num_program_supervised = 0
        # If instance has program-supervision but none of the program-return-types match the answer-supervision-type,
        # we have two option (a) skip the instance (b) remove program supervision and train with only answer supervision
        # True here applies (a). If skip_instances is False, this option doesn't matter
        self.skip_if_progtype_mismatch_anstype: bool = skip_if_progtype_mismatch_anstype and skip_instances

        self.skip_count = 0
        self.num_goldans_not_in_anssupport = 0
        self.num_supprogtype_mismatch_anstype = 0
        self.num_supervised_programs = 0
        # mapping from DROPLanguage functions to their return_types - used for getting return_type of prog-supervision
        self.function2returntype_mapping = get_domainlang_function2returntype_mapping(get_empty_language_object())

        self.spacy_tokenizer = SpacyTokenizer()  # Used to tokenize answer-texts
        self.bio_tagging: bool = bio_tagging
        self.bio_label_scheme: str = bio_label_scheme
        if self.bio_tagging:
            if self.bio_label_scheme == "BIO":
                labels = {'O': 0, 'B': 1, 'I': 2}
            elif self.bio_label_scheme == "IO":
                labels = {'O': 0, 'I': 1}
            else:
                raise Exception("bio_label_scheme not supported: {}".format(self.bio_label_scheme))
            self.bio_answer_generator = BIOAnswerGenerator(ignore_question=True,
                                                           flexibility_threshold=1000,
                                                           labels=labels)

        # Parse and make fields for shared-substructure supervision for certain questions
        self.shared_substructure = shared_substructure
        self.num_w_ss = 0

        self.max_passage_nums = 0
        self.max_composed_nums = 0

        self.max_num_instances = -1     # -1 to deactivate

    @overrides
    def _read(self, file_path: str):
        self.skip_count = 0
        # pylint: disable=logging-fstring-interpolation
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        instances, skip_count = [], 0
        total_qas = 0
        instances_read = 0
        for passage_id, passage_info in dataset.items():
            passage = passage_info[constants.passage]
            passage_charidxs = passage_info[constants.passage_charidxs]
            passage_tokens = passage_info[constants.passage_tokens]
            p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = passage_info[
                constants.passage_date_mens
            ]
            p_date_entidxs: List[int] = passage_info[constants.passage_date_entidx]
            p_date_normvals: List[Tuple[int, int, int]] = passage_info[constants.passage_date_normalized_values]

            p_num_mens: List[Tuple[str, int, int]] = passage_info[constants.passage_num_mens]
            p_num_entidxs: List[int] = passage_info[constants.passage_num_entidx]
            p_num_normvals: List[int] = passage_info[constants.passage_num_normalized_values]

            # start / end (exclusive) token offsets for sentences in the passage
            p_sent_boundaries: List[Tuple[int, int]] = passage_info[constants.passage_sent_idxs]

            for qa in passage_info[constants.qa_pairs]:
                total_qas += 1
                question_id = qa[constants.query_id]
                question = qa[constants.question]
                question_tokens = qa[constants.question_tokens]
                question_charidxs = qa[constants.question_charidxs]

                q_num_mens: List[Tuple[str, int, int]] = qa[constants.q_num_mens]
                q_num_entidxs: List[int] = qa[constants.q_num_entidx]
                q_num_normvals: List[int] = qa[constants.q_num_normalized_values]

                answer_passage_spans = qa.get(constants.answer_passage_spans, None)
                answer_question_spans = qa.get(constants.answer_question_spans, None)
                answer_annotations = []
                if "answer" in qa:
                    answer_annotations.append(qa["answer"])
                if "validated_answers" in qa:
                    answer_annotations += qa["validated_answers"]

                program_supervision = qa.get(constants.program_supervision, None)
                program_supervised = True if program_supervision is not None else False
                execution_supervised = qa.get(constants.execution_supervised, False)
                strongly_supervised = program_supervised and execution_supervised

                # shared substructure annotations
                shared_substructure_annotations = None
                if self.shared_substructure and constants.shared_substructure_annotations in qa:
                    shared_substructure_annotations: List[Dict] = qa[constants.shared_substructure_annotations]


                instance = self.text_to_instance(
                    question,
                    question_charidxs,
                    question_tokens,
                    passage,
                    passage_charidxs,
                    passage_tokens,
                    p_sent_boundaries,
                    p_date_mens,
                    p_date_entidxs,
                    p_date_normvals,
                    p_num_mens,
                    p_num_entidxs,
                    p_num_normvals,
                    q_num_mens,
                    q_num_entidxs,
                    q_num_normvals,
                    program_supervised,
                    program_supervision,
                    execution_supervised,
                    strongly_supervised,
                    shared_substructure_annotations,
                    answer_passage_spans,
                    answer_question_spans,
                    question_id,
                    passage_id,
                    answer_annotations,
                )

                if self.only_strongly_supervised:
                    if not strongly_supervised:
                        instance = None

                if instance is not None:
                    instances_read += 1
                    if self.max_num_instances != -1 and instances_read > self.max_num_instances:
                        break
                    yield instance
            if self.max_num_instances != -1 and instances_read > self.max_num_instances:
                break

        logger.info(f"Total QAs: {total_qas}. Instances made: {instances_read}  (skipped: {self.skip_count})")
        logger.info(f"num program supervised: {self.num_supervised_programs}")
        logger.info(f"gold-ans not in ans support: {self.num_goldans_not_in_anssupport}")
        logger.info(f"supervised-program-type mismatches answer-type(s): {self.num_supprogtype_mismatch_anstype}")
        logger.info("Max passage nums: {} Max composed nums : {} ".format(self.max_passage_nums,
                                                                          self.max_composed_nums))
        logger.info("Instances w/ shared-substructure annotation: {}".format(self.num_w_ss))

    @overrides
    def text_to_instance(
            self,
            question: str,
            question_charidxs: List[int],
            question_tokens: List[str],
            passage: str,
            passage_charidxs: List[int],
            passage_tokens: List[str],
            p_sent_boundaries: List[Tuple[int, int]],
            p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
            p_date_entidxs: List[int],
            p_date_normvals: List[Tuple[int, int, int]],
            p_num_mens: List[Tuple[str, int, int]],
            p_num_entidxs: List[int],
            p_num_normvals: List[int],
            q_num_mens: List[Tuple[str, int, int]],
            q_num_entidxs: List[int],
            q_num_normvals: List[int],
            program_supervised: bool,
            program_supervision: Dict,
            execution_supervised: bool,
            strongly_supervised: bool,
            shared_substructure_annotations: Union[None, List[Dict]],
            answer_passage_spans: List[Tuple[int, int]],
            answer_question_spans: List[Tuple[int, int]],
            question_id: str = None,
            passage_id: str = None,
            answer_annotations: List[Dict[str, Union[str, Dict, List]]] = None,
    ) -> Union[Instance, None]:

        metadata = {
            "passage": passage,
            "question": question,
            "passage_id": passage_id,
            "question_id": question_id,
        }

        language = get_empty_language_object()

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        fields = {}
        fields["actions"] = action_field

        # Word-piece tokenize the tokens in Q and P. Get maps;
        #  tokenidx2wpidx: List[List[int]] map from token idx to list of word-piece indices that correspond to it
        #  wpidx2tokenidx: List[int] map from word-piece index to original token idx
        # List[str], List[List[int]], List[int]
        passage_wps, p_tokenidx2wpidx, p_wpidx2tokenidx = tokenize_bert(self._tokenizer, passage_tokens)
        question_wps, q_tokenidx2wpidx, q_wpidx2tokenidx = tokenize_bert(self._tokenizer, question_tokens)

        # Truncate question_wps and wpidx2tokenidx to maximum allowable length
        question_wps = question_wps[0:self.max_question_wps]
        q_wpidx2tokenidx = q_wpidx2tokenidx[0:self.max_question_wps]
        unpadded_q_wps_len = len(question_wps)
        last_q_tokenidx = q_wpidx2tokenidx[-1]  # The last token_idx after truncation
        q_token_len = last_q_tokenidx + 1
        # NOTE: Last token's all word-pieces might not be included; take precaution later
        q_tokenidx2wpidx = q_tokenidx2wpidx[0:q_token_len]

        hf_tokenizer = self._tokenizer.tokenizer  # HuggingFace tokenizer / BertTokenizerFast
        pad_token_str, pad_token_id = hf_tokenizer.pad_token, hf_tokenizer.pad_token_id
        cls_token_str, cls_token_id = hf_tokenizer.cls_token, hf_tokenizer.cls_token_id
        sep_token_str, sep_token_id = hf_tokenizer.sep_token, hf_tokenizer.sep_token_id

        # Padding question to max length since it makes possible to separate question and passage in the model
        ques_padding_len = self.max_question_wps - len(question_wps)
        question_wps.extend([pad_token_str] * ques_padding_len)
        q_wpidx2tokenidx.extend([-1] * ques_padding_len)

        question_wps_tokens: List[Token] = [Token(text=t, text_id=hf_tokenizer.convert_tokens_to_ids(t))
                                            for t in question_wps]
        # question_wps_tokens = [Token(text=wp_text, text_id=wp_id)
        #                        for wp_text, wp_id in zip(question_wps,
        #                                                  self._tokenizer.tokenizer.convert_tokens_to_ids(question_wps))]

        # Passage_len = Max seq len - CLS - SEP - SEP - Max_Qlen -- Questions will be padded to max length
        max_passage_wps = self.max_transformer_length - 3 - self.max_question_wps
        passage_wps = passage_wps[0:max_passage_wps]
        p_wpidx2tokenidx = p_wpidx2tokenidx[0:max_passage_wps]
        last_p_tokenidx = p_wpidx2tokenidx[-1]
        p_token_len = last_p_tokenidx + 1
        p_tokenidx2wpidx = p_tokenidx2wpidx[0:p_token_len]

        # passage_wps.append(sep_token_str)
        # p_wpidx2tokenidx.append(-1)

        passage_wps_tokens: List[Token] = [Token(text=t, text_id=hf_tokenizer.convert_tokens_to_ids(t))
                                           for t in passage_wps]
        # passage_wps_tokens = [Token(text=wp_text, text_id=wp_id)
        #                       for wp_text, wp_id in zip(passage_wps,
        #                                                 self._tokenizer.tokenizer.convert_tokens_to_ids(passage_wps))]

        passage_wps_len = len(passage_wps)

        cls_token = Token(cls_token_str, text_id=cls_token_id)
        sep_token = Token(sep_token_str, text_id=sep_token_id)

        # This would be in the input to BERT
        question_passage_tokens: List[Token] = [cls_token] + question_wps_tokens + [sep_token] + \
                                                passage_wps_tokens + [sep_token]
        # question_passage_tokens = self._tokenizer.add_special_tokens(question_wps_tokens, passage_wps_tokens)


        fields["question"] = TextField([cls_token] + question_wps_tokens + [sep_token], self._token_indexers)
        fields["passage"] = TextField([cls_token] + passage_wps_tokens + [sep_token], self._token_indexers)
        fields["question_passage"] = TextField(question_passage_tokens, self._token_indexers)
        p_sentboundary_wps_field, _ = spans_into_listofspanfields(
            spans=p_sent_boundaries, tokenidx2wpidx=p_tokenidx2wpidx,
            seq_wps_len=passage_wps_len, field=fields["passage"], end_exclusive=True)
        fields["p_sentboundary_wps"] = p_sentboundary_wps_field

        # Now need to take care of metadata due to passage length truncation
        # 1. Date and Number mentions - remove mentions that exceed passage length, recompute mention2entidx and values
        #     list, keeping mapping from old-ent-idx to new-ent-idx to re-map execution supervision
        # 2. Answers as question and passage spans

        (p_date_mens, p_date_entidxs, p_date_normvals,
         old2new_p_dateidx, new2old_p_dateidx) = process_date_mentions(p_token_len, p_date_mens, p_date_entidxs,
                                                                       p_date_normvals)
        (p_num_mens, p_num_entidxs, p_num_normvals,
         old2new_p_numidx, new2old_p_numidx) = process_num_mentions(p_token_len, p_num_mens, p_num_entidxs,
                                                                    p_num_normvals)
        (q_num_mens, q_num_entidxs, q_num_normvals,
         old2new_q_numidx, new2old_q_numidx) = process_num_mentions(q_token_len, q_num_mens, q_num_entidxs,
                                                                    q_num_normvals)
        # List of (start, end) char offsets for each passage and question word-piece
        passage_wp_offsets: List[Tuple[int, int]] = self.update_charoffsets(
            wpidx2tokenidx=p_wpidx2tokenidx, tokens=passage_tokens, token_charidxs=passage_charidxs
        )
        question_wp_offsets: List[Tuple[int, int]] = self.update_charoffsets(
            wpidx2tokenidx=q_wpidx2tokenidx, tokens=question_tokens, token_charidxs=question_charidxs
        )

        # answer_passage_spans = process_spans(p_token_len, answer_passage_spans)
        # answer_question_spans = process_spans(q_token_len, answer_question_spans)

        # Passage Number
        passage_number_values = [int(x) if int(x) == x else x for x in p_num_normvals]
        nums_from_passage = set(passage_number_values)
        # composed_numbers: List[int/float] is sorted
        # passage_number_values: now contains implicit numbers. Since they are added at the end,
        #  indexing should be fine.
        # compnumber2addcombinations (sub): Dict: {composed_number: List[(pass-num1, pass-num2)]} - mapping from
        #  composed number to list of passage-num-tuples that combine to form the number-key using the operation
        (composed_numbers, passage_number_values,
         compnumber2addcombinations, compnumber2subcombinations,
         nums_from_addition, nums_from_subtraction) = self.compute_number_support(
            numbers=passage_number_values,
            implicit_numbers=DropLanguage.implicit_numbers,
            max_number_of_numbers_to_consider=2,
        )
        composed_numbers = round_all(composed_numbers, 5)

        self.max_passage_nums = max(len(passage_number_values), self.max_passage_nums)
        self.max_composed_nums = max(len(composed_numbers), self.max_composed_nums)
        if not passage_number_values:
            passage_number_values = [0]
        if not composed_numbers:
            composed_numbers = [0]
        # TODO(nitishg): Change this repr to (token_idx, value) repr.
        passage_number_entidxs = p_num_entidxs  # Index of passage_num_tokens in passage_number_values list
        passage_number_tokenids = [tokenidx for (_, tokenidx, _) in p_num_mens]
        assert len(passage_number_entidxs) == len(passage_number_tokenids)
        passage_num_wpidx2entidx = [-1 for _ in range(passage_wps_len)]  # number_ent_idx for each token (pad=-1)
        passage_number_wpindices = []  # wp_idxs that are numbers
        for passage_num_tokenidx, number_ent_idx in zip(passage_number_tokenids, passage_number_entidxs):
            wp_index = p_tokenidx2wpidx[passage_num_tokenidx][0]  # WP-idx of the number token
            passage_num_wpidx2entidx[wp_index] = number_ent_idx  # Index of the num-value in passsge_num_values
            passage_number_wpindices.append(wp_index)
        if not passage_number_wpindices:  # If no numbers in the passage, padding by faking the 0-th token as a number
            passage_num_wpidx2entidx[0] = 0
            passage_number_wpindices = [0]

        fields["passageidx2numberidx"] = ArrayField(np.array(passage_num_wpidx2entidx), padding_value=-1)
        fields["passage_number_values"] = MetadataField(passage_number_values)
        fields["composed_numbers"] = MetadataField(composed_numbers)
        fields["passage_number_sortedtokenidxs"] = MetadataField(passage_number_wpindices)   # already sorted

        # NP.array of shape: (size_of_number_support, max_num_combinations, 2)
        add_number_combinations_indices, max_num_add_combs = self.make_addsub_combination_array(
            composed_numbers=composed_numbers, passage_numbers=passage_number_values,
            compnumber2numcombinations=compnumber2addcombinations
        )
        sub_number_combinations_indices, max_num_sub_combs = self.make_addsub_combination_array(
            composed_numbers=composed_numbers, passage_numbers=passage_number_values,
            compnumber2numcombinations=compnumber2subcombinations
        )

        fields["add_number_combinations_indices"] = ArrayField(
            array=add_number_combinations_indices, padding_value=-1, dtype=np.int32
        )
        fields["sub_number_combinations_indices"] = ArrayField(
            array=sub_number_combinations_indices, padding_value=-1, dtype=np.int32
        )
        fields["max_num_add_combs"] = MetadataField(max_num_add_combs)
        fields["max_num_sub_combs"] = MetadataField(max_num_sub_combs)

        ##  Passage Dates
        passage_date_entidxs = p_date_entidxs
        passage_date_values = p_date_normvals
        passage_date_spanidxs: List[Tuple[int, int]] = []
        for (_, (start_token_idx, end_token_idx), _) in p_date_mens:
            start_wp_idx = p_tokenidx2wpidx[start_token_idx][0]
            end_wp_idx = min(p_tokenidx2wpidx[end_token_idx][-1], passage_wps_len - 1)
            passage_date_spanidxs.append((start_wp_idx, end_wp_idx))

        passage_date_idx2dateidx = [-1 for _ in range(passage_wps_len)]
        if passage_date_spanidxs:
            for passage_date_span, date_idx in zip(passage_date_spanidxs, passage_date_entidxs):
                (s, e) = passage_date_span
                passage_date_idx2dateidx[s: e + 1] = [date_idx] * (e + 1 - s)
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

        count_values = list(range(10))
        fields["count_values"] = MetadataField(count_values)

        metadata.update(
            {
                "passage_token_charidxs": passage_charidxs,
                "question_token_charidxs": question_charidxs,
                "passage_wp_offsets": passage_wp_offsets,
                "question_wp_offsets": question_wp_offsets,
                "passage_tokenidx2wpidx": p_tokenidx2wpidx,
                "question_tokenidx2wpidx": q_tokenidx2wpidx,
                "passage_wpidx2tokenidx": p_wpidx2tokenidx,
                "question_wpidx2tokenidx": q_wpidx2tokenidx,
                "unpadded_q_wps_len": unpadded_q_wps_len,
                "question_wps": question_wps,
                "passage_wps": passage_wps,
                "question_tokens": question_tokens,
                "passage_tokens": passage_tokens,
                "passage_date_values": passage_date_strvals,
                "composed_numbers": composed_numbers,
                "passage_number_values": passage_number_values,
                "passage_year_diffs": year_differences,
                "count_values": count_values,
            }
        )

        ########     ANSWER FIELDS      ###################
        # This list contains the possible-start-types for programs that can yield the correct answer
        # For example, if the answer is a number but also in passage, this will contain two keys
        # If the answer is a number, we'll find which kind and that program-start-type will be added here
        answer_program_start_types: List[str] = []
        if answer_annotations:
            metadata.update({"answer_annotations": answer_annotations})
            # Using the first one supervision (training actually only has one)
            answer_annotation = answer_annotations[0]

            spacy_passage_tokens: List[Token] = [Token(text=t, idx=idx)
                                                 for t, idx in zip(passage_tokens, passage_charidxs)]

            if self.bio_tagging:
                # "answer_as_list_of_bios": `List[LabelsField]` List of different BIO label seqs
                # "answer_as_text_to_disjoint_bios": `List[List[LabelsField]]` A list of dijoint BIO tags for each ans-text
                # "span_bio_labels": `LabelsField` BIO tags with all spans
                # "is_bio_mask": `LabelField` one of {0, 1} to indicate if span answers
                # span_answer_fields, has_passage_span_ans \
                (answer_spans_as_bios_field, bios_mask, answer_spans_for_possible_taggings_field,
                 all_spans, packed_gold_spans_list, _, has_passage_span_ans) = self.bio_answer_generator.get_bio_labels(
                    answer_annotation=answer_annotation,
                    passage_tokens=spacy_passage_tokens,
                    max_passage_len=p_token_len,
                    p_tokenidx2wpidx=p_tokenidx2wpidx,
                    passage_wps_len=passage_wps_len,
                    passage_field=fields["passage"])

                metadata.update({"answer_passage_spans": all_spans})
                span_answer_fields = {"passage_span_answer": answer_spans_as_bios_field,
                                      "passage_span_answer_mask": bios_mask,
                                      "answer_spans_for_possible_taggings": answer_spans_for_possible_taggings_field}

            else:
                (passage_span_answer_field, answer_spans_mask, answer_spans_for_possible_taggings_field,
                 answer_passage_spans, has_passage_span_ans) = get_single_answer_span_fields(
                    passage_tokens=spacy_passage_tokens,
                    max_passage_token_len=p_token_len,
                    answer_annotation=answer_annotation,
                    spacy_tokenizer=self.spacy_tokenizer,
                    passage_field=fields["passage"],
                    p_tokenidx2wpidx=p_tokenidx2wpidx)

                metadata.update({"answer_passage_spans": answer_passage_spans})
                span_answer_fields = {"passage_span_answer": passage_span_answer_field,
                                      "passage_span_answer_mask": answer_spans_mask,
                                      "answer_spans_for_possible_taggings": answer_spans_for_possible_taggings_field}


            fields.update(span_answer_fields)
            if has_passage_span_ans:
                answer_program_start_types.append("PassageSpanAnswer")

            # Possible_Start_Types -- PassageSpanAnswer, YearDifference, PassageNumber, ComposedNumber, CountNumber

            # (passage_span_fields,
            #  spans_found) = spans_into_listofspanfields(spans=answer_passage_spans, tokenidx2wpidx=p_tokenidx2wpidx,
            #                                             seq_wps_len=passage_wps_len, field=fields["passage"])
            # if spans_found:
            #     answer_program_start_types.append("PassageSpanAnswer")
            # fields["answer_as_passage_spans"] = passage_span_fields

            # Question-span answer
            question_span_fields = [SpanField(-1, -1, fields["question"])]
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            # Number answers
            number_answer_str = answer_annotation["number"]
            if not number_answer_str:
                # Answer as number string does not exist.
                if self.convert_spananswer_to_num:
                    # Try to convert "X" or "X-yard(s)" into number(X)
                    span_answer_text = None
                    try:
                        span_answer_text = answer_annotation["spans"][0]
                        span_answer_number = float(span_answer_text)
                    except:
                        span_answer_number = None
                    if span_answer_number is None and span_answer_text is not None:
                        split_hyphen = span_answer_text.split("-")
                        if len(split_hyphen) == 2:
                            try:
                                span_answer_number = float(split_hyphen[0])
                            except:
                                span_answer_number = None
                        else:
                            span_answer_number = None
                    if span_answer_number is not None:
                        answer_number = (
                            int(span_answer_number)
                            if int(span_answer_number) == span_answer_number
                            else span_answer_number
                        )
                    else:
                        answer_number = None
                else:
                    answer_number = None
            else:
                answer_number = float(number_answer_str)
                answer_number = int(answer_number) if int(answer_number) == answer_number else answer_number

            ans_as_passage_number = [0] * len(passage_number_values)
            ans_as_composed_number = [0] * len(composed_numbers)
            ans_as_year_difference = [0] * len(year_differences)
            answer_as_count = [0] * len(count_values)
            composed_num_ans_composition_types = set()
            if answer_number is not None:
                # Passage-number answer
                if answer_number in passage_number_values:
                    answer_program_start_types.append("PassageNumber")
                    ans_as_passage_number[passage_number_values.index(answer_number)] = 1
                # Composed-number answer
                if answer_number in composed_numbers:
                    answer_program_start_types.append("ComposedNumber")
                    ans_as_composed_number[composed_numbers.index(answer_number)] = 1
                    if answer_number in nums_from_addition:
                        composed_num_ans_composition_types.add("passage_num_addition")
                    if answer_number in nums_from_subtraction:
                        composed_num_ans_composition_types.add("passage_num_subtraction")
                    assert len(composed_num_ans_composition_types) != 0
                # Year-difference answer
                if answer_number in year_differences:
                    answer_program_start_types.append("YearDifference")
                    ans_as_year_difference[year_differences.index(answer_number)] = 1
                # Count answer
                if answer_number in count_values:
                    answer_program_start_types.append("CountNumber")
                    answer_as_count[count_values.index(answer_number)] = 1

            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_composed_number"] = MetadataField(ans_as_composed_number)
            fields["answer_as_year_difference"] = MetadataField(ans_as_year_difference)
            fields["answer_as_count"] = MetadataField(answer_as_count)
            fields["composed_num_ans_composition_types"] = MetadataField(composed_num_ans_composition_types)

        # End if answer-annotation
        if not answer_program_start_types:
            self.num_goldans_not_in_anssupport += 1
            if self.skip_instances:     # Answer not found in support. Skipping this instance
                self.skip_count += 1
                return None

        # Program supervision
        # Get gold action_seqs for strongly_supervised questions
        action2idx_map = {rule: i for i, rule in enumerate(language.all_possible_productions())}

        if program_supervision:
            program_node = node_from_dict(program_supervision)
            program_return_type = self.function2returntype_mapping[program_node.predicate]
            if program_return_type not in answer_program_start_types:
                self.num_supprogtype_mismatch_anstype += 1
                if self.skip_if_progtype_mismatch_anstype:
                    self.skip_count += 1
                    return None
                else:
                    program_supervision = None
            else:
                # If sup-program return type is in answer-types, reduce answer-types to sup-program-types
                # This is debatable though; if we want the model to explore programs that yield the answer of a
                # different type than the program supervisied, we shouldn't prune answer-prog-start-types.
                answer_program_start_types = [program_return_type]

        if program_supervision:
            program_node = node_from_dict(program_supervision)
            question_attention_supervision_to_wps(program_node, q_tokenidx2wpidx, unpadded_q_wps_len,
                                                  self.max_question_wps)
            revise_date_num_execution_supervision(program_node=program_node, old2new_dateidxs=old2new_p_dateidx,
                                                  old2new_numidxs=old2new_p_numidx)
            program_supervision = program_node.to_dict()


        # Tuple[List[List[int]], List[List[int]]]
        (
            gold_action_seqs,
            gold_actionseq_masks,
            gold_function2actionidx_map,
            gold_program_start_types,
            program_supervised,
        ) = self.get_gold_action_seqs(
            program_supervision=program_supervision,
            language=language,
            action2idx_map=action2idx_map)

        # FIELDS FOR STRONG-SUPERVISION
        fields["program_supervised"] = MetadataField(program_supervised)
        self.num_supervised_programs += 1 if program_supervised else 0
        execution_supervised = execution_supervised and program_supervised
        strongly_supervised = execution_supervised
        fields["execution_supervised"] = MetadataField(execution_supervised)
        fields["strongly_supervised"] = MetadataField(strongly_supervised)

        fields["gold_action_seqs"] = MetadataField((gold_action_seqs, gold_actionseq_masks))
        fields["gold_function2actionidx_maps"] = MetadataField(gold_function2actionidx_map)
        # wrapping in a list to support multiple program-supervisions
        fields["gold_program_dicts"] = MetadataField([program_supervision])
        metadata.update({"program_supervision": program_supervision})
        # This list is made here since it can get modfied due to program-supervision
        fields["answer_program_start_types"] = MetadataField(answer_program_start_types)

        if self.shared_substructure and shared_substructure_annotations is not None and program_supervision:
            # There might be plenty; we'll only use one
            shared_substruc_dict: Dict = shared_substructure_annotations[0]
            aux_question: str = shared_substruc_dict[constants.question]
            aux_question_tokens: List[str] = shared_substruc_dict[constants.question_tokens]
            aux_program_node: Node = node_from_dict(shared_substruc_dict[constants.program_supervision])
            aux_program_lisp = nested_expression_to_lisp(aux_program_node.get_nested_expression())
            orig_program_lisp: str = shared_substruc_dict["orig_program_lisp"]
            origprog_postorder_node_idx = shared_substruc_dict["origprog_postorder_node_idx"]
            sharedprog_postorder_node_idx = shared_substruc_dict["sharedprog_postorder_node_idx"]

            aux_question_wps, aux_q_tokenidx2wpidx, aux_q_wpidx2tokenidx = tokenize_bert(self._tokenizer,
                                                                                         aux_question_tokens)
            # Truncate question_wps and wpidx2tokenidx to maximum allowable length
            aux_question_wps = aux_question_wps[0:self.max_question_wps]
            aux_q_wpidx2tokenidx = aux_q_wpidx2tokenidx[0:self.max_question_wps]
            unpadded_aux_q_wps_len = len(aux_question_wps)

            aux_q_token_len = aux_q_wpidx2tokenidx[-1] + 1
            aux_q_tokenidx2wpidx = aux_q_tokenidx2wpidx[0:aux_q_token_len]

            ques_padding_len = self.max_question_wps - len(aux_question_wps)
            aux_question_wps.extend([pad_token_str] * ques_padding_len)
            aux_q_wpidx2tokenidx.extend([-1] * ques_padding_len)

            question_attention_supervision_to_wps(aux_program_node, aux_q_tokenidx2wpidx, unpadded_aux_q_wps_len,
                                                  self.max_question_wps)

            aux_question_wps_tokens: List[Token] = [Token(text=t, text_id=hf_tokenizer.convert_tokens_to_ids(t))
                                                    for t in aux_question_wps]

            aux_question_passage_tokens: List[Token] = [cls_token] + aux_question_wps_tokens + [sep_token] + \
                                                        passage_wps_tokens + [sep_token]

            # Making lists sice one question might have multiple aux-supervisions
            fields["sharedsub_question_passage"] = ListField([
                TextField(aux_question_passage_tokens, self._token_indexers)])
            fields["sharedsub_program_nodes"] = MetadataField([aux_program_node])
            fields["sharedsub_program_lisp"] = MetadataField([aux_program_lisp])
            fields["sharedsub_orig_program_lisp"] = MetadataField(orig_program_lisp)
            action_seq: List[str] = language.logical_form_to_action_sequence(aux_program_lisp)
            gold_function2actionidx_map: List[int] = function_to_action_string_alignment(aux_program_node, action_seq)
            fields["sharedsub_function2actionidx_maps"] = MetadataField([gold_function2actionidx_map])
            fields["orig_sharedsub_postorder_node_idx"] = MetadataField([(origprog_postorder_node_idx,
                                                                          sharedprog_postorder_node_idx)])
            fields["sharedsub_mask"] = ArrayField(np.array([1]), padding_value=0)
            self.num_w_ss += 1

        # elif not self.shared_substructure:
        else:
            # Make empty fields so that TextField gets padded appropriately
            aux_question_passage_tokens: List[Token] = [cls_token, sep_token, sep_token]
            fields["sharedsub_question_passage"] = ListField([
                TextField(aux_question_passage_tokens, self._token_indexers)])
            fields["sharedsub_program_nodes"] = MetadataField([None])
            fields["sharedsub_program_lisp"] = MetadataField([None])
            fields["sharedsub_orig_program_lisp"] = MetadataField(None)
            fields["sharedsub_function2actionidx_maps"] = MetadataField([None])
            fields["orig_sharedsub_postorder_node_idx"] = MetadataField([(-1, -1)])
            fields["sharedsub_mask"] = ArrayField(np.array([0]), padding_value=0)
        # else:
        #     return None


        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def compute_number_support(
            numbers: List[Union[int, float]],
            implicit_numbers: List[Union[int, float]] = None,
            max_number_of_numbers_to_consider: int = 2,
    ) -> Tuple[List[Union[int, float]], List[Union[int, float]], Dict, Dict, Set, Set]:
        """Compute the number support based on combinations of input numbers.
        This function considers all possible addition/subtraction between all pairs of numbers (even self). This forms
        the support of the possible answers. The output is a sorted list of number support.

        Args:
            numbers: input numbers -- usually passage numbers
            implicit_numbers: Extra numbers not part of the passage, but added in language. E.g. 100, 0
            max_number_of_numbers_to_consider: number of numbers to consider to combine
        Returns:
            composed_numbers: List of output composed numbers (also includes implicit numbers)
            compnumber2addcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
            compnumber2subcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
                Map from number to set of number combinations that can create it using the addition/sub operator.
                For example, {2: set((1,1), (0,2))} is a valid entry for addcombinations
        """
        if max_number_of_numbers_to_consider > 2:
            raise NotImplementedError

        passagenums_w_implicitnums = [x for x in numbers]
        # Adding implicit numbers here after checking if 0 is a part of original numbers so that we don't add tons of
        #  combinations of the kind x = x + 0 / x - 0
        zero_in_passage = True if 0 in numbers else False
        # Adding implicit-numbers to the input-numbers list since they can take part in composition with input-numbers.
        if implicit_numbers:
            passagenums_w_implicitnums.extend(implicit_numbers)

        composed_num_set = set()
        # Map from composed-number to list of number-combination that lead to this number from the add/sub operation
        compnumber2subcombinations = defaultdict(set)
        compnumber2addcombinations = defaultdict(set)
        nums_from_addition = set()
        nums_from_subtraction = set()
        signs = [-1, 1]
        # all_sign_combinations = list(itertools.product(signs, repeat=2))
        # Since our modules will only perform num1-num2 / num1+num2. Computation like -num1+num2 would not be done
        all_sign_combinations = [(1.0, -1.0), (1.0, 1.0)]
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            # for number_combination in itertools.combinations(numbers, r=number_of_numbers_to_consider):
            for indexed_number_combination in itertools.product(
                    enumerate(passagenums_w_implicitnums), repeat=number_of_numbers_to_consider
            ):
                ((idx1, num1), (idx2, num2)) = indexed_number_combination
                number_combination = (num1, num2)
                # if idx1 == idx2: continue     # Commented: 0 in support. Un-commented: 0 not in support
                # print(indexed_number_combination)
                for sign_combination in all_sign_combinations:
                    value = sum([sign * num for (sign, num) in zip(sign_combination, number_combination)])
                    value = round_all(value, 5)
                    if value >= 0:
                        # If 0 was originally in numbers then allow its combinations, o/w don't to avoid the
                        # combinations from getting bloated with x = x+0, 0+x, x-0
                        if (0 in number_combination and zero_in_passage) or (0 not in number_combination):
                            composed_num_set.add(value)
                            if sign_combination == (1, 1):
                                compnumber2addcombinations[value].add(number_combination)
                                nums_from_addition.add(value)
                            else:  # sign_combination == [1, -1]:
                                compnumber2subcombinations[value].add(number_combination)
                                nums_from_subtraction.add(value)

        composed_numbers = sorted(list(composed_num_set))

        return (composed_numbers, passagenums_w_implicitnums, compnumber2addcombinations, compnumber2subcombinations,
                nums_from_addition, nums_from_subtraction)

    @staticmethod
    def make_addsub_combination_array(
            composed_numbers: List[Union[int, float]], passage_numbers: List[Union[int, float]],
            compnumber2numcombinations: Dict[Union[int, float], List[Tuple]]
    ):
        """Make a (size_of_composed_numbers, max_num_combinations, 2) sized numpy array which would contain indices into
        the composed_numbers list.

        Each entry (i, :) will be a list of tuples, where (i, j)-th = x tuple would signifiy that
        composed_number[i] = passage_number[x[0] OP passage_number[x[1]]

        dim=1 will be padded to the max num of combinations possible for a number for this instance.
        Later on this will further be padded based on instances in the batch.
        """
        max_num_combinations = max(len(combinations) for (_, combinations) in compnumber2numcombinations.items())
        number_combinations_indices = -1 * np.ones(shape=(len(composed_numbers), max_num_combinations, 2),
                                                   dtype=np.int32)

        for composed_num, combinations in compnumber2numcombinations.items():
            compnumber_idx = composed_numbers.index(composed_num)
            for combination_num, (num1, num2) in enumerate(combinations):
                (passagenum1idx, passagenum2idx) = (passage_numbers.index(num1), passage_numbers.index(num2))
                number_combinations_indices[compnumber_idx, combination_num, :] = [passagenum1idx, passagenum2idx]
        return number_combinations_indices, max_num_combinations

    @staticmethod
    def update_charoffsets(wpidx2tokenidx, tokens, token_charidxs) -> List[Tuple[int, int]]:
        """Char start and end (exclusive) offset for each word-piece against the original text.
        The offsets for a word-piece are the offsets for the original token containing it.
        Therefore, if systemic becomes system, ##ic then the offsets for both word-pieces will be the same

        Returns:
            char_offsets: List[(start, end(ex)] same length as number of word-pieces
        """
        # List of (start, end) char offsets for each passage and question token. (end exclusive)
        char_offsets: List[Tuple[int, int]] = []
        for token_idx in wpidx2tokenidx:
            if token_idx >= 0:
                token_len = len(tokens[token_idx])
                # This is the start char offset for this token_idx
                token_start_charidx = token_charidxs[token_idx]
                char_offsets.append((token_start_charidx, token_start_charidx + token_len))
            else:
                char_offsets.append((0, 0))

        return char_offsets

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
                year_diff_idx = year_differences.index(year_diff)  # We know this will not fail
                year_difference_mat[date_idx1, date_idx2, year_diff_idx] = 1

        return year_differences, year_difference_mat

    def get_gold_action_seqs(
            self,
            program_supervision: Union[Dict, None],
            language: DropLanguage,
            action2idx_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[str],
                                                     bool]:

        gold_actionseq_idxs: List[List[int]] = []
        gold_actionseq_mask: List[List[int]] = []
        gold_function2actionidx_map: List[List[int]] = []
        gold_start_types: List[str] = []

        def return_no_program_supervision():
            gold_actionseq_idxs.append([0])
            gold_actionseq_mask.append([0])
            gold_function2actionidx_map.append([0])
            gold_start_types.append("UNK")
            return gold_actionseq_idxs, gold_actionseq_mask, gold_function2actionidx_map, gold_start_types, False

        if program_supervision is None:
            return return_no_program_supervision()

        program_node: Node = node_from_dict(program_supervision)
        program_return_type = self.function2returntype_mapping[program_node.predicate]
        program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
        try:
            gold_actions: List[str] = language.logical_form_to_action_sequence(program_lisp)
            actionseq_idxs: List[int] = [action2idx_map[a] for a in gold_actions]
            actionseq_mask: List[int] = [1 for _ in range(len(actionseq_idxs))]
            function2actionidx: List[int] = function_to_action_string_alignment(program_node, gold_actions)

            gold_actionseq_idxs.append(actionseq_idxs)
            gold_actionseq_mask.append(actionseq_mask)
            gold_function2actionidx_map.append(function2actionidx)
            gold_start_types.append(program_return_type)
            return gold_actionseq_idxs, gold_actionseq_mask, gold_function2actionidx_map, gold_start_types, True
        except:
            return return_no_program_supervision()


def question_attention_supervision_to_wps(program_node: Node, q_tokenidx2wpidx: List[List[int]], unpadded_q_wps_len,
                                          max_question_wps):
    supervision_dict: Dict = program_node.supervision
    if "question_attention_supervision" in supervision_dict:
        wps_attention = []
        question_attention = supervision_dict["question_attention_supervision"]
        for token_idx, token_attn in enumerate(question_attention):
            if token_idx < len(q_tokenidx2wpidx):
                wps_idxs = q_tokenidx2wpidx[token_idx]
                wps_attention.extend([token_attn] * len(wps_idxs))  # add as many attentions as wps for this token
        wps_attention = wps_attention[:max_question_wps]
        assert len(wps_attention) == unpadded_q_wps_len, f"Attn-len: {len(wps_attention)}  Q-len: {unpadded_q_wps_len}"
        supervision_dict["question_attention_supervision"] = wps_attention

    for c in program_node.children:
        question_attention_supervision_to_wps(c, q_tokenidx2wpidx, unpadded_q_wps_len, max_question_wps)


def revise_date_num_execution_supervision(program_node: Node, old2new_dateidxs: Dict[int, int],
                                          old2new_numidxs: Dict[int, int]):
    def is_date_num_supervision(sup_key: str) -> Tuple[bool, str]:
        """function to check if the supervision key is a date or number execution supervision.
            e.g. "num1_entidxs", "date2_entidxs", "num_entidxs", etc.
        """
        is_date_num_bool = True
        if sup_key.split("_")[1] != "entidxs":
            is_date_num_bool = False
        if sup_key[0:3] != "num" and sup_key[0:4] != "date":
            is_date_num_bool = False

        date_or_num_str = None
        if is_date_num_bool:
            date_or_num_str = "date" if sup_key[0:4] == "date" else "num"
        return is_date_num_bool, date_or_num_str

    supervision_dict: Dict = program_node.supervision

    for supervision_key in supervision_dict:
        is_date_num, date_or_num = is_date_num_supervision(supervision_key)
        if is_date_num:
            # This can be list of date or num entidxs
            symbol_entidxs = supervision_dict[supervision_key]
            old2new_entidxs = old2new_dateidxs if date_or_num == "date" else old2new_numidxs
            new_symbol_entidxs = []
            for entidx in symbol_entidxs:
                if entidx in old2new_entidxs:
                    new_symbol_entidxs.append(old2new_entidxs[entidx])
            supervision_dict[supervision_key] = new_symbol_entidxs

    for c in program_node.children:
        revise_date_num_execution_supervision(c, old2new_dateidxs, old2new_numidxs)
