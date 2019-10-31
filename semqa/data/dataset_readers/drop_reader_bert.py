import json
import logging
import itertools
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Tuple
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, MetadataField, ListField, SpanField, ArrayField

from allennlp_semparse.fields import ProductionRuleField

from pytorch_pretrained_bert import BertTokenizer

from semqa.domain_languages.drop_language import DropLanguage, Date, get_empty_language_object
from datasets.drop import constants


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


def tokenize_bert(bert_tokenizer: BertTokenizer, tokens: List[str]):
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
        wps = bert_tokenizer.tokenize(token)
        wordpiece_tokens.extend(wps)
        wpidx2tokenidx.extend([tokenidx] * len(wps))
        # Word-piece idxs for this token, wp_idx is the idx for the first token, and +i for the number of wps
        wp_idxs = [wp_idx + i for i, _ in enumerate(wps)]
        tokenidx2wpidxs.append(wp_idxs)

    assert len(wordpiece_tokens) == len(wpidx2tokenidx)
    assert len(tokens) == len(tokenidx2wpidxs)

    return wordpiece_tokens, tokenidx2wpidxs, wpidx2tokenidx


@TokenIndexer.register("bert-drop")
class BertDropTokenIndexer(WordpieceIndexer):
    def __init__(self, pretrained_model: str, max_pieces: int = 512) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        super().__init__(
            vocab=bert_tokenizer.vocab,
            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            do_lowercase=True,
            max_pieces=max_pieces,
            namespace="bert",
            separator_token="[SEP]",
        )


@DatasetReader.register("drop_reader_bert")
class DROPReaderNew(DatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        pretrained_model: str = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        relaxed_span_match: bool = True,
        do_augmentation: bool = True,
        question_length_limit: int = None,
        only_strongly_supervised: bool = False,
        skip_instances=False,
        skip_due_to_gold_programs=False,
        convert_spananswer_to_num=False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self._token_indexers = token_indexers
        self._relaxed_span_match = relaxed_span_match
        self._do_augmentation = do_augmentation
        self.question_length_limit = question_length_limit
        self.only_strongly_supervised = only_strongly_supervised
        self.skip_instances = skip_instances
        self.skip_due_to_gold_programs = skip_due_to_gold_programs
        self.convert_spananswer_to_num = convert_spananswer_to_num
        self.skip_count = 0
        self.skip_due_to_gold_not_in_answer = 0

        self.max_passage_nums = 0
        self.max_composed_nums = 0

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
        max_question_len = self.question_length_limit
        total_qas = 0
        instances_read = 0
        for passage_id, passage_info in dataset.items():
            original_passage_text = passage_info[constants.cleaned_passage]
            passage_text = passage_info[constants.tokenized_passage]
            passage_charidxs = passage_info[constants.passage_charidxs]
            p_date_mens: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = passage_info[
                constants.passage_date_mens
            ]
            p_date_entidxs: List[int] = passage_info[constants.passage_date_entidx]
            p_date_normvals: List[Tuple[int, int, int]] = passage_info[constants.passage_date_normalized_values]

            p_num_mens: List[Tuple[str, int, int]] = passage_info[constants.passage_num_mens]
            p_num_entidxs: List[int] = passage_info[constants.passage_num_entidx]
            p_num_normvals: List[int] = passage_info[constants.passage_num_normalized_values]

            for qa in passage_info[constants.qa_pairs]:
                total_qas += 1
                question_id = qa[constants.query_id]
                original_ques_text = qa[constants.cleaned_question]
                question_text = qa[constants.tokenized_question]
                question_charidxs = qa[constants.question_charidxs]

                if constants.answer_passage_spans in qa:
                    answer_passage_spans = qa[constants.answer_passage_spans]
                else:
                    answer_passage_spans = None
                if constants.answer_question_spans in qa:
                    answer_question_spans = qa[constants.answer_question_spans]
                else:
                    answer_question_spans = None
                answer_annotations = []
                if "answer" in qa:
                    answer_annotations.append(qa["answer"])
                if "validated_answers" in qa:
                    answer_annotations += qa["validated_answers"]

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
                            # This can be a 1- or 2- tuple of number groundings
                            num_grounding_supervision = qa[constants.qspan_numgrounding_supervision]

                # passage_att_supervision is probably never used
                passage_attn_supervision = None
                pattn_supervised = False
                if constants.pattn_supervised in qa:
                    pattn_supervised = qa[constants.pattn_supervised]
                    if constants.passage_attn_supervision in qa:
                        passage_attn_supervision = qa[constants.passage_attn_supervision]

                synthetic_numground_metadata = None
                if constants.SYN_NUMGROUND_METADATA in qa:
                    synthetic_numground_metadata = qa[constants.SYN_NUMGROUND_METADATA]

                strongly_supervised = program_supervised and qattn_supervised and execution_supervised

                if qattn_supervised is True:
                    assert program_supervised is True and qtype is not "UNK"
                if execution_supervised is True:
                    assert qattn_supervised is True

                instance = self.text_to_instance(
                    question_text,
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
                    pattn_supervised,
                    strongly_supervised,
                    ques_attn_supervision,
                    date_grounding_supervision,
                    num_grounding_supervision,
                    passage_attn_supervision,
                    synthetic_numground_metadata,
                    answer_passage_spans,
                    answer_question_spans,
                    question_id,
                    passage_id,
                    answer_annotations,
                    max_question_len,
                )

                if self.only_strongly_supervised:
                    if not strongly_supervised:
                        instance = None

                if instance is not None:
                    instances_read += 1
                    # print("\n\n")
                    yield instance

        #         if instance is not None:
        #             instances.append(instance)
        #         else:
        #             skip_count += 1
        # logger.info(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        # return instances
        logger.info(f"Total QAs: {total_qas}. Instances read: {instances_read}")
        logger.info(f"Instances Skipped: {self.skip_count}")
        logger.info(
            f"Instances skipped due to gold-answer not in gold_program_types: {self.skip_due_to_gold_not_in_answer}"
        )
        logger.info("Max passage nums: {} Max composed nums : {} ".format(self.max_passage_nums,
                                                                          self.max_composed_nums))

    @overrides
    def text_to_instance(
        self,
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
        pattn_supervised: bool,
        strongly_supervised: bool,
        ques_attn_supervision: Tuple[List[float]],
        date_grounding_supervision: Tuple[List[int], List[int]],
        num_grounding_supervision: Tuple[List[int], List[int]],
        passage_attn_supervision: List[float],
        synthetic_numground_metadata: List[Tuple[int, int]],
        answer_passage_spans: List[Tuple[int, int]],
        answer_question_spans: List[Tuple[int, int]],
        question_id: str = None,
        passage_id: str = None,
        answer_annotations: List[Dict[str, Union[str, Dict, List]]] = None,
        max_question_len: int = None,
    ) -> Union[Instance, None]:

        metadata = {
            "original_passage": original_passage_text,
            "original_question": original_ques_text,
            "passage_id": passage_id,
            "question_id": question_id,
        }

        language = get_empty_language_object()

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        passage_tokens: List[str] = passage_text.split(" ")
        question_tokens: List[str] = question_text.split(" ")

        # Word-piece tokenize the tokens in Q and P. Get maps;
        #  tokenidx2wpidx: List[List[int]] map from token idx to list of word-piece indices that correspond to it
        #  wpidx2tokenidx: List[int] map from word-piece index to original token idx
        # List[str], List[List[int]], List[int]
        passage_wps, p_tokenidx2wpidx, p_wpidx2tokenidx = tokenize_bert(self._tokenizer, passage_tokens)
        question_wps, q_tokenidx2wpidx, q_wpidx2tokenidx = tokenize_bert(self._tokenizer, question_tokens)

        # Truncate question_wps and wpidx2tokenidx to maximum allowable length
        question_wps = question_wps[0:max_question_len]
        q_wpidx2tokenidx = q_wpidx2tokenidx[0:max_question_len]
        max_q_tokenidx = q_wpidx2tokenidx[-1]  # The last token_idx after truncation
        max_q_token_len = max_q_tokenidx + 1
        # NOTE: Last token's all word-pieces might not be included; take precaution later
        q_tokenidx2wpidx = q_tokenidx2wpidx[0:max_q_token_len]

        # Padding question to max length since it makes possible to separate question and passage in the model
        q_wp_len = len(question_wps)
        q_wp_pad_len = max_question_len - q_wp_len
        question_wps.extend(["[PAD]"] * q_wp_pad_len)
        q_wpidx2tokenidx.extend([-1] * q_wp_pad_len)

        question_wps_tokens = [Token(text=t) for t in question_wps]

        # Passage_len = Max seq len - CLS - SEP - SEP - Max_Qlen -- Questions will be padded to max length
        max_passage_len = min(512 - 3 - max_question_len, len(passage_wps))
        passage_wps = passage_wps[0:max_passage_len]
        p_wpidx2tokenidx = p_wpidx2tokenidx[0:max_passage_len]
        max_p_tokenidx = p_wpidx2tokenidx[-1]
        max_p_token_len = max_p_tokenidx + 1
        p_tokenidx2wpidx = p_tokenidx2wpidx[0:max_p_token_len]

        passage_wps.append("[SEP]")
        p_wpidx2tokenidx.append(-1)

        passage_wps_tokens = [Token(text=t) for t in passage_wps]
        # This would be in the input to BERT
        question_passage_tokens = [Token("[CLS]")] + question_wps_tokens + [Token("[SEP]")] + passage_wps_tokens

        (
            p_date_mens,
            p_date_entidxs,
            p_date_normvals,
            p_num_mens,
            p_num_entidxs,
            p_num_normvals,
            answer_passage_spans,
            date_grounding_supervision,
            num_grounding_supervision,
            passage_attn_supervision,
        ) = self.prune_for_passage_len(
            max_p_token_len,
            p_date_mens,
            p_date_entidxs,
            p_date_normvals,
            p_num_mens,
            p_num_entidxs,
            p_num_normvals,
            answer_passage_spans,
            date_grounding_supervision,
            num_grounding_supervision,
            passage_attn_supervision,
        )

        (answer_question_spans, ques_attn_supervision) = self.prune_for_question_len(
            max_q_token_len, answer_question_spans, ques_attn_supervision
        )

        fields = {}
        fields["actions"] = action_field
        fields["question"] = TextField(question_wps_tokens, self._token_indexers)
        fields["passage"] = TextField(passage_wps_tokens + [Token("[SEP]")], self._token_indexers)
        fields["question_passage"] = TextField(question_passage_tokens, self._token_indexers)
        # List of (start, end) char offsets for each passage and question word-piece
        passage_offsets: List[Tuple[int, int]] = self.update_charoffsets(
            wpidx2tokenidx=p_wpidx2tokenidx, tokens=passage_tokens, token_charidxs=passage_charidxs
        )
        question_offsets: List[Tuple[int, int]] = self.update_charoffsets(
            wpidx2tokenidx=q_wpidx2tokenidx, tokens=question_tokens, token_charidxs=question_charidxs
        )
        # Passage Number
        passage_number_values = [int(x) if int(x) == x else x for x in p_num_normvals]
        # composed_numbers: List[int/float] is sorted
        # passage_number_values: now contains implicit numbers. Since they are added at the end,
        #  indexing should be fine.
        # compnumber2addcombinations (sub): Dict: {composed_number: List[(pass-num1, pass-num2)]} - mapping from
        #  composed number to list of passage-num-tuples that combine to form the number-key using the operation
        (composed_numbers, passage_number_values,
         compnumber2addcombinations, compnumber2subcombinations) = self.compute_number_support(
            numbers=passage_number_values,
            implicit_numbers=DropLanguage.implicit_numbers,
            max_number_of_numbers_to_consider=2,
        )

        self.max_passage_nums = max(len(passage_number_values), self.max_passage_nums)
        self.max_composed_nums = max(len(composed_numbers), self.max_composed_nums)
        if not passage_number_values:
            passage_number_values = [0]
        if not composed_numbers:
            composed_numbers = [0]
        # TODO(nitishg): Change this repr to (token_idx, value) repr.
        passage_number_entidxs = p_num_entidxs      # Index of passage_num_tokens in passage_number_values list
        passage_number_tokenids = [tokenidx for (_, tokenidx, _) in p_num_mens]
        assert len(passage_number_entidxs) == len(passage_number_tokenids)
        passage_num_wpidx2entidx = [-1 for _ in range(len(passage_wps))]  # number_ent_idx for each token (pad=-1)
        passage_number_wpindices = []  # wp_idxs that are numbers
        for passage_num_tokenidx, number_ent_idx in zip(passage_number_tokenids, passage_number_entidxs):
            wp_index = p_tokenidx2wpidx[passage_num_tokenidx][0]        # WP-idx of the number token
            passage_num_wpidx2entidx[wp_index] = number_ent_idx         # Index of the num-value in passsge_num_values
            passage_number_wpindices.append(wp_index)
        if not passage_number_wpindices:  # If no numbers in the passage, padding by faking the 0-th token as a number
            passage_num_wpidx2entidx[0] = 0
            passage_number_wpindices = [0]

        # Making a list of wp_idxs so that their corresponding values are increasingly sorted for differentiable min/max
        numvals_wpidx = [(passage_number_values[passage_num_wpidx2entidx[wpidx]], wpidx)
                         for wpidx in passage_number_wpindices]     # [(value, w_idx)]
        sorted_passage_number_wpindices = [x[1] for x in sorted(numvals_wpidx, key=lambda x: x[0])]
        # print("sorted_passage_number_wpindices: {}".format(sorted_passage_number_wpindices))

        fields["passageidx2numberidx"] = ArrayField(np.array(passage_num_wpidx2entidx), padding_value=-1)
        fields["passage_number_values"] = MetadataField(passage_number_values)
        fields["composed_numbers"] = MetadataField(composed_numbers)
        fields["passage_number_sortedtokenidxs"] = MetadataField(sorted_passage_number_wpindices)

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
            # Even though the p_tokenidx2wpidx is truncated, the wps might overflow
            end_wp_idx = min(p_tokenidx2wpidx[end_token_idx][-1], len(passage_wps) - 1)
            passage_date_spanidxs.append((start_wp_idx, end_wp_idx))

        passage_date_idx2dateidx = [-1 for _ in range(len(passage_wps))]
        if passage_date_spanidxs:
            for passage_date_span, date_idx in zip(passage_date_spanidxs, passage_date_entidxs):
                (s, e) = passage_date_span
                passage_date_idx2dateidx[s : e + 1] = [date_idx] * (e + 1 - s)
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
                "passage_token_offsets": passage_offsets,
                "question_token_offsets": question_offsets,
                "question_tokens": question_wps,
                "passage_tokens": passage_wps,
                "passage_date_values": passage_date_strvals,
                "composed_numbers": composed_numbers,
                "passage_number_values": passage_number_values,
                "passage_year_diffs": year_differences,
                "count_values": count_values,
            }
        )

        # FIELDS FOR STRONG-SUPERVISION
        fields["strongly_supervised"] = MetadataField(strongly_supervised)
        fields["program_supervised"] = MetadataField(program_supervised)
        fields["qattn_supervised"] = MetadataField(qattn_supervised)
        fields["execution_supervised"] = MetadataField(execution_supervised)
        fields["pattn_supervised"] = MetadataField(pattn_supervised)
        fields["qtypes"] = MetadataField(qtype)
        fields["synthetic_numground_metadata"] = MetadataField(synthetic_numground_metadata)

        # Question Attention Supervision
        if ques_attn_supervision:
            ques_attn_supervision_wp = []
            for qattn in ques_attn_supervision:
                qattn_wp = [0.0] * len(question_wps)
                for tokenidx, attnval in enumerate(qattn):
                    wp_idxs = q_tokenidx2wpidx[tokenidx]
                    for wpidx in wp_idxs:
                        if wpidx < len(question_wps):
                            qattn_wp[wpidx] = attnval
                ques_attn_supervision_wp.append(qattn_wp)

            fields["qattn_supervision"] = ArrayField(np.array(ques_attn_supervision_wp), padding_value=0)
        else:
            empty_question_attention = [0.0] * len(question_wps)
            empty_question_attention_tuple = [empty_question_attention]
            fields["qattn_supervision"] = ArrayField(np.array(empty_question_attention_tuple), padding_value=0)

        if passage_attn_supervision:
            fields["passage_attn_supervision"] = ArrayField(np.array(passage_attn_supervision), padding_value=0)
        else:
            empty_passage_attention = [0.0] * len(passage_wps)
            fields["passage_attn_supervision"] = ArrayField(np.array(empty_passage_attention), padding_value=0)

        # Date-comparison - Date Grounding Supervision
        if date_grounding_supervision:
            fields["datecomp_ques_event_date_groundings"] = MetadataField(date_grounding_supervision)
        else:
            empty_date_grounding = [0.0] * len(passage_date_objs)
            empty_date_grounding_tuple = (empty_date_grounding, empty_date_grounding)
            fields["datecomp_ques_event_date_groundings"] = MetadataField(empty_date_grounding_tuple)

        # Number Comparison - Passage Number Grounding Supervision
        if num_grounding_supervision:
            # print("Num Grounding Sup: {}".format(num_grounding_supervision))
            # number groundings need to be updated by padding with 0 for we now added implicit_numbers in passage nums
            num_implicit_nums = len(DropLanguage.implicit_numbers)
            new_num_grounding_supervision = []
            for grounding_sup in num_grounding_supervision:
                grounding_sup.extend([0]*num_implicit_nums)
                new_num_grounding_supervision.append(grounding_sup)
            # print("New num Grounding Sup: {}".format(new_num_grounding_supervision))
            fields["numcomp_qspan_num_groundings"] = MetadataField(new_num_grounding_supervision)
        else:
            empty_passagenum_grounding = [0.0] * len(passage_number_values)
            empty_passagenum_grounding_tuple = (empty_passagenum_grounding, empty_passagenum_grounding)
            fields["numcomp_qspan_num_groundings"] = MetadataField(empty_passagenum_grounding_tuple)

        # Get gold action_seqs for strongly_supervised questions
        action2idx_map = {rule: i for i, rule in enumerate(language.all_possible_productions())}

        # Tuple[List[List[int]], List[List[int]]]
        (
            gold_action_seqs,
            gold_actionseq_masks,
            gold_program_start_types,
            program_supervised,
        ) = self.get_gold_action_seqs(
            program_supervised=program_supervised,
            qtype=qtype,
            question_tokens=question_text.split(" "),
            language=language,
            action2idx_map=action2idx_map,
        )
        fields["program_supervised"] = MetadataField(program_supervised)
        fields["gold_action_seqs"] = MetadataField((gold_action_seqs, gold_actionseq_masks))

        ########     ANSWER FIELDS      ###################
        if answer_annotations:
            metadata.update({"answer_annotations": answer_annotations})
            # Using the first one supervision (training actually only has one)
            answer_annotation = answer_annotations[0]

            # This list contains the possible-start-types for programs that can yield the correct answer
            # For example, if the answer is a number but also in passage, this will contain two keys
            # If the answer is a number, we'll find which kind and that program-start-type will be added here
            answer_program_start_types: List[str] = []

            # We've pre-parsed the span types to passage / question spans

            # Passage-span answer
            if answer_passage_spans:
                answer_program_start_types.append("passage_span")
                passage_span_fields = []
                for (start_token_idx, end_token_idx) in answer_passage_spans:
                    start_wp_idx = p_tokenidx2wpidx[start_token_idx][0]
                    # Even though the p_tokenidx2wpidx is truncated, the wps might overflow
                    end_wp_idx = min(p_tokenidx2wpidx[end_token_idx][-1], len(passage_wps) - 1)
                    passage_span_fields.append(SpanField(start_wp_idx, end_wp_idx, fields["passage"]))
                metadata.update({"answer_passage_spans": answer_passage_spans})
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
            if answer_number is not None:
                # Passage-number answer
                if answer_number in passage_number_values:
                    answer_program_start_types.append("passage_number")
                    ans_as_passage_number[passage_number_values.index(answer_number)] = 1
                # Composed-number answer
                if answer_number in composed_numbers:
                    answer_program_start_types.append("composed_number")
                    ans_as_composed_number[composed_numbers.index(answer_number)] = 1
                # Year-difference answer
                if answer_number in year_differences:
                    answer_program_start_types.append("year_difference")
                    ans_as_year_difference[year_differences.index(answer_number)] = 1
                # Count answer
                if answer_number in count_values:
                    answer_program_start_types.append("count_number")
                    answer_as_count[count_values.index(answer_number)] = 1

            fields["answer_as_passage_number"] = MetadataField(ans_as_passage_number)
            fields["answer_as_composed_number"] = MetadataField(ans_as_composed_number)
            fields["answer_as_year_difference"] = MetadataField(ans_as_year_difference)
            fields["answer_as_count"] = MetadataField(answer_as_count)

            fields["answer_program_start_types"] = MetadataField(answer_program_start_types)

            # If we already have gold program(s), removing program_start_types that don't come from these gold_programs
            # print(f"AnswerTypes: {answer_program_start_types}")
            # print(f"New AnswerTypes: {new_answer_program_start_types}")

            if self.skip_due_to_gold_programs:
                if program_supervised:
                    new_answer_program_start_types = []
                    for answer_program_type in answer_program_start_types:
                        if answer_program_type in gold_program_start_types:
                            new_answer_program_start_types.append(answer_program_type)
                else:
                    new_answer_program_start_types = answer_program_start_types
                # Answer exists as other programs but not for gold-program
                if len(answer_program_start_types) != 0 and len(new_answer_program_start_types) == 0:
                    self.skip_due_to_gold_not_in_answer += 1
                answer_program_start_types = new_answer_program_start_types

            if self.skip_instances:
                if len(answer_program_start_types) == 0:
                    self.skip_count += 1
                    # print("\nNo answer grounding")
                    # print(original_ques_text)
                    # print(original_passage_text)
                    # print(answer_annotation)
                    # print(answer_passage_spans)
                    # print(answer_question_spans)
                    # print(f"NumSupport: {number_support}")
                    return None
        # End if answer-annotation

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def prune_for_passage_len(
        self,
        max_passage_len: int,
        p_date_mens,
        p_date_entidxs,
        p_date_normvals,
        p_num_mens,
        p_num_entidxs,
        p_num_normvals,
        answer_passage_spans,
        date_grounding_supervision,
        num_grounding_supervision,
        passage_attn_supervision,
    ):

        """ Prunes the passage and related data for a maximum length

            For the given max_passage_len, we first need to find out the pruned date and number mentions
            Since these might remove some dates and numbers from the passage, we need to find the
            pruned list of p_date_normvals (p_date_entidxs with the new date_entidxs)
            pruned list of p_num_normvals (p_num_entidxs with new num_entidxs) -- make sure the numbers are still sorted

            answer_passage_spans - only spans that are contained in the pruned passage

            date_grounding_supervision, num_grounding_supervision -- both these are the length of original dates/nums
            we need to find the new value by pruning and mapping old ent idxs to new ones.

            passage_attn_supervision: if not None, is a list the length of the passage
        """
        pruned_date_mens = []  # New passage date mens
        pruned_old_dateidxs = []
        for date_men, date_idx in zip(p_date_mens, p_date_entidxs):
            _, (x, y), _ = date_men
            if y < max_passage_len:
                pruned_date_mens.append(date_men)
                pruned_old_dateidxs.append(date_idx)

        new_date_values = []  # New passage date values
        new2old_dateidx = {}
        old2new_dateidx = {}
        for old_date_idx, date_value in enumerate(p_date_normvals):
            # Atleast one mention of this old_date_idx remains
            if old_date_idx in pruned_old_dateidxs:
                new_date_idx = len(new_date_values)
                new2old_dateidx[new_date_idx] = old_date_idx
                old2new_dateidx[old_date_idx] = new_date_idx
                new_date_values.append(date_value)

        new_date_entidxs = [old2new_dateidx[x] for x in pruned_old_dateidxs]  # New passage date entidxs

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

        if answer_passage_spans:
            new_answer_passage_spans = [span for span in answer_passage_spans if span[1] < max_passage_len]
        else:
            new_answer_passage_spans = answer_passage_spans

        if passage_attn_supervision is not None and len(passage_attn_supervision) > max_passage_len:
            new_passage_attn_supervision = passage_attn_supervision[0:max_passage_len]
        else:
            new_passage_attn_supervision = passage_attn_supervision

        return (
            pruned_date_mens,
            new_date_entidxs,
            new_date_values,
            pruned_num_mens,
            new_num_idxs,
            new_num_values,
            new_answer_passage_spans,
            new_dategrounding_supervision,
            new_numgrounding_supervision,
            new_passage_attn_supervision,
        )

    def prune_for_question_len(self, max_question_len, answer_question_spans, ques_attn_supervision):
        if answer_question_spans:
            new_answer_question_spans = [span for span in answer_question_spans if span[1] < max_question_len]
        else:
            new_answer_question_spans = answer_question_spans

        if ques_attn_supervision is not None:
            new_qattn_supervision = [qattn[0:max_question_len] for qattn in ques_attn_supervision]
        else:
            new_qattn_supervision = None

        return (new_answer_question_spans, new_qattn_supervision)

    @staticmethod
    def compute_number_support(
        numbers: List[Union[int, float]],
        implicit_numbers: List[Union[int, float]] = None,
        max_number_of_numbers_to_consider: int = 2,
    ) -> Tuple[List[Union[int, float]], List[Union[int, float]], Dict, Dict]:
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

        numbers_w_impnums = [x for x in numbers]
        # Adding implicit numbers here after checking if 0 is a part of original numbers so that we don't add tons of
        #  combinations of the kind x = x + 0 / x - 0
        zero_in_numbers = True if 0 in numbers else False
        # Adding implicit-numbers to the input-numbers list since they can take part in composition with input-numbers.
        if implicit_numbers:
            numbers_w_impnums.extend(implicit_numbers)

        composed_num_set = set()
        # Map from composed-number to list of number-combination that lead to this number from the add/sub operation
        compnumber2subcombinations = defaultdict(set)
        compnumber2addcombinations = defaultdict(set)
        signs = [-1, 1]
        # all_sign_combinations = list(itertools.product(signs, repeat=2))
        # Since our modules will only perform num1-num2 / num1+num2. Computation like -num1+num2 would not be done
        all_sign_combinations = [(1.0, -1.0), (1.0, 1.0)]
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            # for number_combination in itertools.combinations(numbers, r=number_of_numbers_to_consider):
            for indexed_number_combination in itertools.product(
                enumerate(numbers_w_impnums), repeat=number_of_numbers_to_consider
            ):
                ((idx1, num1), (idx2, num2)) = indexed_number_combination
                number_combination = (num1, num2)
                # if idx1 == idx2: continue     # Commented: 0 in support. Un-commented: 0 not in support
                # print(indexed_number_combination)
                for sign_combination in all_sign_combinations:
                    value = sum([sign * num for (sign, num) in zip(sign_combination, number_combination)])
                    if value >= 0:
                        composed_num_set.add(value)
                        # If 0 was originally in numbers then allow its combinations, o/w don't to avoid the
                        # combinations from getting bloated with x = x+0, 0+x, x-0
                        if (0 in number_combination and zero_in_numbers) or (0 not in number_combination):
                            if sign_combination == (1, 1):
                                compnumber2addcombinations[value].add(number_combination)
                            else:  #  sign_combination == [1, -1]:
                                compnumber2subcombinations[value].add(number_combination)

        composed_numbers = sorted(list(composed_num_set))

        return composed_numbers, numbers_w_impnums, compnumber2addcombinations, compnumber2subcombinations

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
        program_supervised: bool,
        qtype: str,
        question_tokens: List[str],
        language: DropLanguage,
        action2idx_map: Dict[str, int],
    ) -> Tuple[List[List[int]], List[List[int]], List[str], bool]:

        qtype_to_lffunc = {
            constants.DATECOMP_QTYPE: self.datecomp_logicalforms,
            constants.NUMCOMP_QTYPE: self.numcomp_logicalforms,
            constants.NUM_find_qtype: self.findnum_logicalforms,
            constants.NUM_filter_find_qtype: self.filterfindnum_logicalforms,
            constants.MIN_find_qtype: self.minnum_find_logicalforms,
            constants.MIN_filter_find_qtype: self.minnum_filterfind_logicalforms,
            constants.MAX_find_qtype: self.maxnum_find_logicalforms,
            constants.MAX_filter_find_qtype: self.maxnum_filterfind_logicalforms,
            constants.COUNT_find_qtype: self.count_find_logicalforms,
            constants.COUNT_filter_find_qtype: self.count_filterfind_logicalforms,
            constants.RELOC_find_qtype: self.relocate_logicalforms,
            constants.RELOC_filterfind_qtype: self.relocate_logicalforms,
            constants.RELOC_maxfind_qtype: self.relocate_logicalforms,
            constants.RELOC_maxfilterfind_qtype: self.relocate_logicalforms,
            constants.RELOC_minfind_qtype: self.relocate_logicalforms,
            constants.RELOC_minfilterfind_qtype: self.relocate_logicalforms,
            constants.YEARDIFF_SE_qtype: self.yeardiff_singleevent_logicalforms,
            constants.YEARDIFF_TE_qtype: self.yeardiff_twoevent_logicalforms,
        }

        gold_actionseq_idxs: List[List[int]] = []
        gold_actionseq_mask: List[List[int]] = []
        gold_start_types: List[str] = []

        if not program_supervised:
            gold_actionseq_idxs.append([0])
            gold_actionseq_mask.append([0])
            gold_start_types.append("UNK")
            return (gold_actionseq_idxs, gold_actionseq_mask, gold_start_types, program_supervised)

        if qtype in qtype_to_lffunc:
            # Tuple[List[str], List[str]]
            (gold_logical_forms, gold_start_types) = qtype_to_lffunc[qtype](
                question_tokens=question_tokens, language=language, qtype=qtype
            )
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
            gold_start_types.append("UNK")
            logger.error(f"Tried get gold logical form for: {qtype}")

        return (gold_actionseq_idxs, gold_actionseq_mask, gold_start_types, program_supervised)

    @staticmethod
    def filter_passageattn_lf() -> str:
        gold_lf = "(filter_PassageAttention find_PassageAttention)"
        return gold_lf

    @staticmethod
    def findnum_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = "(find_PassageNumber find_PassageAttention)"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def filterfindnum_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        filter_passage_attention_lf = DROPReaderNew.filter_passageattn_lf()
        gold_lf = f"(find_PassageNumber {filter_passage_attention_lf})"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def minnum_find_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = f"(find_PassageNumber (minNumPattn find_PassageAttention))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def minnum_filterfind_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        filter_passage_attention_lf = DROPReaderNew.filter_passageattn_lf()
        gold_lf = f"(find_PassageNumber (minNumPattn {filter_passage_attention_lf}))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def maxnum_find_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = f"(find_PassageNumber (maxNumPattn find_PassageAttention))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def maxnum_filterfind_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        filter_passage_attention_lf = DROPReaderNew.filter_passageattn_lf()
        gold_lf = f"(find_PassageNumber (maxNumPattn {filter_passage_attention_lf}))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def count_find_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = "(passageAttn2Count find_PassageAttention)"
        return [gold_lf], ["count_number"]

    @staticmethod
    def count_filterfind_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        filter_passageattn_lf = DROPReaderNew.filter_passageattn_lf()
        gold_lf = f"(passageAttn2Count {filter_passageattn_lf})"
        return [gold_lf], ["count_number"]

    @staticmethod
    def relocate_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        qtype = kwargs["qtype"]
        # Could be one of
        # 'relocate_filterfind_qtype', 'relocate_minfind_qtype', 'relocate_maxfind_qtype',
        # 'relocate_maxfilterfind_qtype', 'relocate_find_qtype', 'relocate_minfilterfind_qtype'

        find = "find_PassageAttention"
        filterfind = "(filter_PassageAttention find_PassageAttention)"
        maxfind = "(maxNumPattn find_PassageAttention)"
        maxfilterfind = f"(maxNumPattn {filterfind})"
        minfind = "(minNumPattn find_PassageAttention)"
        minfilterfind = f"(minNumPattn {filterfind})"

        outer_leftside = "(find_passageSpanAnswer (relocate_PassageAttention "
        outer_rightside = "))"

        if qtype == constants.RELOC_find_qtype:
            gold_lf = outer_leftside + find + outer_rightside
        elif qtype == constants.RELOC_filterfind_qtype:
            gold_lf = outer_leftside + filterfind + outer_rightside
        elif qtype == constants.RELOC_maxfind_qtype:
            gold_lf = outer_leftside + maxfind + outer_rightside
        elif qtype == constants.RELOC_maxfilterfind_qtype:
            gold_lf = outer_leftside + maxfilterfind + outer_rightside
        elif qtype == constants.RELOC_minfind_qtype:
            gold_lf = outer_leftside + minfind + outer_rightside
        elif qtype == constants.RELOC_minfilterfind_qtype:
            gold_lf = outer_leftside + minfilterfind + outer_rightside
        else:
            raise NotImplementedError

        return [gold_lf], ["passage_span"]

    @staticmethod
    def yeardiff_singleevent_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        qtype = kwargs["qtype"]
        gold_lf = "(year_difference_single_event find_PassageAttention)"

        return [gold_lf], ["year_difference"]

    @staticmethod
    def yeardiff_twoevent_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        qtype = kwargs["qtype"]
        gold_lf = "(year_difference find_PassageAttention find_PassageAttention)"

        return [gold_lf], ["year_difference"]

    @staticmethod
    def numdiff_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        qtype = kwargs["qtype"]
        # Qtype of form: diff_maxmin_qtype
        numtypes = qtype.split("_")[1]
        first_num = numtypes[0:3]  # first 3 chars
        second_num = numtypes[3:6]  # last 3 chars

        max_num_program = "(max_PassageNumber (find_PassageNumber find_PassageAttention))"
        min_num_program = "(min_PassageNumber (find_PassageNumber find_PassageAttention))"
        find_num_program = "(find_PassageNumber find_PassageAttention)"

        if first_num == "max":
            first_num_prog = max_num_program
        elif first_num == "min":
            first_num_prog = min_num_program
        elif first_num == "num":
            first_num_prog = find_num_program
        else:
            raise NotImplementedError

        if second_num == "max":
            second_num_prog = max_num_program
        elif second_num == "min":
            second_num_prog = min_num_program
        elif second_num == "num":
            second_num_prog = find_num_program
        else:
            raise NotImplementedError

        # "(passagenumber_difference first_num_prog second_num_program)"
        gold_lf = f"(passagenumber_difference {first_num_prog} {second_num_prog})"

        return [gold_lf], ["passagenum_diff"]

    @staticmethod
    def yardsshortest_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = "(min_PassageNumber (find_PassageNumber find_PassageAttention))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def yardslongest_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        gold_lf = "(max_PassageNumber (find_PassageNumber find_PassageAttention))"
        return [gold_lf], ["passage_number"]

    @staticmethod
    def datecomp_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        question_tokens: List[str] = kwargs["question_tokens"]
        language: DropLanguage = kwargs["language"]
        # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
        psa_start = "(find_passageSpanAnswer ("
        qsa_start = "(find_questionSpanAnswer ("
        # lf1 = "(find_passageSpanAnswer ("

        lf2 = " find_PassageAttention find_PassageAttention))"
        greater_than = "compare_date_greater_than"
        lesser_than = "compare_date_lesser_than"

        # Correct if Attn1 is first event
        lesser_tokens = ["first", "earlier", "forst", "firts"]
        greater_tokens = ["later", "last", "second"]

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
        gold_start_types = []
        if "@start@ -> PassageSpanAnswer" in language.all_possible_productions():
            gold_logical_forms.append(f"{psa_start}{operator_action}{lf2}")
            gold_start_types.append("passage_span")  # from drop_parser.get_valid_start_actionids
        if "@start@ -> QuestionSpanAnswer" in language.all_possible_productions():
            gold_logical_forms.append(f"{qsa_start}{operator_action}{lf2}")
            gold_start_types.append("question_span")  # from drop_parser.get_valid_start_actionids

        return gold_logical_forms, gold_start_types

    @staticmethod
    def numcomp_logicalforms(**kwargs) -> Tuple[List[str], List[str]]:
        question_tokens: List[str] = kwargs["question_tokens"]
        language: DropLanguage = kwargs["language"]
        # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
        psa_start = "(find_passageSpanAnswer ("
        qsa_start = "(find_questionSpanAnswer ("

        lf2 = " find_PassageAttention find_PassageAttention))"
        greater_than = "compare_num_greater_than"
        lesser_than = "compare_num_lesser_than"

        # Correct if Attn1 is first event
        greater_tokens = ["larger", "more", "largest", "bigger", "higher", "highest", "most", "greater"]
        lesser_tokens = ["smaller", "fewer", "lowest", "smallest", "less", "least", "fewest", "lower"]

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
        gold_start_types = []
        if "@start@ -> PassageSpanAnswer" in language.all_possible_productions():
            gold_logical_forms.append(f"{psa_start}{operator_action}{lf2}")
            gold_start_types.append("passage_span")  # from drop_parser.get_valid_start_actionids
        if "@start@ -> QuestionSpanAnswer" in language.all_possible_productions():
            gold_logical_forms.append(f"{qsa_start}{operator_action}{lf2}")
            gold_start_types.append("question_span")  # from drop_parser.get_valid_start_actionids

        return gold_logical_forms, gold_start_types
