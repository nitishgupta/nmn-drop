import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import numpy as np
from overrides import overrides
import torch
import gc

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.data import TextFieldTensors
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules import Attention, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn import Activation
from allennlp.modules.matrix_attention import DotProductMatrixAttention, LinearMatrixAttention
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average

from allennlp_semparse.state_machines.states import GrammarBasedState
from allennlp_semparse.state_machines.transition_functions import BasicTransitionFunction
from allennlp_semparse.state_machines.trainers.maximum_marginal_likelihood import MaximumMarginalLikelihood
from allennlp_semparse.state_machines import BeamSearch
from allennlp_semparse.state_machines import ConstrainedBeamSearch
from allennlp_semparse.fields.production_rule_field import ProductionRule
from semqa.modules.symbolic.utils import compute_token_symbol_alignments

from semqa.utils.rc_utils import DropEmAndF1
from semqa.state_machines.prefixed_beam_search import PrefixedConstrainedBeamSearch
from semqa.models.utils import semparse_utils
from semqa.utils import qdmr_utils
from semqa.utils.qdmr_utils import Node
from semqa.models.drop_parser_base import DROPParserBase
from semqa.domain_languages.drop_language import (
    DropLanguage,
    Date,
    CountNumber,
)
from semqa.domain_languages.drop_execution_parameters import ExecutorParameters
from semqa.modules.spans import SingleSpanAnswer, MultiSpanAnswer
from semqa.modules.qp_encodings import BertJointQPEncoding, BertIndependentQPEncoding
from semqa.modules.qrepr_module.qrepr_to_module import QReprModuleExecution
import datasets.drop.constants as dropconstants
from semqa.profiler.profile import Profile, profile_func_decorator

from semqa.modules.shared_substructure_module import shared_substructure_utils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("drop_parser_bert")
class DROPParserBERT(DROPParserBase):
    def __init__(
        self,
        vocab: Vocabulary,
        max_ques_len: int,
        action_embedding_dim: int,
        transitionfunc_attention: Attention,
        passage_attention_to_span: Seq2SeqEncoder,
        beam_size: int,
        max_decoding_steps: int,
        transformer_model_name: str,
        bio_tagging: bool,
        bio_label_scheme: str = "IO",
        tie_biocount: bool = False,
        passage_attention_to_count: Seq2SeqEncoder = None,
        qp_encoding_style: str = "bert_joint_qp_encoder",
        qrepr_style: str = "attn",
        shared_substructure: bool = False,
        countfixed: bool = False,
        auxwinloss: bool = False,
        excloss: bool = False,
        qattloss: bool = False,
        mmlloss: bool = False,
        hardem_epoch: int = 0,
        dropout: float = 0.0,
        debug: bool = False,
        interpret: bool = False,    # This is a flag for interpret-acl20
        gc_freq: int = 500,
        profile_freq: Optional[int] = None,
        cuda_device: int = -1,
        initializers: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super(DROPParserBERT, self).__init__(
            vocab=vocab,
            action_embedding_dim=action_embedding_dim,
            dropout=dropout,
            debug=debug,
            regularizer=regularizer,
        )

        self._text_field_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )

        self._allennlp_tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model_name)
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._pad_token_id = self._tokenizer.pad_token_id
        self._sep_token_id = self._tokenizer.sep_token_id

        if qp_encoding_style == "bert_joint_qp_encoder":
            self.qp_encoder = BertJointQPEncoding(text_field_embedder=self._text_field_embedder,
                                                  pad_token_id=self._pad_token_id,
                                                  sep_token_id=self._sep_token_id)
        elif qp_encoding_style == "bert_independent_qp_encoding":
            self.qp_encoder = BertIndependentQPEncoding(text_field_embedder=self._text_field_embedder,
                                                        pad_token_id=self._pad_token_id,
                                                        sep_token_id=self._sep_token_id)
        else:
            raise NotImplementedError


        self.qrepr_module_exec = QReprModuleExecution(qrepr_style=qrepr_style,
                                                      qp_encoding_style=qp_encoding_style)

        self.max_ques_len = max_ques_len

        question_encoding_dim = self._text_field_embedder.get_output_dim()
        self.bert_dim = question_encoding_dim

        self._decoder_step = BasicTransitionFunction(
            encoder_output_dim=question_encoding_dim,
            action_embedding_dim=action_embedding_dim,
            input_attention=transitionfunc_attention,
            activation=Activation.by_name("tanh")(),
            add_action_bias=False,
            dropout=dropout,
        )
        self._mml = MaximumMarginalLikelihood()

        self._beam_size = beam_size
        self._decoder_beam_search = BeamSearch(beam_size=self._beam_size)
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

        # This metrircs measure accuracy of
        # (1) Top-predicted program, (2) ExpectedDenotation from the beam (3) Best accuracy from topK(5) programs
        self.top1_acc_metric = Average()
        self.expden_acc_metric = Average()
        self.topk_acc_metric = Average()
        self.aux_goldparse_loss = Average()
        self.qent_loss = Average()
        self.qattn_cov_loss_metric = Average()

        # Use a separate encoder for passage - date - num similarity

        self.qp_matrix_attention = LinearMatrixAttention(
            tensor_1_dim=self.bert_dim, tensor_2_dim=self.bert_dim, combination="x,y,x*y"
        )

        # self.passage_token_to_date = passage_token_to_date
        self.dotprod_matrix_attn = DotProductMatrixAttention()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self.passage_attention_bio_encoder = None
        self.passage_attention_to_span = passage_attention_to_span
        self.passage_startend_predictor = None
        self.passage_bio_predictor = None

        self.bio_tagging = bio_tagging
        self.bio_label_scheme = bio_label_scheme
        self.bio_labels = None
        if self.bio_tagging:
            if self.bio_label_scheme == "BIO":
                self.bio_labels = {'O': 0, 'B': 1, 'I': 2}
            elif self.bio_label_scheme == "IO":
                self.bio_labels = {'O': 0, 'I': 1}
            else:
                raise Exception("bio_label_scheme not supported: {}".format(self.bio_label_scheme))

            self.span_answer = MultiSpanAnswer(ignore_question=True,
                                               prediction_method="viterbi",
                                               decoding_style="at_least_one",
                                               training_style="",
                                               labels=self.bio_labels)
            self.passage_bio_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(),
                                                         len(self.bio_labels))
        else:
            self.span_answer = SingleSpanAnswer()
            self.passage_attention_to_span = passage_attention_to_span
            self.passage_startend_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(), 2)

        self.num_counts = 10
        if tie_biocount:
            self.passage_attention_to_count = self.passage_attention_to_span
        else:
            self.passage_attention_to_count = passage_attention_to_count
        self.passage_count_hidden2logits = torch.nn.Linear(
            self.passage_attention_to_count.get_output_dim(), 1, bias=True
        )

        self._num_implicit_nums = len(DropLanguage.implicit_numbers)

        self._executor_parameters = ExecutorParameters(
            question_encoding_dim=self.bert_dim,
            passage_encoding_dim=self.bert_dim,
            passage_attention_to_span=self.passage_attention_to_span,
            passage_startend_predictor=self.passage_startend_predictor,
            passage_bio_predictor=self.passage_bio_predictor,
            passage_attention_to_count=self.passage_attention_to_count,
            passage_count_hidden2logits=self.passage_count_hidden2logits,
            num_implicit_nums=self._num_implicit_nums,
            dropout=dropout,
        )

        self.shared_substructure = shared_substructure

        self.modelloss_metric = Average()
        self.excloss_metric = Average()
        self.qattloss_metric = Average()
        self.mmlloss_metric = Average()
        self.shrdsubloss_metric = Average()
        self.auxwinloss_metric = Average()
        self._drop_metrics = DropEmAndF1()

        self.auxwinloss = auxwinloss

        # Main loss for QA
        # Auxiliary losses, such as - Prog-MML, QAttn, DateGrounding etc.
        self.excloss = excloss
        self.qattloss = qattloss
        self.mmlloss = mmlloss

        # Hard-EM will start from this epoch (0-indexed meaning from the beginning); until then MML loss will be used
        self.hardem_epoch = hardem_epoch if hardem_epoch >= 0 else None

        initializers(self)

        for name, parameter in self.named_parameters():
            if name == "_text_field_embedder.token_embedder_tokens.weight":
                parameter.requires_grad = False

        # # # Fix parameters for Counting
        count_parameter_names = ["passage_attention_to_count", "passage_count_hidden2logits", "passage_count_predictor"]
        if countfixed:
            for name, parameter in self.named_parameters():
                if any(span in name for span in count_parameter_names):
                    parameter.requires_grad = False

        self.profile_steps = 0
        self.profile_freq = None if profile_freq == 0 else profile_freq
        self.device_id = None
        self.interpret = interpret
        self.gc_steps = 0
        self.gc_freq = gc_freq

    @profile_func_decorator
    @overrides
    def forward(
        self,
        question_passage: TextFieldTensors,
        question: TextFieldTensors,
        passage: TextFieldTensors,
        p_sentboundary_wps: torch.LongTensor,
        passageidx2numberidx: torch.Tensor,
        passage_number_values: List[List[float]],
        composed_numbers: List[List[float]],
        passage_number_sortedtokenidxs: List[List[int]],
        add_number_combinations_indices: torch.LongTensor,
        sub_number_combinations_indices: torch.LongTensor,
        max_num_add_combs: List[int],
        max_num_sub_combs: List[int],
        passageidx2dateidx: torch.LongTensor,
        passage_date_values: List[List[Date]],
        year_differences: List[List[int]],
        year_differences_mat: List[np.array],
        count_values: List[List[int]],
        actions: List[List[ProductionRule]],
        passage_span_answer: torch.LongTensor = None,     # BIO: (bs, num_tagging, passage_len), S/E: (bs, num_spans, 2)
        passage_span_answer_mask: torch.LongTensor = None,   # BIO: (bs, num_tagging), S/E: (bs, num_spans)
        answer_spans_for_possible_taggings: torch.LongTensor = None,    # (batch_size, num_tagging, num_spans, 2)
        answer_as_question_spans: torch.LongTensor = None,
        answer_as_passage_number: List[List[int]] = None,
        answer_as_composed_number: List[List[int]] = None,
        answer_as_year_difference: List[List[int]] = None,
        answer_as_count: List[List[int]] = None,
        composed_num_ans_composition_types: List[Set[str]] = None,
        answer_program_start_types: List[Union[List[str], None]] = None,
        program_supervised: List[bool] = None,
        execution_supervised: List[bool] = None,
        strongly_supervised: List[bool] = None,
        gold_action_seqs: List[Tuple[List[List[int]], List[List[int]]]] = None,
        gold_function2actionidx_maps: List[List[List[int]]] = None,
        gold_program_dicts: List[List[Union[Dict, None]]] = None,
        sharedsub_question_passage: TextFieldTensors = None,
        sharedsub_program_nodes: Optional[List[Node]] = None,
        sharedsub_function2actionidx_maps: Optional[List[List[List[int]]]] = None,
        sharedsub_program_lisp: Union[None, List[List[str]]] = None,
        sharedsub_orig_program_lisp: Union[None, List[str]] = None,
        orig_sharedsub_postorder_node_idx: Union[List[List[Tuple[int, int]]], None] = None,
        sharedsub_mask: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            epoch = self.epoch
        else:
            epoch = None

        self.gc_steps += 1
        if self.gc_steps % self.gc_freq == 0:
            gc.collect()

        if self.device_id is None:
            self.device_id = allenutil.get_device_of(self.qp_matrix_attention._weight_vector.data)

        self.profile_steps += 1
        if self.profile_freq is not None:
            if self.profile_steps % self.profile_freq == 0:
                logger.info(Profile.to_string())

        qp_encoder_output = self.qp_encoder.get_representation(question=question,
                                                               passage=passage,
                                                               question_passage=question_passage,
                                                               max_ques_len=self.max_ques_len)

        question_token_idxs = qp_encoder_output["question_token_idxs"]
        passage_token_idxs = qp_encoder_output["passage_token_idxs"]
        question_mask = qp_encoder_output["question_mask"]
        passage_mask = qp_encoder_output["passage_mask"]
        encoded_question = qp_encoder_output["encoded_question"]
        encoded_passage = qp_encoder_output["encoded_passage"]
        bert_pooled_out = qp_encoder_output["pooled_encoding"]

        batch_size = len(actions)

        modeled_passage = encoded_passage
        passage_length = modeled_passage.size()[1]

        # question_passage_similarity = self.qp_matrix_attention(encoded_question, modeled_passage)
        # passage_question_similarity = question_passage_similarity.transpose(1, 2)

        """ No more num-date alignment
        # Passage Token - Date Alignment
        # Shape: (batch_size, passage_length, passage_length)
        passage_passage_token2date_alignment = compute_token_symbol_alignments(
            modeled_passage=modeled_passage,
            passage_mask=passage_mask,
            passageidx2symbolidx=passageidx2dateidx,
            passage_to_symbol_attention_params=self._executor_parameters.passage_to_date_attention
        )

        passage_passage_token2startdate_alignment = compute_token_symbol_alignments(
            modeled_passage=modeled_passage,
            passage_mask=passage_mask,
            passageidx2symbolidx=passageidx2dateidx,
            passage_to_symbol_attention_params=self._executor_parameters.passage_to_start_date_attention
        )

        passage_passage_token2enddate_alignment = compute_token_symbol_alignments(
            modeled_passage=modeled_passage,
            passage_mask=passage_mask,
            passageidx2symbolidx=passageidx2dateidx,
            passage_to_symbol_attention_params=self._executor_parameters.passage_to_end_date_attention
        )
        # Passage Token - Num Alignment
        passage_passage_token2num_alignment = compute_token_symbol_alignments(
            modeled_passage=modeled_passage,
            passage_mask=passage_mask,
            passageidx2symbolidx=passageidx2numberidx,
            passage_to_symbol_attention_params=self._executor_parameters.passage_to_num_attention
        )
        No more num-date alignment """

        # json_dicts = []
        # for i in range(batch_size):
        #     ques_tokens = metadata[i]['question_tokens']
        #     passage_tokens = metadata[i]['passage_tokens']
        #     p2num_sim = myutils.tocpuNPList(passage_passage_token2num_similarity[i])
        #     outdict = {'q': ques_tokens, 'p': passage_tokens, 'num': p2num_sim}
        #     json_dicts.append(outdict)
        # json.dump(json_dicts, num_attentions_f)
        # num_attentions_f.close()
        # exit()
        """ Aux Loss """

        """ No more window loss """
        # if self.auxwinloss:
        #     inwindow_mask, outwindow_mask = self.masking_blockdiagonal(passage_length, 15, self.device_id)
        #     passage_tokenidx2numidx_mask = (passageidx2numberidx > -1).float()
        #     num_aux_loss = self.window_loss_numdate(
        #         passage_passage_token2num_alignment, passage_tokenidx2numidx_mask, inwindow_mask, outwindow_mask
        #     )
        #
        #     passage_tokenidx2dateidx_mask = (passageidx2dateidx > -1).float()
        #     date_aux_loss = self.window_loss_numdate(
        #         passage_passage_token2date_alignment, passage_tokenidx2dateidx_mask, inwindow_mask, outwindow_mask
        #     )
        #
        #     start_date_aux_loss = self.window_loss_numdate(
        #         passage_passage_token2startdate_alignment, passage_tokenidx2dateidx_mask, inwindow_mask,
        #         outwindow_mask)
        #
        #     end_date_aux_loss = self.window_loss_numdate(
        #         passage_passage_token2enddate_alignment, passage_tokenidx2dateidx_mask, inwindow_mask,
        #         outwindow_mask)
        #     aux_win_loss = num_aux_loss + date_aux_loss + start_date_aux_loss + end_date_aux_loss
        # else:
        #     aux_win_loss = 0.0
        """ No more window loss """

        """ Parser setup """
        # Shape: (B, encoding_dim)
        question_encoded_final_state = bert_pooled_out
        question_encoded_aslist = [encoded_question[i] for i in range(batch_size)]
        question_mask_aslist = [question_mask[i] for i in range(batch_size)]
        passage_encoded_aslist = [encoded_passage[i] for i in range(batch_size)]
        passage_modeled_aslist = [modeled_passage[i] for i in range(batch_size)]
        # p2pdate_alignment_aslist = [passage_passage_token2date_alignment[i] for i in range(batch_size)]
        # p2pstartdate_alignment_aslist = [passage_passage_token2startdate_alignment[i] for i in range(batch_size)]
        # p2penddate_alignment_aslist = [passage_passage_token2enddate_alignment[i] for i in range(batch_size)]
        # p2pnum_alignment_aslist = [passage_passage_token2num_alignment[i] for i in range(batch_size)]
        size_composednums_aslist = [len(x) for x in composed_numbers]
        # Shape: (size_num_support_i, max_num_add_combs_i, 2) where _i is per instance
        add_num_combination_aslist = [
            add_number_combinations_indices[i, 0:size_composednums_aslist[i], 0:max_num_add_combs[i], :]
            for i in range(batch_size)
        ]
        sub_num_combination_aslist = [
            sub_number_combinations_indices[i, 0:size_composednums_aslist[i], 0:max_num_sub_combs[i], :]
            for i in range(batch_size)
        ]

        with Profile("lang_init"):
            languages = [
                DropLanguage(
                    encoded_passage=passage_encoded_aslist[i],
                    modeled_passage=passage_modeled_aslist[i],
                    passage_mask=passage_mask[i],  # passage_mask_aslist[i],
                    passage_sentence_boundaries=p_sentboundary_wps[i],
                    passage_tokenidx2dateidx=passageidx2dateidx[i],
                    passage_date_values=passage_date_values[i],
                    passage_tokenidx2numidx=passageidx2numberidx[i],
                    passage_num_values=passage_number_values[i],
                    composed_numbers=composed_numbers[i],
                    passage_number_sortedtokenidxs=passage_number_sortedtokenidxs[i],
                    add_num_combination_indices=add_num_combination_aslist[i],
                    sub_num_combination_indices=sub_num_combination_aslist[i],
                    year_differences=year_differences[i],
                    year_differences_mat=year_differences_mat[i],
                    count_num_values=count_values[i],
                    parameters=self._executor_parameters,
                    start_types=None,  # batch_start_types[i],
                    device_id=self.device_id,
                    debug=self._debug,
                    metadata=metadata[i],
                )
                for i in range(batch_size)
            ]

            # This works because all instance-languages have the same action space
            action2idx_map = {rule: i for i, rule in enumerate(languages[0].all_possible_productions())}

        action_indices = list(action2idx_map.values())
        batch_action_indices: List[List[int]] = [action_indices for _ in range(batch_size)]

        """
        While training, we know the correct start-types for all instances and the gold-programs for some.
        For instances,
            #   with gold-programs, we should run a ConstrainedBeamSearch with target_sequences,
            #   with start-types, figure out the valid start-action-ids and run ConstrainedBeamSearch with firststep_allo..
        During Validation, we should **always** be running an un-constrained BeamSearch on the full language
        """

        # print(qtypes)
        # print(program_supervised)
        # print(gold_action_seqs)

        mml_loss = 0
        with Profile("beam-sear"):
            if self.training:
                # If any instance is provided with goldprog, we need to divide the batch into supervised / unsupervised
                # and run fully-constrained decoding on supervised, and start-type-constrained-decoding on the rest
                if any(program_supervised):
                    supervised_instances = [i for (i, ss) in enumerate(program_supervised) if ss is True]
                    unsupervised_instances = [i for (i, ss) in enumerate(program_supervised) if ss is False]

                    # List of (gold_actionseq_idxs, gold_actionseq_masks) -- for supervised instances
                    supervised_gold_actionseqs = self._select_indices_from_list(gold_action_seqs, supervised_instances)
                    s_gold_actionseq_idxs, s_gold_actionseq_masks = zip(*supervised_gold_actionseqs)
                    s_gold_actionseq_idxs = list(s_gold_actionseq_idxs)
                    s_gold_actionseq_masks = list(s_gold_actionseq_masks)
                    (supervised_initial_state, _, _) = self.initialState_forInstanceIndices(
                        supervised_instances,
                        languages,
                        actions,
                        encoded_question,
                        question_mask,
                        question_encoded_final_state,
                        question_encoded_aslist,
                        question_mask_aslist,
                    )
                    constrained_search = ConstrainedBeamSearch(
                        self._beam_size,
                        allowed_sequences=s_gold_actionseq_idxs,
                        allowed_sequence_mask=s_gold_actionseq_masks,
                    )

                    supervised_final_states = constrained_search.search(
                        initial_state=supervised_initial_state, transition_function=self._decoder_step
                    )
                    # Computing MML Loss
                    for instance_states in supervised_final_states.values():
                        scores = [state.score[0].view(-1) for state in instance_states]
                        mml_loss += -allenutil.logsumexp(torch.cat(scores))
                    mml_loss = mml_loss / len(supervised_final_states)

                    if len(unsupervised_instances) > 0:
                        (unsupervised_initial_state, _, _) = self.initialState_forInstanceIndices(
                            unsupervised_instances,
                            languages,
                            actions,
                            encoded_question,
                            question_mask,
                            question_encoded_final_state,
                            question_encoded_aslist,
                            question_mask_aslist,
                        )

                        unsupervised_answer_types: List[List[str]] = self._select_indices_from_list(
                            answer_program_start_types, unsupervised_instances
                        )
                        unsupervised_numcomposition_types: List[Set[str]] = self._select_indices_from_list(
                            composed_num_ans_composition_types, unsupervised_instances
                        )
                        unsupervised_action_indices: List[List[int]] = self._select_indices_from_list(
                            batch_action_indices, unsupervised_instances)

                        # unsupervised_ins_start_actionids: List[Set[int]] = \
                        unsupervised_action_prefixes: List[List[List[int]]] = self.get_valid_start_actionids(
                            answer_types=unsupervised_answer_types,
                            action2actionidx=action2idx_map,
                            valid_numcomposition_types=unsupervised_numcomposition_types,
                        )

                        prefixed_beam_search = PrefixedConstrainedBeamSearch(
                            beam_size=self._beam_size, allowed_sequences=unsupervised_action_prefixes,
                            all_action_indices=unsupervised_action_indices)

                        unsup_final_states = prefixed_beam_search.search(
                            initial_state=unsupervised_initial_state,
                            transition_function=self._decoder_step,
                            num_steps=self._max_decoding_steps,
                            keep_final_unfinished_states=False,
                        )
                    else:
                        unsup_final_states = []

                    # Merge final_states for supervised and unsupervised instances
                    best_final_states = self.merge_final_states(
                        supervised_final_states, unsup_final_states, supervised_instances, unsupervised_instances
                    )
                else:
                    # No program-supervision in this batch; run beam-search w/ starting action(s) known
                    (initial_state, _, _) = self.getInitialDecoderState(
                        languages,
                        actions,
                        encoded_question,
                        question_mask,
                        question_encoded_final_state,
                        question_encoded_aslist,
                        question_mask_aslist,
                        batch_size,
                    )
                    batch_action_prefixes: List[List[List[int]]] = self.get_valid_start_actionids(
                        answer_types=answer_program_start_types,
                        action2actionidx=action2idx_map,
                        valid_numcomposition_types=composed_num_ans_composition_types,
                    )
                    prefixed_beam_search = PrefixedConstrainedBeamSearch(
                        beam_size=self._beam_size, allowed_sequences=batch_action_prefixes,
                        all_action_indices=batch_action_indices)
                    # Mapping[int, Sequence[StateType]])
                    best_final_states = prefixed_beam_search.search(
                        initial_state=initial_state,
                        transition_function=self._decoder_step,
                        num_steps=self._max_decoding_steps,
                        keep_final_unfinished_states=False,
                    )
            elif not self.interpret:
                # Prediction-mode; Run beam-search
                (initial_state, _, _) = self.getInitialDecoderState(
                    languages,
                    actions,
                    encoded_question,
                    question_mask,
                    question_encoded_final_state,
                    question_encoded_aslist,
                    question_mask_aslist,
                    batch_size,
                )
                # This is unconstrained beam-search
                best_final_states = self._decoder_beam_search.search(
                    self._max_decoding_steps, initial_state, self._decoder_step, keep_final_unfinished_states=False
                )
            else:
                # Running constrained decoding for dev-set assuming that all dev examples have gold-program labels
                assert self.training is False and self.interpret is True
                assert all(program_supervised)
                # assert all([x is not "UNK" for x in qtypes])
                actionseq_idxs, actionseq_masks = zip(*gold_action_seqs)
                actionseq_idxs = list(actionseq_idxs)
                actionseq_masks = list(actionseq_masks)

                (initial_state, _, _) = self.getInitialDecoderState(
                    languages,
                    actions,
                    encoded_question,
                    question_mask,
                    question_encoded_final_state,
                    question_encoded_aslist,
                    question_mask_aslist,
                    batch_size,
                )
                constrained_search = ConstrainedBeamSearch(
                    self._beam_size,
                    allowed_sequences=actionseq_idxs,
                    allowed_sequence_mask=actionseq_masks,
                )

                best_final_states = constrained_search.search(
                    initial_state=initial_state, transition_function=self._decoder_step
                )

            # batch_actionidxs: List[List[List[int]]]: All action sequence indices for each instance in the batch
            # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
            # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
            # batch_actionseq_probs: List[torch.FloatTensor]: Tensor containing normalized_prog_probs for each instance - no longer
            # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
            # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
            # since the action_ids are assigned based on the order passed there.
            (
                batch_actionidxs,
                batch_actionseqs,
                batch_actionseq_logprobs,
                batch_actionseq_sideargs,
            ) = semparse_utils._convert_finalstates_to_actions(
                    best_final_states=best_final_states, possible_actions=actions, batch_size=batch_size
            )
            batch_actionseq_probs = [[torch.exp(logprob) for logprob in instance_programs]
                                     for instance_programs in batch_actionseq_logprobs]

            self.add_aux_supervision_to_program_side_args(languages=languages,
                                                          batch_action_seqs=batch_actionseqs,
                                                          batch_actionseq_sideargs=batch_actionseq_sideargs,
                                                          program_supervised=program_supervised,
                                                          gold_program_dicts=gold_program_dicts,
                                                          gold_function2actionidx_maps=gold_function2actionidx_maps)

            self.qrepr_module_exec.add_weighted_question_vector_to_sideargs(
                batch_action_seqs=batch_actionseqs,
                batch_actionseq_sideargs=batch_actionseq_sideargs,
                text_field_embedder=self._text_field_embedder,
                question_encoding=encoded_question,
                question_mask=question_mask,
                question=question,
                question_passage=question_passage,
                max_ques_len=self.max_ques_len,
                device_id=self.device_id,
                question_attn_mask_threshold=0.1)

        # PRINT PRED PROGRAMS
        # for idx, instance_progs in enumerate(batch_actionseqs):
        #     print(f"--------  InstanceIdx: {idx}  ----------")
        #     print(metadata[idx]["question"])
        #     probs = batch_actionseq_probs[idx]
        #     logprobs = batch_actionseq_logprobs[idx]
        #     start_types = answer_program_start_types[idx]
        #     numcomposition_types = composed_num_ans_composition_types[idx]
        #     print(f"AnsTypes: {start_types} \t CompositionFunctions:{numcomposition_types}")
        #     for prog, prob, logprob in zip(instance_progs, probs, logprobs):
        #         print(f"{prob}: {languages[idx].action_sequence_to_logical_form(prog)} -- {logprob}")
        # print()
        # import pdb
        # pdb.set_trace()

        # with Profile("get-deno"):
        # List[List[Any]], List[List[str]]: Denotations and their types for all instances
        batch_denotations, batch_denotation_types = self._get_denotations(
            batch_actionseqs, languages, batch_actionseq_sideargs
        )

        if self.training and self.shared_substructure:
        # if self.shared_substructure:
            orig_action_seqs: List[List[str]] = []
            orig_module_outs: List[List[Dict]] = []
            for batchidx in range(batch_size):
                instance_actionseqs: List[List[str]] = batch_actionseqs[batchidx]
                # 0-th program is max-score; for instances with gold-program there should ideally be only (gold) program
                if not instance_actionseqs:
                    orig_action_seqs.append([])
                    orig_module_outs.append([])
                    sharedsub_mask[batchidx, :] = 0
                else:
                    orig_action_seqs.append(instance_actionseqs[0])
                    orig_module_outs.append(languages[batchidx].modules_debug_info[0])

            sharedsub_loss = shared_substructure_utils.compute_loss(
                device_id=self.device_id,
                max_ques_len=self.max_ques_len,
                executor_parameters=self._executor_parameters,
                passageidx2dateidx=passageidx2dateidx,
                passageidx2numberidx=passageidx2numberidx,
                qp_encoder=self.qp_encoder,
                languages=languages,
                sharedsub_question_passage=sharedsub_question_passage,
                sharedsub_program_nodes=sharedsub_program_nodes,
                sharedsub_function2actionidx_maps=sharedsub_function2actionidx_maps,
                sharedsub_program_lisp=sharedsub_program_lisp,
                sharedsub_orig_program_lisp=sharedsub_orig_program_lisp,
                orig_sharedsub_postorder_node_idx=orig_sharedsub_postorder_node_idx,
                sharedsub_mask=sharedsub_mask,
                orig_action_seqs=orig_action_seqs,
                orig_program_outputs=orig_module_outs,
                year_differences_mat=year_differences_mat,
                metadata=metadata,
                question_passage=question_passage
            )
            # sharedsub_loss = 5 * sharedsub_loss

            if sharedsub_loss != 0.0:
                self.shrdsubloss_metric(sharedsub_loss.item())
        else:
            sharedsub_loss = 0.0

        output_dict = {}
        # Computing losses if gold answers are given
        if answer_program_start_types is not None:
            # Execution losses --
            total_aux_loss = allenutil.move_to_device(torch.tensor(0.0), self.device_id).float()
            total_aux_loss += sharedsub_loss

            # total_aux_loss += aux_win_loss
            # if aux_win_loss != 0:
            #     self.auxwinloss_metric(aux_win_loss.item())

            if self.excloss:
                exec_loss = 0.0
                batch_exec_loss = 0.0
                execloss_normalizer = 0.0
                for ins_dens in batch_denotations:
                    for den in ins_dens:
                        exec_loss += den.loss
                        if den.loss > 0.0:
                            execloss_normalizer += 1.0
                if execloss_normalizer > 0:
                    batch_exec_loss = exec_loss / execloss_normalizer
                # This check is made explicit here since not all batches have this loss, hence a 0.0 value
                # only bloats the denominator in the metric. This is also done for other losses in below
                if batch_exec_loss != 0.0:
                    self.excloss_metric(batch_exec_loss.item())
                total_aux_loss += batch_exec_loss

            if self.qattloss:
                # Compute Question Attention Supervision auxiliary loss
                qattn_loss = self._ques_attention_loss(batch_actionseq_sideargs, question_mask, metadata)
                # print(qattn_loss)
                if qattn_loss != 0.0:
                    self.qattloss_metric(qattn_loss.item())
                total_aux_loss += qattn_loss

            if self.mmlloss:
                # This is computed above during beam search
                if mml_loss != 0.0:
                    self.mmlloss_metric(mml_loss.item())
                total_aux_loss += mml_loss

            if torch.isnan(total_aux_loss):
                logger.info(f"TotalAuxLoss is nan.")
                total_aux_loss = 0.0

            total_denotation_loss = allenutil.move_to_device(torch.tensor(0.0), self.device_id)
            batch_ansprob_list = []
            for i in range(batch_size):
                # Programs for an instance can be of multiple types;
                # For each program, based on it's return type, we compute the log-likelihood
                # against the appropriate gold-answer and add it to the instance_log_likelihood_list
                # This is then weighed by the program-log-likelihood and added to the batch_loss

                instance_prog_denotations, instance_prog_types = (batch_denotations[i], batch_denotation_types[i])
                instance_progs_logprob_list = batch_actionseq_logprobs[i]

                # This instance does not have completed programs that were found in beam-search
                if len(instance_prog_denotations) == 0:
                    continue

                instance_log_likelihood_list = []
                # new_instance_progs_logprob_list = []
                for progidx in range(len(instance_prog_denotations)):
                    denotation = instance_prog_denotations[progidx]
                    progtype = instance_prog_types[progidx]
                    prog_logprob = instance_progs_logprob_list[progidx]

                    if progtype == "PassageSpanAnswer":
                        # Tuple of start, end log_probs
                        span_answer_loss_inputs = {"passage_span_answer": passage_span_answer[i],
                                                   "passage_span_answer_mask": passage_span_answer_mask[i]}
                        if not self.bio_tagging:
                            span_answer_loss_inputs.update({
                                "span_start_log_probs": denotation.passage_span_start_log_probs,
                                "span_end_log_probs": denotation.passage_span_end_log_probs,
                            })
                        else:
                            span_answer_loss_inputs.update({
                                # "answer_as_list_of_bios": answer_as_list_of_bios[i, :, :],
                                # "span_bio_labels": span_bio_labels[i, :],
                                "log_probs": denotation.bio_logprobs,
                                "passage_mask": passage_mask[i, :],
                            })
                        log_likelihood = self.span_answer.gold_log_marginal_likelihood(**span_answer_loss_inputs)
                        # if epoch is not None and epoch <= 6:
                        #     pattn_loss = self.span_answer.passage_attention_loss(
                        #         passage_attention=denotation.passage_attn, passage_mask=passage_mask[i, :],
                        #         answer_spans_for_possible_taggings=answer_spans_for_possible_taggings[i],
                        #         device_id=self.device_id)
                        #     total_aux_loss += pattn_loss
                    elif progtype == "QuestionSpanAnswer":
                        raise NotImplementedError
                    elif progtype == "YearDifference":
                        # Distribution over year_differences
                        pred_year_difference_dist = denotation._value
                        pred_year_diff_log_probs = torch.log(pred_year_difference_dist + 1e-40)
                        gold_year_difference_dist = allenutil.move_to_device(
                            torch.FloatTensor(answer_as_year_difference[i]), cuda_device=self.device_id
                        )
                        log_likelihood = torch.sum(pred_year_diff_log_probs * gold_year_difference_dist)
                    elif progtype == "PassageNumber":
                        # Distribution over PassageNumbers
                        pred_passagenumber_dist = denotation._value
                        pred_passagenumber_logprobs = torch.log(pred_passagenumber_dist + 1e-40)
                        gold_passagenum_dist = allenutil.move_to_device(
                            torch.FloatTensor(answer_as_passage_number[i]), cuda_device=self.device_id
                        )
                        log_likelihood = torch.sum(pred_passagenumber_logprobs * gold_passagenum_dist)
                    elif progtype == "ComposedNumber":
                        # Distribution over ComposedNumbers
                        pred_composednumber_dist = denotation._value
                        pred_composednumber_logprobs = torch.log(pred_composednumber_dist + 1e-40)
                        gold_composednum_dist = allenutil.move_to_device(
                            torch.FloatTensor(answer_as_composed_number[i]), cuda_device=self.device_id
                        )
                        log_likelihood = torch.sum(pred_composednumber_logprobs * gold_composednum_dist)
                    elif progtype == "CountNumber":
                        count_distribution = denotation._value
                        count_log_probs = torch.log(count_distribution + 1e-40)
                        gold_count_distribution = allenutil.move_to_device(
                            torch.FloatTensor(answer_as_count[i]), cuda_device=self.device_id
                        )
                        log_likelihood = torch.sum(count_log_probs * gold_count_distribution)
                    # Here implement losses for other program-return-types in else-ifs
                    else:
                        raise NotImplementedError
                    if torch.isnan(log_likelihood):
                        logger.info(f"Nan-loss encountered for denotation_log_likelihood")
                        log_likelihood = allenutil.move_to_device(torch.tensor(0.0), self.device_id)
                    if torch.isnan(prog_logprob):
                        logger.info(f"Nan-loss encountered for ProgType: {progtype}")
                        prog_logprob = allenutil.move_to_device(torch.tensor(0.0), self.device_id)

                    # new_instance_progs_logprob_list.append(prog_logprob)
                    instance_log_likelihood_list.append(log_likelihood)

                instance_ansprob_list = [torch.exp(x) for x in instance_log_likelihood_list]
                batch_ansprob_list.append(instance_ansprob_list)
                # Each is the shape of (number_of_progs,)
                # tensor of log p(y_i|z_i)
                instance_denotation_log_likelihoods = torch.stack(instance_log_likelihood_list, dim=-1)
                # print(f"{i}: {instance_log_likelihood_list}")

                # tensor of log p(z_i|x)
                instance_progs_log_probs = torch.stack(instance_progs_logprob_list, dim=-1)
                # instance_progs_log_probs = torch.stack(new_instance_progs_logprob_list, dim=-1)
                # tensor of \log(p(y_i|z_i) * p(z_i|x))
                allprogs_log_marginal_likelihoods = instance_denotation_log_likelihoods + instance_progs_log_probs
                if self.hardem_epoch is not None and epoch is not None and epoch >= (self.hardem_epoch + 1):
                    max_prg_idx = torch.argmax(allprogs_log_marginal_likelihoods)
                    instance_marginal_log_likelihood = allprogs_log_marginal_likelihoods[max_prg_idx]
                else:
                    # tensor of \log[\sum_i \exp (log(p(y_i|z_i) * p(z_i|x)))] = \log[\sum_i p(y_i|z_i) * p(z_i|x)]
                    instance_marginal_log_likelihood = allenutil.logsumexp(allprogs_log_marginal_likelihoods)
                # Added sum to remove empty-dim
                instance_marginal_log_likelihood = torch.sum(instance_marginal_log_likelihood)
                if torch.isnan(instance_marginal_log_likelihood):
                    logger.info(f"Nan-loss encountered for instance_marginal_log_likelihood")
                    logger.info(f"Prog log probs: {instance_progs_log_probs}")
                    logger.info(f"Instance denotation likelihoods: {instance_denotation_log_likelihoods}")

                    instance_marginal_log_likelihood = 0.0
                total_denotation_loss += -1.0 * instance_marginal_log_likelihood

                if torch.isnan(total_denotation_loss):
                    total_denotation_loss = 0.0

            batch_denotation_loss = total_denotation_loss / batch_size
            self.modelloss_metric(batch_denotation_loss.item())
            output_dict["loss"] = batch_denotation_loss + total_aux_loss

        """ DEBUG
        if self.training:
            self.num_train_steps += 1
            if self.num_train_steps % self.log_freq == 0:
                for idx in range(batch_size):
                    question = metadata[idx]["original_question"]
                    action_seqs = batch_actionseqs[idx]
                    probs = batch_actionseq_probs[idx]
                    ans_probs = batch_ansprob_list[idx]
                    self.logfile.write(f"{self.num_train_steps}\t{question}\n")
                    for acseq, prob, ansp in zip(action_seqs, probs, ans_probs):
                        lf = languages[idx].action_sequence_to_logical_form(acseq)
                        self.logfile.write(f"{prob}\t{ansp}\t{lf}\n")
                    self.logfile.write("\n")

            if self.num_train_steps % 5000 == 0:
                print(Profile.to_string())
            if self.num_train_steps >= self.num_log_steps:
                self.logfile.close()
                print(Profile.to_string())
                exit()
        END DEBUG """


        # Get the predicted answers irrespective of loss computation.
        # For each program, given it's return type compute the predicted answer string
        if metadata is not None:
            output_dict["predicted_answer"] = []
            output_dict["all_predicted_answers"] = []

            for i in range(batch_size):
                original_question = metadata[i]["question"]
                original_passage = metadata[i]["passage"]
                passage_wp_offsets = metadata[i]["passage_wp_offsets"]
                passage_token_charidxs = metadata[i]["passage_token_charidxs"]
                passage_tokens = metadata[i]["passage_tokens"]
                p_tokenidx2wpidx = metadata[i]['passage_tokenidx2wpidx']
                instance_year_differences = year_differences[i]
                instance_passage_numbers = passage_number_values[i]
                instance_composed_numbers= composed_numbers[i]
                instance_count_values = count_values[i]
                instance_prog_denotations, instance_prog_types = (batch_denotations[i], batch_denotation_types[i])

                all_instance_progs_predicted_answer_strs: List[str] = []  # List of answers from diff instance progs
                for progidx in range(len(instance_prog_denotations)):
                    denotation = instance_prog_denotations[progidx]
                    progtype = instance_prog_types[progidx]
                    if progtype == "PassageSpanAnswer":
                        span_answer_decode_inputs = {}
                        if not self.bio_tagging:
                            # Single-span predictions are made based on word-pieces
                            span_answer_decode_inputs.update({
                                "span_start_logits": denotation.start_logits,
                                "span_end_logits": denotation.end_logits,
                                "passage_token_offsets": passage_wp_offsets,
                                "passage_text": original_passage,
                            })
                        else:
                            span_answer_decode_inputs.update({
                                'log_probs': denotation.bio_logprobs,
                                'passage_mask': passage_mask[i, :],
                                'p_text': original_passage,
                                'p_tokenidx2wpidx': p_tokenidx2wpidx,
                                'passage_token_charidxs': passage_token_charidxs,
                                'passage_tokens': passage_tokens,
                                'original_question': original_question,
                            })

                        predicted_answer: Union[str, List[str]] = self.span_answer.decode_answer(
                            **span_answer_decode_inputs)

                    elif progtype == "QuestionSpanAnswer":
                        raise NotImplementedError
                    elif progtype == "YearDifference":
                        # Distribution over year_differences vector
                        year_differences_dist = denotation._value.detach().cpu().numpy()
                        predicted_yeardiff_idx = np.argmax(year_differences_dist)
                        # If not predicting year_diff = 0
                        # if predicted_yeardiff_idx == 0 and len(instance_year_differences) > 1:
                        #     predicted_yeardiff_idx = np.argmax(year_differences_dist[1:])
                        #     predicted_yeardiff_idx += 1
                        predicted_year_difference = instance_year_differences[predicted_yeardiff_idx]  # int
                        predicted_answer = str(predicted_year_difference)
                    elif progtype == "PassageNumber":
                        predicted_passagenum_idx = torch.argmax(denotation._value).detach().cpu().numpy()
                        predicted_passage_number = instance_passage_numbers[predicted_passagenum_idx]  # int/float
                        predicted_passage_number = (
                            int(predicted_passage_number)
                            if int(predicted_passage_number) == predicted_passage_number
                            else predicted_passage_number
                        )
                        predicted_answer = str(predicted_passage_number)
                    elif progtype == "ComposedNumber":
                        predicted_composednum_idx = torch.argmax(denotation._value).detach().cpu().numpy()
                        predicted_composed_number = instance_composed_numbers[predicted_composednum_idx]  # int/float
                        predicted_composed_number = (
                            int(predicted_composed_number)
                            if int(predicted_composed_number) == predicted_composed_number
                            else predicted_composed_number
                        )
                        predicted_answer = str(predicted_composed_number)
                    elif progtype == "CountNumber":
                        denotation: CountNumber = denotation
                        count_idx = torch.argmax(denotation._value).detach().cpu().numpy()
                        predicted_count_answer = instance_count_values[count_idx]
                        predicted_count_answer = (
                            int(predicted_count_answer)
                            if int(predicted_count_answer) == predicted_count_answer
                            else predicted_count_answer
                        )
                        predicted_answer = str(predicted_count_answer)

                    else:
                        raise NotImplementedError

                    all_instance_progs_predicted_answer_strs.append(predicted_answer)
                # If no program was found in beam-search
                if len(all_instance_progs_predicted_answer_strs) == 0:
                    all_instance_progs_predicted_answer_strs.append("")
                # Since the programs are sorted by decreasing scores, we can directly take the first pred answer
                instance_predicted_answer = all_instance_progs_predicted_answer_strs[0]
                output_dict["predicted_answer"].append(instance_predicted_answer)
                output_dict["all_predicted_answers"].append(all_instance_progs_predicted_answer_strs)

                answer_annotations = metadata[i].get("answer_annotations", [])
                if answer_annotations:
                    self._drop_metrics(instance_predicted_answer, answer_annotations)

            if not self.training: # and self._debug:
                output_dict["metadata"] = metadata
                output_dict["question_mask"] = question_mask
                output_dict["passage_mask"] = passage_mask
                output_dict["passage_token_idxs"] = passage_token_idxs
                # if answer_as_passage_spans is not None:
                #     output_dict["answer_as_passage_spans"] = answer_as_passage_spans
                output_dict["batch_action_seqs"] = batch_actionseqs
                batch_logical_programs = []
                for instance_idx, instance_actionseqs in enumerate(batch_actionseqs):
                    instance_logical_progs = []
                    for ins_actionseq in instance_actionseqs:
                        logical_form = languages[instance_idx].action_sequence_to_logical_form(ins_actionseq)
                        instance_logical_progs.append(logical_form)
                    batch_logical_programs.append(instance_logical_progs)

                output_dict["logical_forms"] = batch_logical_programs
                output_dict["actionseq_logprobs"] = batch_actionseq_logprobs
                output_dict["actionseq_probs"] = batch_actionseq_probs
                modules_debug_infos = [l.modules_debug_info for l in languages]
                output_dict["modules_debug_infos"] = modules_debug_infos

        return output_dict

    def compute_avg_norm(self, tensor):
        dim0_size = tensor.size()[0]
        dim1_size = tensor.size()[1]

        tensor_norm = tensor.norm(p=2, dim=2).sum() / (dim0_size * dim1_size)

        return tensor_norm

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        model_loss = self.modelloss_metric.get_metric(reset)
        exec_loss = self.excloss_metric.get_metric(reset)
        qatt_loss = self.qattloss_metric.get_metric(reset)
        mml_loss = self.mmlloss_metric.get_metric(reset)
        winloss = self.auxwinloss_metric.get_metric(reset)
        shrdloss = self.shrdsubloss_metric.get_metric(reset)
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        metric_dict.update(
            {
                "em": exact_match,
                "f1": f1_score,
                "ans": model_loss,
                "exc": exec_loss,
                "qatt": qatt_loss,
                # "mml": mml_loss,
                "ss": shrdloss,
                "win": winloss,
            }
        )

        return metric_dict

    @staticmethod
    def get_valid_start_actionids(answer_types: List[List[str]], action2actionidx: Dict[str, int],
                                  valid_numcomposition_types: List[Set[str]] = None) -> List[List[List[int]]]:
        """ For each instance, given answer_types return the set of valid start action ids that return an object of
            that type.
            For example, given start_type as 'PassageSpanAnswer', '@start@ -> PassageSpanAnswer' is a valid start action
            See the reader for all possible values of start_types

            This is used while *training* to run constrained-beam-search to only search through programs for which
            the gold answer is known. These *** shouldn't *** be used during validation
        Parameters:
        -----------
        answer_types: ``List[List[str]]``
            List of answer-types for each instance. These should be (a) possible start-types of the DomainLanguage (DL)
            and (b) "@start@ -> TYPE" action should exist for any TYPE, i.e. they should be types of the DL.
        action2actionidx:
            Mapping action-strings to action-idxs.
        valid_numcomposition_types:
            If answer-type is ComposedNumber, then the function (e.g. add or subtract) that would lead to the answer.
            This helps us constrain the program-search even further
        composed_num_ans_types: ``List[Set[str]]``
            For each instance, if the answer is of ComposedNumber type, then this tells valid compositions.
            Can be used to limit search space
        Returns:
        --------
        start_types: `List[Set[int]]`
            For each instance, a set of possible start_types
        """

        # This is the next action to go from ComposedNumber to a function that could generate it
        composed_num_function_action = "ComposedNumber -> [<PassageNumber,PassageNumber:ComposedNumber>, PassageNumber, PassageNumber]"
        # These two actions satisfies the function signature in composed_num_function_action
        addition_function_action = "<PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_addition"
        subtraction_function_action = "<PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_difference"

        # This is the key that the reader sends in valid_numcomposition_types
        num_composition_type_action_mapping = {
            "passage_num_addition": addition_function_action,
            "passage_num_subtraction": subtraction_function_action,
        }

        # We are aiming to make a List[List[List[int]]] -- for each instance, a list of prefix sequences.
        #  Each prefix sequence is a list of action indices. The prefixes don't need to be of the same length.
        #  If an instance does not have any prefixes, its prefix_sequences_list can remain empty
        action_prefixes: List[List[List[int]]] = []

        for i in range(len(answer_types)):
            instance_answer_types: List[str] = answer_types[i]
            instance_prefix_sequences: List[List[int]] = []
            for start_type in instance_answer_types:
                start_action = "@start@ -> {}".format(start_type)
                start_action_idx = action2actionidx[start_action]
                if start_type == "ComposedNumber":
                    # Containing "passage_num_addition" and/or "passage_num_subtraction"
                    instance_numcomposition_types: Set[str] = valid_numcomposition_types[i]
                    assert len(instance_numcomposition_types) > 0, "No composition type info given"
                    for composition_type in instance_numcomposition_types:
                        composition_function_action = num_composition_type_action_mapping[composition_type]
                        composition_function_action_idx = action2actionidx[composition_function_action]
                        composednumtype_to_funcsign_actionidx = action2actionidx[composed_num_function_action]
                        prefix_sequence = [start_action_idx, composednumtype_to_funcsign_actionidx,
                                           composition_function_action_idx]
                        instance_prefix_sequences.append(prefix_sequence)
                else:
                    prefix_sequence = [start_action_idx]
                    instance_prefix_sequences.append(prefix_sequence)
                # else:
                #     logging.error(f"StartType: {start_type} not present in {answer_type_to_start_action_mapping}")
            # valid_start_action_ids.append(start_action_ids)
            action_prefixes.append(instance_prefix_sequences)

        return action_prefixes

    def synthetic_num_grounding_loss(
        self, qtypes, synthetic_numgrounding_metadata, passage_passage_num_similarity, passageidx2numberidx
    ):
        """
        Parameters:
        -----------
        passage_passage_num_similarity: (B, P, P)
        passage_tokenidx2dateidx: (B, P) containing non -1 values for number tokens. We'll use this for masking.
        synthetic_numgrounding_metadata: For each instance, list of (token_idx, number_idx), i.e.
            for row = token_idx, column = number_idx should be high
        """

        # (B, P)
        passageidx2numberidx_mask = (passageidx2numberidx > -1).float()

        # (B, P, P) -- with each row now normalized for number tokens
        passage_passage_num_attention = allenutil.masked_softmax(
            passage_passage_num_similarity, mask=passageidx2numberidx_mask.unsqueeze(1)
        )
        log_likelihood = 0
        normalizer = 0
        for idx, (qtype, token_number_idx_pairs) in enumerate(zip(qtypes, synthetic_numgrounding_metadata)):
            if qtype == dropconstants.SYN_NUMGROUND_qtype:
                for token_idx, number_idx in token_number_idx_pairs:
                    log_likelihood += torch.log(passage_passage_num_attention[idx, token_idx, number_idx] + 1e-40)
                    normalizer += 1
        if normalizer > 0:
            log_likelihood = log_likelihood / normalizer

        loss = -1 * log_likelihood

        return loss


    def _ques_attention_loss(self,
                             batch_actionseq_sideargs: List[List[List[Dict]]], question_mask: torch.FloatTensor,
                             metadata):
        """Compute question attention loss against the `question_attention_supervision` key in side-args.

        Relevant actions should contain the `question_attention_supervision` key; compute loss against that.
        """
        loss = 0.0
        normalizer = 0
        for instance_idx in range(len(batch_actionseq_sideargs)):
            instance_actionseq_sideargs: List[List[Dict]] = batch_actionseq_sideargs[instance_idx]
            for program_sideargs in instance_actionseq_sideargs:
                for action_sideargs in program_sideargs:
                    if "question_attention_supervision" in action_sideargs:
                        gold_attn: List[int] = action_sideargs["question_attention_supervision"]
                        gold_attn_len = len(gold_attn)
                        gold_attn_tensor: torch.FloatTensor = allenutil.move_to_device(
                            torch.FloatTensor(gold_attn), cuda_device=self.device_id)
                        pred_attn = action_sideargs["question_attention"]
                        pred_attn = pred_attn[0:gold_attn_len]
                        mask = question_mask[instance_idx, 0:gold_attn_len]
                        gold_attn_tensor = gold_attn_tensor * mask
                        pred_attn = pred_attn * mask
                        if torch.sum(gold_attn_tensor) > 0:     # without this we get NaN loss when gold-attn is 0
                            attn_sum = torch.sum(pred_attn * gold_attn_tensor)
                            loss += torch.log(attn_sum + 1e-40)
                            # log_question_attention = torch.log(pred_attn + 1e-40) * mask
                            # loss += torch.sum(log_question_attention * gold_attn_tensor)
                            normalizer += 1

        if normalizer == 0:
            return loss
        else:
            return -1 * (loss / normalizer)


    def getInitialDecoderState(
        self,
        languages: List[DropLanguage],
        actions: List[List[ProductionRule]],
        encoded_question: torch.FloatTensor,
        question_mask: torch.FloatTensor,
        question_encoded_final_state: torch.FloatTensor,
        question_encoded_aslist: List[torch.Tensor],
        question_mask_aslist: List[torch.Tensor],
        batch_size: int,
    ):
        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [encoded_question.new_zeros(1, dtype=torch.float) for _ in range(batch_size)]

        initial_grammar_statelets = []
        batch_action2actionidx: List[Dict[str, int]] = []
        # This is kind of useless, only needed for debugging in BasicTransitionFunction
        batch_actionidx2actionstr: List[List[str]] = []

        for i in range(batch_size):
            (grammar_statelet, action2actionidx, actionidx2actionstr) = self._create_grammar_statelet(
                languages[i], actions[i]
            )

            initial_grammar_statelets.append(grammar_statelet)
            batch_actionidx2actionstr.append(actionidx2actionstr)
            batch_action2actionidx.append(action2actionidx)

        initial_rnn_states = self._get_initial_rnn_state(
            question_encoded=encoded_question,
            question_mask=question_mask,
            question_encoded_finalstate=question_encoded_final_state,
            question_encoded_aslist=question_encoded_aslist,
            question_mask_aslist=question_mask_aslist,
        )

        initial_side_args = [[] for _ in range(batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(
            batch_indices=list(range(batch_size)),
            action_history=[[] for _ in range(batch_size)],
            score=initial_score_list,
            rnn_state=initial_rnn_states,
            grammar_state=initial_grammar_statelets,
            possible_actions=actions,
            extras=batch_actionidx2actionstr,
            debug_info=initial_side_args,
        )

        return (initial_state, batch_action2actionidx, batch_actionidx2actionstr)

    def _select_indices_from_list(self, list, indices):
        new_list = [list[i] for i in indices]
        return new_list

    def initialState_forInstanceIndices(
        self,
        instances_list: List[int],
        languages: List[DropLanguage],
        actions: List[List[ProductionRule]],
        encoded_question: torch.FloatTensor,
        question_mask: torch.FloatTensor,
        question_encoded_final_state: torch.FloatTensor,
        question_encoded_aslist: List[torch.Tensor],
        question_mask_aslist: List[torch.Tensor],
    ):

        s_languages = self._select_indices_from_list(languages, instances_list)
        s_actions = self._select_indices_from_list(actions, instances_list)
        s_encoded_question = encoded_question[instances_list]
        s_question_mask = question_mask[instances_list]
        s_question_encoded_final_state = question_encoded_final_state[instances_list]
        s_question_encoded_aslist = self._select_indices_from_list(question_encoded_aslist, instances_list)
        s_question_mask_aslist = self._select_indices_from_list(question_mask_aslist, instances_list)

        num_instances = len(instances_list)

        return self.getInitialDecoderState(
            s_languages,
            s_actions,
            s_encoded_question,
            s_question_mask,
            s_question_encoded_final_state,
            s_question_encoded_aslist,
            s_question_mask_aslist,
            num_instances,
        )

    def merge_final_states(
        self,
        supervised_final_states,
        unsupervised_final_states,
        supervised_instances: List[int],
        unsupervised_instances: List[int],
    ):

        """ Supervised and unsupervised final_states are dicts with keys in order from 0 - len(dict)
            The final keys are the instances' batch index which is stored in (un)supervised_instances list
            i.e. index = supervised_instances[0] is the index of the instance in the original batch
            whose final state is now in supervised_final_states[0].
            Therefore the final_state should contain; final_state[index] = supervised_final_states[0]
        """

        if len(supervised_instances) == 0:
            return unsupervised_final_states

        if len(unsupervised_instances) == 0:
            return supervised_final_states

        batch_size = len(supervised_instances) + len(unsupervised_instances)

        final_states = {}

        for i in range(batch_size):
            if i in supervised_instances:
                idx = supervised_instances.index(i)
                state_value = supervised_final_states[idx]
                final_states[i] = state_value
            else:
                idx = unsupervised_instances.index(i)
                # Unsupervised instances go through beam search and not always is a program found for them
                # If idx does not exist in unsupervised_final_states, don't add in final_states
                # Only add a instance_idx if it exists in final states -- not all beam-searches result in valid-paths
                if idx in unsupervised_final_states:
                    state_value = unsupervised_final_states[idx]
                    final_states[i] = state_value

        return final_states

    def masking_blockdiagonal(self, passage_length, window, device_id):
        """ Make a (passage_length, passage_length) tensor M of 1 and -1 in which for each row x,
            M[x, y] = -1 if y < x - window or y > x + window, else it is 1.
            Basically for the x-th row, the [x-win, x+win] columns should be 1, and rest -1
        """

        # The lower and upper limit of token-idx that won't be masked for a given token
        lower = allenutil.get_range_vector(passage_length, device=device_id) - window
        upper = allenutil.get_range_vector(passage_length, device=device_id) + window
        lower = torch.clamp(lower, min=0, max=passage_length - 1)
        upper = torch.clamp(upper, min=0, max=passage_length - 1)
        lower_un = lower.unsqueeze(1)
        upper_un = upper.unsqueeze(1)

        # Range vector for each row
        lower_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)
        upper_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)

        # Masks for lower and upper limits of the mask
        lower_mask = lower_range_vector >= lower_un
        upper_mask = upper_range_vector <= upper_un

        # Final-mask that we require
        inwindow_mask = (lower_mask == upper_mask).float()
        outwindow_mask = (lower_mask != upper_mask).float()
        return inwindow_mask, outwindow_mask

    def window_loss_numdate(self, passage_passage_alignment, passage_tokenidx_mask, inwindow_mask, outwindow_mask):
        """
        The idea is to first softmax the similarity_scores to get a distribution over the date/num tokens from each
        passage token.

        For each passage_token,
            -- increase the sum of prob for date/num tokens around it in a window (don't know which date/num is correct)
            -- decrease the prod of prob for date/num tokens outside the window (know that all date/num are wrong)

        Parameters:
        -----------
        passage_passage_similarity_scores: (batch_size, passage_length, passage_length)
            For each passage_token, a similarity score to other passage_tokens for data/num
            This should ideally, already be masked

        passage_tokenidx_mask: (batch_size, passage_length)
            Mask for tokens that are num/date

        inwindow_mask: (passage_length, passage_length)
            For row x, inwindow_mask[x, x-window : x+window] = 1 and 0 otherwise. Mask for a window around the token
        outwindow_mask: (passage_length, passage_length)
            Opposite of inwindow_mask. For each row x, the columns are 1 outside of a window around x
        """
        inwindow_mask = inwindow_mask.unsqueeze(0) * passage_tokenidx_mask.unsqueeze(1)
        inwindow_probs = passage_passage_alignment * inwindow_mask
        # This signifies that each token can distribute it's prob to nearby-date/num in anyway
        # Shape: (batch_size, passage_length)
        sum_inwindow_probs = inwindow_probs.sum(2)
        mask_sum = (inwindow_mask.sum(2) > 0).float()
        # Image a row where mask = 0, there sum of probs will be zero and we need to compute masked_log
        masked_sum_inwindow_probs = allenutil.replace_masked_values(sum_inwindow_probs, mask_sum.bool(),
                                                                    replace_with=1e-40)
        log_sum_inwindow_probs = torch.log(masked_sum_inwindow_probs + 1e-40) * mask_sum
        inwindow_likelihood = torch.sum(log_sum_inwindow_probs)
        if torch.sum(inwindow_mask) > 0:
            inwindow_likelihood = inwindow_likelihood / torch.sum(inwindow_mask)
        else:
            inwindow_likelihood = 0.0

        outwindow_mask = outwindow_mask.unsqueeze(0) * passage_tokenidx_mask.unsqueeze(1)
        # For tokens outside the window, increase entropy of the distribution. i.e. -\sum p*log(p)
        # Since we'd like to distribute the weight equally to things outside the window
        # Shape: (batch_size, passage_length, passage_length)
        outwindow_probs = passage_passage_alignment * outwindow_mask
        masked_outwindow_probs = allenutil.replace_masked_values(outwindow_probs, outwindow_mask.bool(),
                                                                 replace_with=1e-40)
        outwindow_probs_log = torch.log(masked_outwindow_probs + 1e-40) * outwindow_mask
        # Shape: (batch_length, passage_length)
        outwindow_negentropies = torch.sum(outwindow_probs * outwindow_probs_log)

        if torch.sum(outwindow_mask) > 0:
            outwindow_negentropies = outwindow_negentropies / torch.sum(outwindow_mask)
        else:
            outwindow_negentropies = 0.0

        # Increase inwindow likelihod and decrease outwindow-negative-entropy
        loss = -1 * inwindow_likelihood + outwindow_negentropies
        return loss

    def add_aux_supervision_to_program_side_args(self,
                                                 languages: List[DropLanguage],
                                                 batch_action_seqs: List[List[List[str]]],
                                                 batch_actionseq_sideargs: List[List[List[Dict]]],
                                                 program_supervised: List[bool],
                                                 gold_program_dicts: List[List[Union[Dict, None]]],
                                                 gold_function2actionidx_maps: List[List[List[int]]]):
        """ If instance has a gold-program, update the side-args of the predicted program (same as gold) with
            supervision dictionaries from the gold-program.
        """
        if not self.training:
            return

        for instance_idx in range(len(program_supervised)):
            ins_prog_supervised = program_supervised[instance_idx]
            if not ins_prog_supervised:
                continue
            # Currently we assume there is only one gold-program per instance and this gets decoded as top-program
            # in constrained-decoding
            language = languages[instance_idx]
            gold_program_node = qdmr_utils.node_from_dict(gold_program_dicts[instance_idx][0])
            gold_program_lisp = qdmr_utils.nested_expression_to_lisp(gold_program_node.get_nested_expression())
            gold_action_seq: List[str] = language.logical_form_to_action_sequence(gold_program_lisp)
            # TODO(nitish): Faced issue in next line in evaluate once. list index out of range
            pred_action_seq: List[str] = batch_action_seqs[instance_idx][0]
            pred_sideargs: List[Dict] = batch_actionseq_sideargs[instance_idx][0]
            if gold_action_seq == pred_action_seq:
                # These are the action-idxs which produce terminal-node functions; gold_function2actionidx_maps is
                # packed so that as multiple gold-programs are passed. we assume singe gold-program here
                function2actionidx_map: List[int] = gold_function2actionidx_maps[instance_idx][0]
                inorder_supervision_dicts: List[Dict] = qdmr_utils.get_inorder_supervision_list(gold_program_node)
                assert len(function2actionidx_map) == len(inorder_supervision_dicts), "each func. should have a supdict"
                # Append the appropriate pred_sideargs idxs with the supervision-dict
                for supervision_idx, action_idx in enumerate(function2actionidx_map):
                    # supervision_idx: range(0, num_of_functions) - index into inorder_supervision_dicts
                    # action_idx: range(0, len(pred_sideargs)) - index into the actions
                    pred_sideargs[action_idx].update(inorder_supervision_dicts[supervision_idx])









