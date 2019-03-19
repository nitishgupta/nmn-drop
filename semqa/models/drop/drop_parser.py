import logging
from typing import List, Dict, Any, Tuple
import math

from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import Activation
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions import LinkingTransitionFunction
from allennlp.state_machines.trainers.maximum_marginal_likelihood import MaximumMarginalLikelihood
from allennlp.modules.span_extractors import SpanExtractor
import allennlp.nn.util as allenutil
import allennlp.common.util as alcommon_util
from allennlp.models.archival import load_archive
from allennlp.models.decomposable_attention import DecomposableAttention
from allennlp.nn import InitializerApplicator

from semqa.domain_languages.drop import DropLanguage
from semqa.models.drop.drop_parser_base import DROPParserBase
from semqa.domain_languages.drop.execution_parameters import ExecutorParameters

from semqa.data.datatypes import DateField, NumberField
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from semqa.state_machines.transition_functions.linking_transition_func_emb import LinkingTransitionFunctionEmbeddings
from allennlp.training.metrics import Average
from semqa.models.utils.bidaf_utils import PretrainedBidafModelUtils
from semqa.models.utils import generic_utils as genutils
from semqa.models.utils import semparse_utils
import utils.util as myutils

import datasets.drop.constants as dropconstants

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this three tuple, add "QENT:qent QSTR:qstr" in the twp gaps, and join with space to get the logical form
GOLD_BOOL_LF = ("(bool_and (bool_qent_qstr ", ") (bool_qent_qstr", "))")

def getGoldLF(qent1_action, qent2_action, qstr_action):
    qent1, qent2, qstr = qent1_action.split(' -> ')[1], qent2_action.split(' -> ')[1], qstr_action.split(' -> ')[1]
    # These qent1, qent2, and qstr are actions
    return f"{GOLD_BOOL_LF[0]} {qent1} {qstr}{GOLD_BOOL_LF[1]} {qent2} {qstr}{GOLD_BOOL_LF[2]}"


@Model.register("drop_parser")
class DROPSemanticParser(DROPParserBase):
    def __init__(self,
                 vocab: Vocabulary,
                 action_embedding_dim: int,
                 transitionfunc_attention: Attention,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 decoder_beam_search: ConstrainedBeamSearch,
                 max_decoding_steps: int,
                 goldactions: bool = None,
                 aux_goldprog_loss: bool = None,
                 qatt_coverage_loss: bool = None,
                 question_token_repr_key: str = None,
                 context_token_repr_key: str = None,
                 bidafutils: PretrainedBidafModelUtils = None,
                 dropout: float = 0.0,
                 text_field_embedder: TextFieldEmbedder = None,
                 debug: bool = False,
                 initializers: InitializerApplicator = InitializerApplicator()) -> None:

        if bidafutils is not None:
            _text_field_embedder = bidafutils._bidaf_model._text_field_embedder
            phrase_layer = bidafutils._bidaf_model._phrase_layer
            matrix_attention_layer = bidafutils._bidaf_model._matrix_attention
            modeling_layer = bidafutils._bidaf_model._modeling_layer
            # TODO(nitish): explicity making text_field_embedder = None since it is initialized with empty otherwise
            text_field_embedder = None
        elif text_field_embedder is not None:
            _text_field_embedder = text_field_embedder
        else:
            _text_field_embedder = None
            raise NotImplementedError

        super(DROPSemanticParser, self).__init__(vocab=vocab,
                                                 action_embedding_dim=action_embedding_dim,
                                                 dropout=dropout,
                                                 debug=debug)


        question_encoding_dim = phrase_layer.get_output_dim()
        self._decoder_step = LinkingTransitionFunctionEmbeddings(encoder_output_dim=question_encoding_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=transitionfunc_attention,
                                                                 num_start_types=1,
                                                                 activation=Activation.by_name('tanh')(),
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 dropout=dropout)

        self._decoder_beam_search = decoder_beam_search
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

        self._text_field_embedder = _text_field_embedder

        text_embed_dim = self._text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)

        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)
        self._phrase_layer = phrase_layer
        self._matrix_attention = matrix_attention_layer
        self._modeling_layer = modeling_layer

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._executor_parameters = ExecutorParameters(num_highway_layers=num_highway_layers,
                                                       phrase_layer=phrase_layer,
                                                       matrix_attention_layer=matrix_attention_layer,
                                                       modeling_layer=modeling_layer,
                                                       hidden_dim=200)

        initializers(self)

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                passage_number_indices: torch.LongTensor,
                passage_number_entidxs: torch.LongTensor,
                passage_number_values: List[List[int]],
                passage_date_spans: torch.LongTensor,
                passage_date_entidxs: torch.LongTensor,
                passage_date_values: List[List[Tuple[int, int, int]]],
                actions: List[List[ProductionRule]],
                answer_types: List[str] = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                epoch_num: List[int] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size = len(actions)

        if epoch_num is not None:
            # epoch_num in allennlp starts from 0
            epoch = epoch_num[0] + 1
        else:
            epoch = None

        print(passage_number_indices.size())
        print(passage_number_entidxs.size())
        print(passage_date_spans.size())
        print(passage_date_entidxs.size())

        question_mask = allenutil.get_text_field_mask(question).float()
        passage_mask = allenutil.get_text_field_mask(passage).float()
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        # Shape: (batch_size, question_length, encoding_dim)
        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        print(encoded_question.size())
        print(encoded_passage.size())



        """ Parser setup """
        # Shape: (B, encoding_dim)
        question_encoded_final_state = allenutil.get_final_encoder_states(encoded_question,
                                                                          question_mask,
                                                                          self._phrase_layer.is_bidirectional())
        question_encoded_aslist = [encoded_question[i] for i in range(batch_size)]
        question_mask_aslist = [question_mask[i] for i in range(batch_size)]
        passage_encoded_aslist = [encoded_passage[i] for i in range(batch_size)]
        passage_mask_aslist = [passage_mask[i] for i in range(batch_size)]

        languages = [DropLanguage(encoded_question=question_encoded_aslist[i],
                                  encoded_passage=passage_encoded_aslist[i],
                                  question_mask=question_mask_aslist[i],
                                  passage_mask=passage_mask_aslist[i],
                                  parameters=self._executor_parameters) for i in range(batch_size)]

        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for _ in range(batch_size)]

        initial_grammar_statelets = [self._create_grammar_statelet(languages[i], actions[i]) for i in range(batch_size)]

        initial_rnn_states = self._get_initial_rnn_state(question_encoded=encoded_question,
                                                         question_mask=question_mask,
                                                         question_encoded_finalstate=question_encoded_final_state,
                                                         question_encoded_aslist=question_encoded_aslist,
                                                         question_mask_aslist=question_mask_aslist)

        initial_side_args = [[] for _ in range(batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_states,
                                          grammar_state=initial_grammar_statelets,
                                          possible_actions=actions,
                                          debug_info=initial_side_args)

        # Mapping[int, Sequence[StateType]]
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             # firststep_allowed_actions=firststep_action_ids,
                                                             keep_final_unfinished_states=False)

        # batch_actionidxs: List[List[List[int]]]: All action sequence indices for each instance in the batch
        # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
        # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
        # batch_actionseq_probs: List[torch.FloatTensor]: Tensor containing normalized_prog_probs for each instance
        # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (batch_actionidxs,
         batch_actionseqs,
         batch_actionseq_scores,
         batch_actionseq_probs,
         batch_actionseq_sideargs) = semparse_utils._convert_finalstates_to_actions(best_final_states=best_final_states,
                                                                                    possible_actions=actions,
                                                                                    batch_size=batch_size)

        # List[List[Any]], List[List[str]]: Denotations and their types for all instances
        batch_denotations, batch_denotation_types = self._get_denotations(batch_actionseqs,
                                                                          languages,
                                                                          batch_actionseq_sideargs)

        print(batch_denotation_types)


        for instance_progseqs in batch_actionseqs:
            for progseq in instance_progseqs:
                print(progseq)

        raise NotImplementedError



