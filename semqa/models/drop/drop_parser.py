import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import copy

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.modules.matrix_attention import MatrixAttention, DotProductMatrixAttention
from allennlp.state_machines.states import GrammarBasedState
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.state_machines.transition_functions import BasicTransitionFunction
from allennlp.state_machines.trainers.maximum_marginal_likelihood import MaximumMarginalLikelihood
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from semqa.state_machines.transition_functions.linking_transition_func_emb import LinkingTransitionFunctionEmbeddings
from allennlp.training.metrics import Average, DropEmAndF1
from semqa.models.utils.bidaf_utils import PretrainedBidafModelUtils
from semqa.models.utils import semparse_utils

from semqa.models.drop.drop_parser_base import DROPParserBase
from semqa.domain_languages.drop import DropLanguage, Date, ExecutorParameters, QuestionSpanAnswer, PassageSpanAnswer

import datasets.drop.constants as dropconstants
import utils.util as myutils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this three tuple, add "QENT:qent QSTR:qstr" in the twp gaps, and join with space to get the logical form
GOLD_BOOL_LF = ("(bool_and (bool_qent_qstr ", ") (bool_qent_qstr", "))")


def getGoldLF_datecomparison(question_tokens: List[str]):
    # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
    lf1 = "(find_passageSpanAnswer ("
    lf2 = " find_PassageAttention find_PassageAttention))"
    greater_than = "compare_date_greater_than"
    lesser_than = "compare_date_lesser_than"

    # Correct if Attn1 is first event
    lesser_tokens = ['first', 'earlier', 'forst', 'firts']
    greater_tokens = ['later', 'last', 'second']

    for t in lesser_tokens:
        if t in question_tokens:
            return f"{lf1}{lesser_than}{lf2}", "lesser"

    for t in greater_tokens:
        if t in question_tokens:
            return f"{lf1}{greater_than}{lf2}", "greater"

    return f"{lf1}{greater_than}{lf2}", "greater"


def getGoldLF_numcomparison(question_tokens: List[str]):
    # "(find_passageSpanAnswer (compare_date_greater_than find_PassageAttention find_PassageAttention))"
    lf1 = "(find_passageSpanAnswer ("
    lf2 = " find_PassageAttention find_PassageAttention))"
    greater_than = "compare_num_greater_than"
    lesser_than = "compare_num_lesser_than"

    # Correct if Attn1 is first event
    greater_tokens = ['larger', 'more', 'largest', 'bigger', 'higher', 'highest', 'most', 'greater']
    lesser_tokens = ['smaller', 'fewer', 'lowest', 'smallest', 'less', 'least', 'fewest', 'lower']

    for t in lesser_tokens:
        if t in question_tokens:
            return f"{lf1}{lesser_than}{lf2}", "lesser"

    for t in greater_tokens:
        if t in question_tokens:
            return f"{lf1}{greater_than}{lf2}", "greater"

    return f"{lf1}{greater_than}{lf2}", "greater"


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
                 # passage_token_to_date: Seq2SeqEncoder,
                 passage_attention_to_span: Seq2SeqEncoder,
                 decoder_beam_search: ConstrainedBeamSearch,
                 max_decoding_steps: int,
                 goldactions: bool = None,
                 goldprogs: bool = False,
                 denotationloss: bool = True,
                 excloss: bool = False,
                 qattloss: bool = False,
                 mmlloss: bool = False,
                 aux_goldprog_loss: bool = None,
                 qatt_coverage_loss: bool = None,
                 question_token_repr_key: str = None,
                 context_token_repr_key: str = None,
                 bidafutils: PretrainedBidafModelUtils = None,
                 dropout: float = 0.0,
                 text_field_embedder: TextFieldEmbedder = None,
                 debug: bool = False,
                 initializers: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

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

        _text_field_embedder: BasicTextFieldEmbedder = _text_field_embedder

        super(DROPSemanticParser, self).__init__(vocab=vocab,
                                                 action_embedding_dim=action_embedding_dim,
                                                 dropout=dropout,
                                                 debug=debug,
                                                 regularizer=regularizer)


        question_encoding_dim = phrase_layer.get_output_dim()

        self._decoder_step = BasicTransitionFunction(encoder_output_dim=question_encoding_dim,
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=transitionfunc_attention,
                                                     activation=Activation.by_name('tanh')(),
                                                     predict_start_type_separately=False,
                                                     num_start_types=1,
                                                     add_action_bias=False,
                                                     dropout=dropout)
        self._mml = MaximumMarginalLikelihood()
        '''
        self._decoder_step = LinkingTransitionFunctionEmbeddings(encoder_output_dim=question_encoding_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=transitionfunc_attention,
                                                                 num_start_types=1,
                                                                 activation=Activation.by_name('tanh')(),
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 dropout=dropout)
        '''

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
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        # self.passage_token_to_date = passage_token_to_date
        self.dotprod_matrix_attn = DotProductMatrixAttention()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._executor_parameters = ExecutorParameters(num_highway_layers=num_highway_layers,
                                                       phrase_layer=phrase_layer,
                                                       modeling_proj_layer=self._modeling_proj_layer,
                                                       modeling_layer=modeling_layer,
                                                       passage_attention_to_span=passage_attention_to_span,
                                                       hidden_dim=200,
                                                       dropout=dropout)

        self.modelloss_metric = Average()
        self.excloss_metric = Average()
        self.qattloss_metric = Average()
        self.mmlloss_metric = Average()
        self._drop_metrics = DropEmAndF1()

        self._goldactions = goldactions
        self._goldprogs = goldprogs

        # Main loss for QA
        self.denotation_loss = denotationloss
        # Auxiliary losses, such as - Prog-MML, QAttn, DateGrounding etc.
        self.excloss = excloss
        self.qattloss = qattloss
        self.mmlloss = mmlloss

        initializers(self)

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                passageidx2numberidx: torch.LongTensor,
                passage_number_values: List[List[float]],
                passageidx2dateidx: torch.LongTensor,
                passage_date_values: List[List[Date]],
                actions: List[List[ProductionRule]],
                datecomp_ques_event_date_groundings: List[Tuple[List[int], List[int]]] = None,
                numcomp_qspan_num_groundings: List[Tuple[List[int], List[int]]] = None,
                strongly_supervised: List[bool] = None,
                qtypes: List[str] = None,
                qattn_supervision: torch.FloatTensor = None,
                answer_types: List[str] = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                epoch_num: List[int] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size = len(actions)
        device_id = self._get_prediction_device()

        # if epoch_num is not None:
        #     # epoch_num in allennlp starts from 0
        #     epoch = epoch_num[0] + 1
        # else:
        #     epoch = None

        question_mask = allenutil.get_text_field_mask(question).float()
        passage_mask = allenutil.get_text_field_mask(passage).float()

        rawemb_question = self._dropout(self._text_field_embedder(question))
        rawemb_passage = self._dropout(self._text_field_embedder(passage))

        embedded_question = self._highway_layer(self._embedding_proj_layer(rawemb_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(rawemb_passage))

        embedded_question = self._dropout(embedded_question)
        embedded_passage = self._dropout(embedded_passage)

        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        # # stripped version
        # projected_embedded_question = rawemb_question
        # projected_embedded_passage = rawemb_passage

        # Shape: (batch_size, question_length, encoding_dim)
        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        if self._debug:
            rawemb_passage_norm = self.compute_avg_norm(rawemb_passage)
            print(f"Raw embedded passage Norm: {rawemb_passage_norm}")

            projected_embedded_passage_norm = self.compute_avg_norm(projected_embedded_passage)
            print(f"Projected embedded passage Norm: {projected_embedded_passage_norm}")

            encoded_passage_norm = self.compute_avg_norm(encoded_passage)
            print(f"Encoded passage Norm: {encoded_passage_norm}")

        # Shape: (batch_size, question_length, passage_length)
        question_passage_similarity = self._executor_parameters.dotprod_matrix_attn(rawemb_question,
                                                                                    rawemb_passage)
        question_passage_similarity = self._dropout(question_passage_similarity)
        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = allenutil.masked_softmax(question_passage_similarity,
                                                              passage_mask.unsqueeze(1),
                                                              memory_efficient=True)

        # Shape: (batch_size, passage_length, passage_length)
        passage_passage_token2date_similarity = self._executor_parameters.passage_to_date_attention(
                                                        encoded_passage, encoded_passage)
        passage_passage_token2date_similarity = self._dropout(passage_passage_token2date_similarity)
        passage_passage_token2date_similarity = passage_passage_token2date_similarity * passage_mask.unsqueeze(1)
        passage_passage_token2date_similarity = passage_passage_token2date_similarity * passage_mask.unsqueeze(2)

        # Shape: (batch_size, passage_length)
        passage_tokenidx2dateidx_mask = (passageidx2dateidx > -1).float()
        # Shape: (batch_size, passage_length, passage_length)
        passage_passage_token2date_similarity = (passage_passage_token2date_similarity *
                                                 passage_tokenidx2dateidx_mask.unsqueeze(1))

        # Shape: (batch_size, passage_length, passage_length)
        passage_passage_token2num_similarity = self._executor_parameters.passage_to_num_attention(
            encoded_passage, encoded_passage)
        passage_passage_token2num_similarity = self._dropout(passage_passage_token2num_similarity)
        passage_passage_token2num_similarity = passage_passage_token2num_similarity * passage_mask.unsqueeze(1)
        passage_passage_token2num_similarity = passage_passage_token2num_similarity * passage_mask.unsqueeze(2)

        # Shape: (batch_size, passage_length)
        passage_tokenidx2numidx_mask = (passageidx2numberidx > -1).float()
        # Shape: (batch_size, passage_length, passage_length)
        passage_passage_token2num_similarity = (passage_passage_token2num_similarity *
                                                passage_tokenidx2numidx_mask.unsqueeze(1))


        '''
        passage_token2date_encoding = self.passage_token_to_date(encoded_passage, passage_mask)
        scaled_dim = passage_token2date_encoding.size()[-1] ** 0.5
        # Shape: (batch_size, passage_length, passage_length)
        passage_token2datetoken_similarity = self.dotprod_matrix_attn(passage_token2date_encoding,
                                                                      passage_token2date_encoding) / scaled_dim

        passage_token2datetoken_similarity = allenutil.replace_masked_values(passage_token2datetoken_similarity,
                                                                             passage_mask.unsqueeze(2), 0.0)

        passage_token2datetoken_similarity = allenutil.replace_masked_values(passage_token2datetoken_similarity,
                                                                             passage_mask.unsqueeze(1), 0.0)
        '''

        '''
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = allenutil.masked_softmax(
            passage_question_similarity,
            question_mask,
            memory_efficient=True)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = allenutil.weighted_sum(encoded_question, passage_question_attention)

        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = allenutil.masked_softmax(
            passage_question_similarity.transpose(1, 2),
            passage_mask,
            memory_efficient=True)
        # Shape: (batch_size, passage_length, passage_length)
        attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = allenutil.weighted_sum(encoded_passage, attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
            torch.cat([encoded_passage, passage_question_vectors,
                       encoded_passage * passage_question_vectors,
                       encoded_passage * passage_passage_vectors],
                      dim=-1)
        )

        # Shape: (batch_size, passage_length, modeling_dim)
        modeled_passage_input = self._modeling_proj_layer(merged_passage_attention_vectors)
        modeled_passage = self._dropout(self._modeling_layer(modeled_passage_input, passage_mask))
        '''

        """ Parser setup """
        # Shape: (B, encoding_dim)
        question_encoded_final_state = allenutil.get_final_encoder_states(encoded_question,
                                                                          question_mask,
                                                                          self._phrase_layer.is_bidirectional())
        question_rawemb_aslist = [rawemb_question[i] for i in range(batch_size)]
        question_embedded_aslist = [projected_embedded_question[i] for i in range(batch_size)]
        question_encoded_aslist = [encoded_question[i] for i in range(batch_size)]
        question_mask_aslist = [question_mask[i] for i in range(batch_size)]
        passage_rawemb_aslist = [rawemb_passage[i] for i in range(batch_size)]
        passage_embedded_aslist = [projected_embedded_passage[i] for i in range(batch_size)]
        passage_encoded_aslist = [encoded_passage[i] for i in range(batch_size)]
        # passage_modeled_aslist = [modeled_passage[i] for i in range(batch_size)]
        passage_mask_aslist = [passage_mask[i] for i in range(batch_size)]
        q2p_attention_aslist = [question_passage_attention[i] for i in range(batch_size)]
        p2pdate_similarity_aslist = [passage_passage_token2date_similarity[i] for i in range(batch_size)]
        p2pnum_similarity_aslist = [passage_passage_token2num_similarity[i] for i in range(batch_size)]
        # passage_token2datetoken_sim_aslist = [passage_token2datetoken_similarity[i] for i in range(batch_size)]

        # Based on the gold answer - figure out possible start types for the instance_language
        if answer_as_passage_spans is not None:
            # batch_start_types = self.find_valid_start_states(answer_as_question_spans=answer_as_question_spans,
            #                                                  answer_as_passage_spans=answer_as_passage_spans,
            #                                                  batch_size=batch_size)

            # TODO(nitish): Only making programs which have PassageSpan as answers
            batch_start_types = [set([PassageSpanAnswer]) for _ in range(batch_size)]
        else:
            batch_start_types = [None for _ in range(batch_size)]

        languages = [DropLanguage(rawemb_question=question_rawemb_aslist[i],
                                  embedded_question=question_embedded_aslist[i],
                                  encoded_question=question_encoded_aslist[i],
                                  rawemb_passage=passage_rawemb_aslist[i],
                                  embedded_passage=passage_embedded_aslist[i],
                                  encoded_passage=passage_encoded_aslist[i],
                                  # modeled_passage=passage_modeled_aslist[i],
                                  # passage_token2datetoken_sim=None, #passage_token2datetoken_sim_aslist[i],
                                  question_mask=question_mask_aslist[i],
                                  passage_mask=passage_mask_aslist[i],
                                  passage_tokenidx2dateidx=passageidx2dateidx[i],
                                  passage_date_values=passage_date_values[i],
                                  passage_tokenidx2numidx=passageidx2numberidx[i],
                                  passage_num_values=passage_number_values[i],
                                  question_passage_attention=q2p_attention_aslist[i],
                                  passage_token2date_similarity=p2pdate_similarity_aslist[i],
                                  passage_token2num_similarity=p2pnum_similarity_aslist[i],
                                  parameters=self._executor_parameters,
                                  start_types=batch_start_types[i],
                                  device_id=device_id,
                                  question_to_use='encoded',
                                  passage_to_use='encoded',
                                  debug=self._debug,
                                  metadata=metadata[i]) for i in range(batch_size)]

        '''
        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for _ in range(batch_size)]

        initial_grammar_statelets = []
        batch_action2actionidx: List[Dict[str, int]] = []
        # This is kind of useless, only needed for debugging in BasicTransitionFunction
        batch_actionidx2actionstr: List[List[str]] = []
        for i in range(batch_size):
            (grammar_statelet,
             action2actionidx,
             actionidx2actionstr) = self._create_grammar_statelet(languages[i], actions[i])
            initial_grammar_statelets.append(grammar_statelet)
            batch_actionidx2actionstr.append(actionidx2actionstr)
            batch_action2actionidx.append(action2actionidx)

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
                                          extras=batch_actionidx2actionstr,
                                          debug_info=initial_side_args)
        '''

        (initial_state,
         batch_action2actionidx,
         batch_actionidx2actionstr) = self.getInitialDecoderState(question, languages, actions, encoded_question,
                                                                  question_mask, question_encoded_final_state,
                                                                  question_encoded_aslist, question_mask_aslist,
                                                                  batch_size)


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

        if self._goldactions:
            # Gold programs
            if self._goldprogs:
                gold_batch_actionseqs = []
                for idx in range(batch_size):
                    question_tokens = metadata[idx]["question_tokens"]
                    qtype = qtypes[idx]
                    if qtype == dropconstants.DATECOMP_QTYPE:
                        gold_lf, _ = getGoldLF_datecomparison(question_tokens)
                    elif qtype == dropconstants.NUMCOMP_QTYPE:
                        gold_lf, _ = getGoldLF_numcomparison(question_tokens)
                    else:
                        raise NotImplementedError
                    gold_action_seq: List[str] = languages[idx].logical_form_to_action_sequence(gold_lf)
                    gold_batch_actionseqs.append([gold_action_seq])
                batch_actionseqs = gold_batch_actionseqs
            # # Gold attn replacement
            # # List[Tuple[torch.Tensor, torch.Tensor]]
            # datecompare_gold_qattns = self.get_gold_quesattn_datecompare(metadata, encoded_question.size()[1],
            #                                                              device_id)
            # self.datecompare_goldattn_to_sideargs(batch_actionseqs,
            #                                       batch_actionseq_sideargs,
            #                                       datecompare_gold_qattns)

        # Adding Date-Comparison supervised event groundings to relevant actions
        self.datecompare_eventdategr_to_sideargs(batch_actionseqs,
                                                 batch_actionseq_sideargs,
                                                 datecomp_ques_event_date_groundings,
                                                 device_id)

        self.numcompare_eventnumgr_to_sideargs(batch_actionseqs,
                                               batch_actionseq_sideargs,
                                               numcomp_qspan_num_groundings,
                                               device_id)

        # For printing predicted - programs
        # for idx, instance_progs in enumerate(batch_actionseqs):
        #     print(f"InstanceIdx:{idx}")
        #     print(metadata[idx]["question_tokens"])
        #     scores = batch_actionseq_scores[idx]
        #     for prog, score in zip(instance_progs, scores):
        #         # print(f"{languages[idx].action_sequence_to_logical_form(prog)} : {score}")
        #         print(f"{prog} : {score}")
        #     print("\n")

        # List[List[Any]], List[List[str]]: Denotations and their types for all instances
        batch_denotations, batch_denotation_types = self._get_denotations(batch_actionseqs,
                                                                          languages,
                                                                          batch_actionseq_sideargs)

        output_dict = {}
        ''' Computing losses if gold answers are given '''
        if answer_types is not None:
            # Execution losses --
            total_aux_loss = allenutil.move_to_device(torch.tensor(0.0), device_id)
            if self.excloss:
                exec_loss = 0.0
                execloss_normalizer = 0.0
                for ins_dens in batch_denotations:
                    for den in ins_dens:
                        execloss_normalizer += 1.0
                        exec_loss += den.loss
                batch_exec_loss = exec_loss / execloss_normalizer
                # This check is made explicit here since not all batches have this loss, hence a 0.0 value
                # only bloats the denominator in the metric. This is also done for other losses in below
                if batch_exec_loss != 0.0:
                    self.excloss_metric(batch_exec_loss.item())
                total_aux_loss += batch_exec_loss

            if self.qattloss:
                # Compute Question Attention Supervision auxiliary loss
                qattn_loss = self._ques_attention_loss(batch_actionseqs,
                                                       batch_actionseq_sideargs,
                                                       qtypes,
                                                       strongly_supervised,
                                                       qattn_supervision)
                if qattn_loss != 0.0:
                    self.qattloss_metric(qattn_loss.item())
                total_aux_loss += qattn_loss


            if self.mmlloss:
                (gold_actionseq_idxs,
                 gold_actionseq_mask) = self._gold_actionseq_forMML(qtypes=qtypes,
                                                                    strongly_supervised=strongly_supervised,
                                                                    question_tokens=[metadata[idx]["question_tokens"]
                                                                                     for idx in range(batch_size)],
                                                                    languages=languages,
                                                                    batch_action2actionidx=batch_action2actionidx,
                                                                    device_id=device_id)
                # for i in range(batch_size):
                #     print(metadata[i]["question_tokens"])
                #     actionstr_seq = [batch_actionidx2actionstr[i][ii] for ii in gold_actionseq_idxs[i][0]]
                #     lf = languages[i].action_sequence_to_logical_form(actionstr_seq)
                #     print(lf)

                mml_loss = self._mml.decode(initial_state=initial_state,
                                            transition_function=self._decoder_step,
                                            supervision=(gold_actionseq_idxs, gold_actionseq_mask))['loss']

                if mml_loss != 0.0:
                    self.mmlloss_metric(mml_loss.item())
                total_aux_loss += mml_loss


            denotation_loss = allenutil.move_to_device(torch.tensor(0.0), device_id)
            if self.denotation_loss:
                for i in range(batch_size):
                    if answer_types[i] == dropconstants.SPAN_TYPE:
                        instance_prog_denotations, instance_prog_types = batch_denotations[i], batch_denotation_types[i]
                        # Tensor with shape: (num_of_programs, )
                        instance_prog_probs = batch_actionseq_probs[i]
                        instance_progs_logprob_list = batch_actionseq_scores[i]
                        instance_log_likelihood_list = []
                        for progidx in range(len(instance_prog_denotations)):
                            denotation = instance_prog_denotations[progidx]
                            progtype = instance_prog_types[progidx]
                            if progtype == "PassageSpanAnswer":
                                denotation: PassageSpanAnswer = denotation
                                log_likelihood = self._get_span_answer_log_prob(answer_as_spans=answer_as_passage_spans[i],
                                                                                span_log_probs=denotation._value)
                                if torch.isnan(log_likelihood) == 1:
                                    print("\nPassageSpan")
                                    print(denotation.start_logits)
                                    print(denotation.end_logits)
                                    print(denotation._value)
                            else:
                                raise NotImplementedError
                            '''
                            elif progtype == "QuestionSpanAnswer":
                                denotation: QuestionSpanAnswer = denotation
                                log_likelihood = self._get_span_answer_log_prob(
                                    answer_as_spans=answer_as_question_spans[i],
                                    span_log_probs=denotation._value)
                                if torch.isnan(log_likelihood) == 1:
                                    print("\nQuestionSpan")
                                    print(denotation.start_logits)
                                    print(denotation.end_logits)
                                    print(denotation._value)
                            '''

                            instance_log_likelihood_list.append(log_likelihood)

                        # Each is the shape of (number_of_progs,)
                        instance_denotation_log_likelihoods = torch.stack(instance_log_likelihood_list, dim=-1)
                        instance_progs_log_probs = torch.stack(instance_progs_logprob_list, dim=-1)

                        allprogs_log_marginal_likelihoods = instance_denotation_log_likelihoods + instance_progs_log_probs
                        instance_marginal_log_likelihood = allenutil.logsumexp(allprogs_log_marginal_likelihoods)
                        # Added sum to remove empty-dim
                        instance_marginal_log_likelihood = torch.sum(instance_marginal_log_likelihood)
                        denotation_loss += -1.0 * instance_marginal_log_likelihood
                    else:
                        raise NotImplementedError

            batch_denotation_loss = denotation_loss / batch_size

            self.modelloss_metric(batch_denotation_loss.item())
            output_dict["loss"] = batch_denotation_loss + total_aux_loss

        if metadata is not None:
            batch_question_strs = [metadata[i]["original_question"] for i in range(batch_size)]
            batch_passage_strs = [metadata[i]["original_passage"] for i in range(batch_size)]
            batch_passage_token_offsets = [metadata[i]["passage_token_offsets"] for i in range(batch_size)]
            batch_question_token_offsets = [metadata[i]["question_token_offsets"] for i in range(batch_size)]
            answer_annotations = [metadata[i]["answer_annotation"] for i in range(batch_size)]
            (batch_best_spans, batch_predicted_answers) = self._get_best_spans(batch_denotations,
                                                                               batch_denotation_types,
                                                                               batch_question_token_offsets,
                                                                               batch_question_strs,
                                                                               batch_passage_token_offsets,
                                                                               batch_passage_strs,
                                                                               question_mask_aslist,
                                                                               passage_mask_aslist)
            predicted_answers = [batch_predicted_answers[i][0] for i in range(batch_size)]

            output_dict['metadata'] = metadata
            output_dict['best_span_ans_str'] = predicted_answers
            output_dict['answer_as_passage_spans'] = answer_as_passage_spans
            output_dict['predicted_spans'] = batch_best_spans

            output_dict["batch_action_seqs"] = batch_actionseqs
            output_dict["batch_actionseq_scores"] = batch_actionseq_scores
            output_dict["batch_actionseq_sideargs"] = batch_actionseq_sideargs
            output_dict["languages"] = languages
            output_dict["all_pred_ansspans"] = batch_predicted_answers

            if answer_annotations:
                for i in range(batch_size):
                    self._drop_metrics(predicted_answers[i], [answer_annotations[i]])

        batch_denotations = None
        batch_denotation_types = None

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
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        metric_dict.update({'em': exact_match, 'f1': f1_score,
                            'ans': model_loss,
                            'exc': exec_loss,
                            'qatt': qatt_loss,
                            'mml': mml_loss})

        return metric_dict

    @staticmethod
    def _get_span_answer_log_prob(answer_as_spans: torch.LongTensor,
                                  span_log_probs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ Compute the log_marginal_likelihood for the answer_spans given log_probs for start/end
            Compute log_likelihood (product of start/end probs) of each ans_span
            Sum the prob (logsumexp) for each span and return the log_likelihood

        Parameters:
        -----------
        answer: ``torch.LongTensor`` Shape: (number_of_spans, 2)
            These are the gold spans
        span_log_probs: ``torch.FloatTensor``
            2-Tuple with tensors of Shape: (length_of_sequence) for span_start/span_end log_probs

        Returns:
        log_marginal_likelihood_for_passage_span
        """

        # Unsqueezing dim=0 to make a batch_size of 1
        answer_as_spans = answer_as_spans.unsqueeze(0)

        span_start_log_probs, span_end_log_probs = span_log_probs
        span_start_log_probs = span_start_log_probs.unsqueeze(0)
        span_end_log_probs = span_end_log_probs.unsqueeze(0)

        # (batch_size, number_of_ans_spans)
        gold_passage_span_starts = answer_as_spans [:, :, 0]
        gold_passage_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = \
            allenutil.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
        clamped_gold_passage_span_ends = \
            allenutil.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = \
            torch.gather(span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = \
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = \
            allenutil.replace_masked_values(log_likelihood_for_spans, gold_passage_span_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = allenutil.logsumexp(log_likelihood_for_spans)

        return log_marginal_likelihood_for_span


    def _ques_attention_loss(self,
                             batch_actionseqs: List[List[List[str]]],
                             batch_actionseq_sideargs: List[List[List[Dict]]],
                             qtypes: List[str],
                             strongly_supervised: List[bool],
                             qattn_supervision: torch.FloatTensor):

        """ Compute QAttn supervision loss for different kind of questions. Different question-types have diff.
            gold-programs and can have different number of qattn-supervision for each instance.
            There, the shape of qattn_supervision is (B, R, QLen) where R is the maximum number of attn-supervisions
            provided for an instance in this batch. For instances with less number of relevant actions
            the corresponding instance_slice will be padded with all zeros-tensors.

            We hard-code the question-types supported, and for each qtype, the relevant actions for which the
            qattn supervision will (should) be provided. For example, the gold program for date-comparison questions
            contains two 'PassageAttention -> find_PassageAttention' actions which use the question_attention sidearg
            for which the supervision is provided. Hence, qtype2relevant_actions_list - contains the two actions for the
            date-comparison question.

            The loss computed is the negative-log of sum of relevant probabilities.

            NOTE: This loss is only computed for instances that are marked as strongly-annotated and hence we don't
            check if the qattns-supervision needs masking.
        """
        qtype2relevant_actions_list = \
               {
                   dropconstants.DATECOMP_QTYPE: ['PassageAttention -> find_PassageAttention',
                                                  'PassageAttention -> find_PassageAttention'],
                   dropconstants.NUMCOMP_QTYPE: ['PassageAttention -> find_PassageAttention',
                                                 'PassageAttention -> find_PassageAttention']
               }

        loss = 0.0
        normalizer = 0

        for ins_idx in range(len(batch_actionseqs)):
            strongly_supervised_instance = strongly_supervised[ins_idx]
            if not strongly_supervised_instance:
                # no point even bothering
                continue
            qtype = qtypes[ins_idx]
            if qtype not in qtype2relevant_actions_list:
                continue
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            # Shape: (R, question_length)
            instance_qattn_supervision = qattn_supervision[ins_idx]
            # These are the actions for which qattn_supervision should be provided.
            relevant_actions = qtype2relevant_actions_list[qtype]
            num_relevant_actions = len(relevant_actions)
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                # Counter to keep a track of which relevant action we're looking for next
                relevant_action_idx = 0
                relevant_action = relevant_actions[relevant_action_idx]
                gold_qattn = instance_qattn_supervision[relevant_action_idx]
                for action, side_arg in zip(program, side_args):
                    if action == relevant_action:
                        question_attention = side_arg['question_attention']
                        # log_question_attention = torch.log(question_attention + 1e-40)
                        # l = torch.sum(log_question_attention * gold_qattn)
                        # loss += l
                        l = torch.sum(question_attention * gold_qattn)
                        loss += torch.log(l)
                        normalizer += 1
                        relevant_action_idx += 1

                        # All relevant actions for this instance in this program are found
                        if relevant_action_idx >= num_relevant_actions:
                            break
                        else:
                            relevant_action = relevant_actions[relevant_action_idx]
                            gold_qattn = instance_qattn_supervision[relevant_action_idx]
        if normalizer == 0:
            return loss
        else:
            return -1 * (loss/normalizer)


    def _gold_actionseq_forMML(self,
                               qtypes: List[str],
                               strongly_supervised: List[bool],
                               question_tokens: List[List[str]],
                               languages: List[DropLanguage],
                               batch_action2actionidx: List[Dict[str, int]],
                               device_id: int) -> Tuple[List[List[List[int]]],
                                                        List[List[List[int]]]]:

        qtypes_supported = [dropconstants.DATECOMP_QTYPE, dropconstants.NUMCOMP_QTYPE]

        gold_actionseq_idxs: List[List[List[int]]] = []
        gold_actionseq_mask: List[List[List[int]]] = []
        for idx in range(len(question_tokens)):
            instance_gold_actionseqs: List[List[int]] = []
            instance_actionseqs_mask: List[List[int]] = []
            qtokens = question_tokens[idx]
            strongly_supervised_ins = strongly_supervised[idx]
            qtype = qtypes[idx]
            action2actionidx = batch_action2actionidx[idx]
            if (not strongly_supervised) or (qtype not in qtypes_supported):
                instance_gold_actionseqs.append([-1])
                instance_actionseqs_mask.append([0])
                gold_actionseq_idxs.append(instance_gold_actionseqs)
                gold_actionseq_mask.append(instance_actionseqs_mask)
            else:
                if qtype == dropconstants.DATECOMP_QTYPE:
                    gold_logicalform, operator = getGoldLF_datecomparison(qtokens)
                    gold_actions: List[str] = languages[idx].logical_form_to_action_sequence(gold_logicalform)
                    actionseq_idxs: List[int] = [action2actionidx[a] for a in gold_actions]
                    actionseq_mask: List[int] = [1 for _ in range(len(actionseq_idxs))]
                elif qtype == dropconstants.NUMCOMP_QTYPE:
                    gold_logicalform, operator = getGoldLF_numcomparison(qtokens)
                    gold_actions: List[str] = languages[idx].logical_form_to_action_sequence(gold_logicalform)
                    actionseq_idxs: List[int] = [action2actionidx[a] for a in gold_actions]
                    actionseq_mask: List[int] = [1 for _ in range(len(actionseq_idxs))]

                else:
                    actionseq_idxs: List[int] = [0]
                    actionseq_mask: List[int] = [0]
                    raise NotImplementedError

                gold_actionseq_idxs.append([actionseq_idxs])
                gold_actionseq_mask.append([actionseq_mask])

        # if device_id > -1:
        #     batch_goldactions_tensor = torch.cuda.LongTensor(batch_actionseq_idxs)
        #     batch_goldactions_mask = torch.cuda.LongTensor(*batch_goldactions_tensor.size(), device=device_id).fill_(1)

        return (gold_actionseq_idxs, gold_actionseq_mask)


    @staticmethod
    def find_valid_start_states(answer_as_question_spans: torch.LongTensor,
                                answer_as_passage_spans: torch.LongTensor,
                                batch_size: int) -> List[Set[Any]]:
        """ Firgure out valid start types based on gold answers
            If answer as question (passage) span exist, QuestionSpanAnswer (PassageSpanAnswer) are valid start types

        answer_as_question_spans: (B, N1, 2)
        answer_as_passage_spans: (B, N2, 2)

        Returns:
        --------
        start_types: `List[Set[Type]]`
            For each instance, a set of possible start_types
        """

        # List containing 0/1 indicating whether a question / passage answer span is given
        passage_span_ans_bool = [((answer_as_passage_spans[i] != -1).sum() > 0) for i in range(batch_size)]
        question_span_ans_bool = [((answer_as_question_spans[i] != -1).sum() > 0) for i in range(batch_size)]

        start_types = [set() for _ in range(batch_size)]
        for i in range(batch_size):
            if passage_span_ans_bool[i] > 0:
                start_types[i].add(PassageSpanAnswer)
            if question_span_ans_bool[i] > 0:
                start_types[i].add(QuestionSpanAnswer)

        return start_types


    @staticmethod
    def _get_best_spans(batch_denotations, batch_denotation_types,
                        question_char_offsets, question_strs, passage_char_offsets, passage_strs, *args):
        """ For all SpanType denotations, get the best span

        Parameters:
        ----------
        batch_denotations: List[List[Any]]
        batch_denotation_types: List[List[str]]
        """

        (question_mask_aslist, passage_mask_aslist) = args

        batch_best_spans = []
        batch_predicted_answers = []

        for instance_idx in range(len(batch_denotations)):
            instance_prog_denotations = batch_denotations[instance_idx]
            instance_prog_types = batch_denotation_types[instance_idx]

            instance_best_spans = []
            instance_predicted_ans = []

            for denotation, progtype in zip(instance_prog_denotations, instance_prog_types):
                # if progtype == "QuestionSpanAnswwer":
                # Distinction between QuestionSpanAnswer and PassageSpanAnswer is not needed currently,
                # since both classes store the start/end logits as a tuple
                # Shape: (2, )
                best_span = get_best_span(span_start_logits=denotation._value[0].unsqueeze(0),
                                          span_end_logits=denotation._value[1].unsqueeze(0)).squeeze(0)
                instance_best_spans.append(best_span)

                predicted_span = tuple(best_span.detach().cpu().numpy())
                if progtype == "QuestionSpanAnswer":
                    try:
                        start_offset = question_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = question_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = question_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"QuesMaskLen: {question_mask_aslist[instance_idx].size()}")
                        print(f"StartLogProbs:{denotation._value[0]}")
                        print(f"EndLogProbs:{denotation._value[1]}")
                        print(f"LenofOffsets: {len(question_char_offsets[instance_idx])}")
                        print(f"QuesStrLen: {len(question_strs[instance_idx])}")

                elif progtype == "PassageSpanAnswer":
                    try:
                        start_offset = passage_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = passage_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = passage_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"PassageMaskLen: {passage_mask_aslist[instance_idx].size()}")
                        print(f"LenofOffsets: {len(passage_char_offsets[instance_idx])}")
                        print(f"PassageStrLen: {len(passage_strs[instance_idx])}")
                else:
                    raise NotImplementedError

                instance_predicted_ans.append(predicted_answer)

            batch_best_spans.append(instance_best_spans)
            batch_predicted_answers.append(instance_predicted_ans)

        return batch_best_spans, batch_predicted_answers


    def datecompare_goldattn_to_sideargs(self,
                                         batch_actionseqs: List[List[List[str]]],
                                         batch_actionseq_sideargs: List[List[List[Dict]]],
                                         batch_gold_attentions: List[Tuple[torch.Tensor, torch.Tensor]]):

        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            instance_gold_attentions = batch_gold_attentions[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                first_qattn = True   # This tells model which qent attention to use
                # print(side_args)
                # print()
                for action, sidearg_dict in zip(program, side_args):
                    if action == 'PassageAttention -> find_PassageAttention':
                        if first_qattn:
                            sidearg_dict['question_attention'] = instance_gold_attentions[0]
                            first_qattn = False
                        else:
                            sidearg_dict['question_attention'] = instance_gold_attentions[1]

    # def get_gold_quesattn_datecompare(self,
    #                                   metadata,
    #                                   masked_len,
    #                                   device_id) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    #     batch_size = len(metadata)
    #
    #     gold_ques_attns = []
    #
    #     for i in range(batch_size):
    #         question_tokens: List[str] = metadata[i]["question_tokens"]
    #         qstr = ' '.join(question_tokens)
    #         assert len(qstr.split(' ')) == len(question_tokens)
    #
    #         gold_ques_attns.append(self._gold_qattn_for_datecompare(qstr, masked_len, device_id))
    #
    #     return gold_ques_attns
    #
    # def _gold_qattn_for_datecompare(self, qstr: str, masked_len: int,
    #                                 device_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """ Get gold question attention for date_compare questions
    #         Question only has one 'or': Which happened first , event A or event B ?
    #         Attn2 is after 'or' until '?'
    #         Attn1 is after first ',' or ':' ('first', 'last', 'later') until the 'or'
    #     """
    #
    #     or_split = qstr.split(' or ')
    #     assert len(or_split) == 2
    #
    #     tokens = qstr.split(' ')
    #
    #     # attn_1 = torch.cuda.FloatTensor(masked_len, device=0).fill_(0.0)
    #     # attn_2 = torch.cuda.FloatTensor(masked_len, device=0).fill_(0.0)
    #
    #     attn_1 = torch.FloatTensor(masked_len).fill_(0.0)
    #     attn_2 = torch.FloatTensor(masked_len).fill_(0.0)
    #
    #     attn_1 = allenutil.move_to_device(attn_1, device_id)
    #     attn_2 = allenutil.move_to_device(attn_2, device_id)
    #
    #     or_idx = tokens.index('or')
    #     # Last token is ? which we don't want to attend to
    #     attn_2[or_idx + 1: len(tokens) - 1] = 1.0
    #     attn_2 = attn_2 / attn_2.sum()
    #
    #     # Gets first index of the item
    #     try:
    #         comma_idx = tokens.index(',')
    #     except:
    #         comma_idx = 100000
    #     try:
    #         colon_idx = tokens.index(':')
    #     except:
    #         colon_idx = 100000
    #
    #     try:
    #         hyphen_idx = tokens.index('-')
    #     except:
    #         hyphen_idx = 100000
    #
    #     split_idx = min(comma_idx, colon_idx, hyphen_idx)
    #
    #     if split_idx == 100000 or (or_idx - split_idx <= 1):
    #         # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
    #         if 'first' in tokens:
    #             split_idx = tokens.index('first')
    #         elif 'second' in tokens:
    #             split_idx = tokens.index('second')
    #         elif 'last' in tokens:
    #             split_idx = tokens.index('last')
    #         elif 'later' in tokens:
    #             split_idx = tokens.index('later')
    #         else:
    #             split_idx = -1
    #
    #     assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"
    #
    #     attn_1[split_idx + 1: or_idx] = 1.0
    #     attn_1 = attn_1 / attn_1.sum()
    #
    #     # return attn_2, attn_1
    #     return attn_1, attn_2


    def datecompare_eventdategr_to_sideargs(self,
                                            batch_actionseqs: List[List[List[str]]],
                                            batch_actionseq_sideargs: List[List[List[Dict]]],
                                            datecomp_ques_event_date_groundings: List[Tuple[List[float], List[float]]],
                                            device_id):
        """ batch_event_date_groundings: For each question, a two-tuple containing the correct date-grounding for the
            two events mentioned in the question.
            These are in order of the annotation (order of events in question) but later the question attention
            might be predicted in reverse order and these will then be the wrong (reverse) annotations. Take care later.
        """
        # List[Tuple[torch.Tensor, torch.Tensor]]
        q_event_date_groundings = self.get_gold_question_event_date_grounding(datecomp_ques_event_date_groundings,
                                                                              device_id)

        relevant_action1 = '<PassageAttention,PassageAttention:PassageAttention_answer> -> compare_date_greater_than'
        relevant_action2 = '<PassageAttention,PassageAttention:PassageAttention_answer> -> compare_date_lesser_than'
        relevant_actions = [relevant_action1, relevant_action2]

        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            event_date_groundings = q_event_date_groundings[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                for action, sidearg_dict in zip(program, side_args):
                    if action in relevant_actions:
                        sidearg_dict['event_date_groundings'] = event_date_groundings


    def numcompare_eventnumgr_to_sideargs(self,
                                            batch_actionseqs: List[List[List[str]]],
                                            batch_actionseq_sideargs: List[List[List[Dict]]],
                                            numcomp_qspan_num_groundings: List[Tuple[List[float], List[float]]],
                                            device_id):
        """ batch_event_num_groundings: For each question, a two-tuple containing the correct num-grounding for the
            two events mentioned in the question.
            These are in order of the annotation (order of events in question) but later the question attention
            might be predicted in reverse order and these will then be the wrong (reverse) annotations. Take care later.
        """
        # List[Tuple[torch.Tensor, torch.Tensor]]
        # Resuing the function written for dates -- should work fine
        q_event_num_groundings = self.get_gold_question_event_date_grounding(numcomp_qspan_num_groundings,
                                                                             device_id)

        relevant_action1 = '<PassageAttention,PassageAttention:PassageAttention_answer> -> compare_num_greater_than'
        relevant_action2 = '<PassageAttention,PassageAttention:PassageAttention_answer> -> compare_num_lesser_than'
        relevant_actions = [relevant_action1, relevant_action2]

        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            event_num_groundings = q_event_num_groundings[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                for action, sidearg_dict in zip(program, side_args):
                    if action in relevant_actions:
                        sidearg_dict['event_num_groundings'] = event_num_groundings


    def get_gold_question_event_date_grounding(self,
                                               question_event_date_groundings: List[Tuple[List[int], List[int]]],
                                               device_id: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """ Converts input event date groundings (date-comparison) to FloatTensors """
        question_date_groundings = []
        for grounding_1, grounding_2 in question_event_date_groundings:
            g1 = allenutil.move_to_device(torch.FloatTensor(grounding_1), device_id)
            g2 = allenutil.move_to_device(torch.FloatTensor(grounding_2), device_id)
            question_date_groundings.append((g1, g2))
        return question_date_groundings


    def passageAnsSpan_to_PassageAttention(self, answer_as_passage_spans, passage_mask):
        """ Convert answers as passage span into passage attention for model introspection

        Parameters:
        ----------
        answer_as_passage_spans: `torch.Tensor`
            Tensor of shape (batch_size, number_of_ans_spans, 2) containing start / end positions
        passage_mask: `torch.FloatTensor`
            Tensor of shape (batch_size, passage_length)

        Returns:
        --------
        attention: `torch.FloatTensor`
            List of (passage_length, ) shaped tensor containing normalized attention for gold spans
        """
        # Shape: (batch_size, number_of_ans_spans, 2)
        answer_as_spans = answer_as_passage_spans.long()

        # TODO(nitish): ONLY USING FIRST CORRECT SPAN OUT OF MULTIPLE POSSIBLE
        # answer_as_spans = answer_as_spans[:, 0, :].unsqueeze(1)

        # Shape: (batch_size, number_of_ans_spans)
        span_starts = answer_as_spans[:, :, 0]
        span_ends = answer_as_spans[:, :, 1]
        answers_mask = (span_starts >= 0).float()

        # Shape: (batch_size, 1, number_of_ans_spans)
        span_starts_ex = span_starts.unsqueeze(1)
        span_ends_ex = span_ends.unsqueeze(1)

        # Idea: Make a range vector from 0 <-> seq_len - 1 and convert into boolean with (val > start) and (val < end)
        # Such items in the sequence are within the span range
        # Shape: (passage_length, )
        range_vector = allenutil.get_range_vector(passage_mask.size(1), allenutil.get_device_of(passage_mask))

        # Shape: (1, passage_length, 1)
        range_vector = range_vector.unsqueeze(0).unsqueeze(2)

        # Shape: (batch_size, passage_length, number_of_ans_spans) - 1 as tokens in the span, 0 otherwise
        span_range_mask = (range_vector >= span_starts_ex).float() * (range_vector <= span_ends_ex).float()
        span_range_mask = span_range_mask * answers_mask.unsqueeze(1)

        # Shape: (batch_size, passage_length)
        unnormalized_attention = span_range_mask.sum(2)
        normalized_attention = unnormalized_attention / unnormalized_attention.sum(1, keepdim=True)

        attention_aslist = [normalized_attention[i, :] for i in range(normalized_attention.size(0))]

        return attention_aslist

    ''' TESTING
    def passageAnsSpan_to_PassageAttention(self, answer_as_passage_spans, passage_mask):
        final_attentions = []

        for i in range(passage_mask.size(0)):
            attn = passage_mask.new_zeros(passage_mask.size(1))
            ans_spans = answer_as_passage_spans[i]
            for span in ans_spans:
                if span[0] > -1:
                    attn[span[0]] = 1.0
                    attn[span[1]] = 1.0

            final_attentions.append(attn)

        return final_attentions
    '''

    def passageattn_to_startendlogits(self, passage_attention, passage_mask):
        span_start_logits = passage_attention.new_zeros(passage_attention.size())
        span_end_logits = passage_attention.new_zeros(passage_attention.size())

        nonzeroindcs = (passage_attention > 0).nonzero()

        startidx = nonzeroindcs[0]
        endidx = nonzeroindcs[-1]

        print(f"{startidx} {endidx}")

        span_start_logits[startidx] = 2.0
        span_end_logits[endidx] = 2.0

        span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask, -1e32)

        span_start_logits += 1e-7
        span_end_logits += 1e-7

        return (span_start_logits, span_end_logits)


    def passage_ans_attn_to_sideargs(self,
                                     batch_actionseqs: List[List[List[str]]],
                                     batch_actionseq_sideargs: List[List[List[Dict]]],
                                     batch_gold_attentions: List[torch.Tensor]):

        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            instance_gold_attention = batch_gold_attentions[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                first_qattn = True   # This tells model which qent attention to use
                # print(side_args)
                # print()
                for action, sidearg_dict in zip(program, side_args):
                    if action == 'PassageSpanAnswer -> find_passageSpanAnswer':
                        sidearg_dict['passage_attention'] = instance_gold_attention


    def getInitialDecoderState(self,
                               question, languages, actions, encoded_question, question_mask,
                               question_encoded_final_state, question_encoded_aslist, question_mask_aslist,
                               batch_size):
        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for _ in range(batch_size)]

        initial_grammar_statelets = []
        batch_action2actionidx: List[Dict[str, int]] = []
        # This is kind of useless, only needed for debugging in BasicTransitionFunction
        batch_actionidx2actionstr: List[List[str]] = []
        for i in range(batch_size):
            (grammar_statelet,
             action2actionidx,
             actionidx2actionstr) = self._create_grammar_statelet(languages[i], actions[i])

            initial_grammar_statelets.append(grammar_statelet)
            batch_actionidx2actionstr.append(actionidx2actionstr)
            batch_action2actionidx.append(action2actionidx)

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
                                          extras=batch_actionidx2actionstr,
                                          debug_info=initial_side_args)

        return (initial_state, batch_action2actionidx, batch_actionidx2actionstr)
