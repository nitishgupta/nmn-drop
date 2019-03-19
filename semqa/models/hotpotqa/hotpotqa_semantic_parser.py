import logging
from typing import List, Dict, Any, Tuple
import math

from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import Activation
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

import semqa.type_declarations.semqa_type_declaration_wques as types
from semqa.domain_languages.hotpotqa import HotpotQALanguage, HotpotQALanguageWSideArgs, HotpotQALanguageWOSideArgs
from semqa.models.hotpotqa.hotpotqa_parser_base import HotpotQAParserBase
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters

from semqa.data.datatypes import DateField, NumberField
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from semqa.state_machines.transition_functions.linking_transition_func_emb import LinkingTransitionFunctionEmbeddings
from allennlp.training.metrics import Average
from semqa.models.utils.bidaf_utils import PretrainedBidafModelUtils
from semqa.models.utils import generic_utils as genutils
import utils.util as myutils

import datasets.hotpotqa.utils.constants as hpcons

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this three tuple, add "QENT:qent QSTR:qstr" in the twp gaps, and join with space to get the logical form
GOLD_BOOL_LF = ("(bool_and (bool_qent_qstr ", ") (bool_qent_qstr", "))")

def getGoldLF(qent1_action, qent2_action, qstr_action):
    qent1, qent2, qstr = qent1_action.split(' -> ')[1], qent2_action.split(' -> ')[1], qstr_action.split(' -> ')[1]
    # These qent1, qent2, and qstr are actions
    return f"{GOLD_BOOL_LF[0]} {qent1} {qstr}{GOLD_BOOL_LF[1]} {qent2} {qstr}{GOLD_BOOL_LF[2]}"


@Model.register("hotpotqa_parser")
class HotpotQASemanticParser(HotpotQAParserBase):
    """
    ``NlvrDirectSemanticParser`` is an ``NlvrSemanticParser`` that gets around the problem of lack
    of logical form annotations by maximizing the marginal likelihood of an approximate set of target
    sequences that yield the correct denotation. The main difference between this parser and
    ``NlvrCoverageSemanticParser`` is that while this parser takes the output of an offline search
    process as the set of target sequences for training, the latter performs search during training.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    action_embedding_dim : ``int``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the TransitionFunction.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        Maximum number of steps for beam search after training.
    dropout : ``float``, optional (default=0.0)
        Probability of dropout to apply on encoder outputs, decoder outputs and predicted actions.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 action_embedding_dim: int,
                 attention: Attention,
                 decoder_beam_search: ConstrainedBeamSearch,
                 executor_parameters: ExecutorParameters,
                 max_decoding_steps: int,
                 wsideargs: bool,
                 goldactions: bool,
                 question_token_repr_key: str,
                 context_token_repr_key: str,
                 bool_qstrqent_func: str,
                 use_quesspan_actionemb: bool,
                 quesspan2actionemb: FeedForward,
                 aux_goldprog_loss: bool,
                 qatt_coverage_loss: bool,
                 entityspan_qatt_loss: bool,
                 bidafutils: PretrainedBidafModelUtils = None,
                 dropout: float = 0.0,
                 text_field_embedder: TextFieldEmbedder = None,
                 qencoder: Seq2SeqEncoder = None,
                 quesspan_extractor: SpanExtractor = None,
                 debug: bool=False,
                 initializers: InitializerApplicator = InitializerApplicator()) -> None:

        if bidafutils is not None:
            _text_field_embedder = bidafutils._bidaf_model._text_field_embedder
            # TODO(nitish): explicity making text_field_embedder = None since it initialized with empty otherwise
            text_field_embedder = None
        elif text_field_embedder is not None:
            _text_field_embedder = text_field_embedder
        else:
            _text_field_embedder = None
            raise NotImplementedError

        super(HotpotQASemanticParser, self).__init__(vocab=vocab,
                                                     action_embedding_dim=action_embedding_dim,
                                                     executor_parameters=executor_parameters,
                                                     text_field_embedder=_text_field_embedder,
                                                     qencoder=qencoder,
                                                     quesspan_extractor=quesspan_extractor,
                                                     quesspan2actionemb=quesspan2actionemb,
                                                     use_quesspan_actionemb=use_quesspan_actionemb,
                                                     wsideargs=wsideargs,
                                                     dropout=dropout,
                                                     debug=debug)

        # Use the gold attention (for w sideargs) or gold actions (for wo sideargs)
        self._goldactions = goldactions
        # Auxiliary loss for predicting gold-parse
        # w/ sideargs: Loss for predicting correct qatt at the correct timesteps in decoding
        # w/o sideargs: MML Loss for predicting the correct action_seq
        self._aux_goldprog_loss = aux_goldprog_loss
        # Auxiliary loss encouraging the program to attend to all tokens (on aggregate) when predicting find_* actions
        self._qattn_coverage_loss = qatt_coverage_loss

        self._mml = None
        if self._wsideargs is False and self._aux_goldprog_loss is True:
            self._mml = MaximumMarginalLikelihood()

        if self._qencoder is not None:
            encoder_output_dim = self._qencoder.get_output_dim()
        elif bidafutils is not None:
            encoder_output_dim = bidafutils._bidaf_encoded_dim
        else:
            raise NotImplementedError

        if attention._normalize is False:
            attention_activation = Activation.by_name('sigmoid')()
        else:
            attention_activation = None

        self._transitionfunc_att = attention

        self._decoder_step = LinkingTransitionFunctionEmbeddings(encoder_output_dim=encoder_output_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=attention,
                                                                 input_attention_activation=None,
                                                                 num_start_types=1,
                                                                 activation=Activation.by_name('tanh')(),
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 dropout=dropout)

        # self._decoder_step = LinkingTransitionFunction(encoder_output_dim=encoder_output_dim,
        #                                                action_embedding_dim=action_embedding_dim,
        #                                                input_attention=attention,
        #                                                num_start_types=1,
        #                                                activation=Activation.by_name('tanh')(),
        #                                                predict_start_type_separately=False,
        #                                                add_action_bias=False,
        #                                                dropout=dropout)

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

        # This str decides which function to use for Bool_Qent_Qstr function during execution
        self._bool_qstrqent_func = bool_qstrqent_func
        # This key tells which question token and context token representations to use in execution
        self._question_token_repr_key = question_token_repr_key
        self._context_token_repr_key = context_token_repr_key

        # Addition loss to enforce high ques attention on mention spans when predicting Qent -> find_Qent action
        self._entityspan_qatt_loss = entityspan_qatt_loss


        if bidafutils is not None:
            self.executor_parameters._bidafutils = bidafutils
            self._bidafutils = bidafutils
        else:
            self._bidafutils = None
        self.executor_parameters._question_token_repr_key = question_token_repr_key
        self.executor_parameters._context_token_repr_key = context_token_repr_key

        # snli_model_archive = load_archive('/srv/local/data/nitishg/semqa/pretrained_decompatt/decomposable-attention-elmo-2018.02.19.tar.gz')
        # self.snli_model: DecomposableAttention = snli_model_archive.model
        # self.executor_parameters._snli_model = self.snli_model

        # weights_dict = torch.load("./resources/semqa/checkpoints/hpqa/b_wsame/hpqa_parser/BS_4/OPT_adam/LR_0.001/Drop_0.2/TOKENS_glove/FUNC_snli/SIDEARG_true/GOLDAC_true/AUXGPLOSS_false/QENTLOSS_false/ATTCOV_false/best.th")
        # print(weights_dict.keys())

        initializers(self)


    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                q_nemenspan2entidx: List[Dict],
                contexts: Dict[str, torch.LongTensor],
                ent_mens: torch.LongTensor,
                num_nents: torch.LongTensor,
                num_mens: torch.LongTensor,
                num_numents: torch.LongTensor,
                date_mens: torch.LongTensor,
                num_dateents: torch.LongTensor,
                num_normval: List[List[NumberField]],
                date_normval: List[List[DateField]],
                languages: List[HotpotQALanguage],
                actions: List[List[ProductionRule]],
                linked_rule2idx: List[Dict],
                quesspanaction2linkingscore: torch.FloatTensor,
                quesspanactions2spanfield: torch.LongTensor,
                gold_ans_type: List[str]=None,
                snli_ques=None,
                snli_contexts=None,
                epoch_num=None,
                **kwargs) -> Dict[str, torch.Tensor]:

        """
        Forward call to the parser.

        Parameters:
        -----------
        question : Dict[str, torch.LongTensor]
            Question from TextField
        q_qstr2idx : List[Dict]
            For each q, Dict from QSTR to idx into the qstr_span_field (q_qstr_spans)
        q_qstr_spans : torch.LongTensor,
            (B, Q_STR, 2) Spanfield for each QSTR in the q. Padding is -1
        q_nemenspan2entidx: List[Dict],
            Map from q_ne_ent span to entity grounding idx.
            This can be used to directly index into the context mentions tensor
        contexts: Dict[str, torch.LongTensor]
            TextField for each context wrapped into a ListField. Appropriate squeeze/unsqueeze needed to work with this
        ent_mens : ``torch.LongTensor``
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_nents : ``torch.LongTensor``
            [B] sized tensor of number of NE entities in the instance
        num_mens : torch.LongTensor
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_numents: torch.LongTensor
            [B] sized tensor of number of NUM entities in the instance
        date_mens : torch.LongTensor
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_dateents : torch.LongTensor
            [B] sized tensor of number of DATE entities in the instance
        num_normval : List[List[NumberField]]
            For each instance, for each NUM entity it's normalized value
        date_normval: List[List[DateField]]
            For each instance, for each DATE entity it's normalized value
        languages: List[HotpotQALanguage]
            DomainLanguage object for each instance
        actions: List[List[ProductionRule]]
            For each instance, List of all possible ProductionRules (global and instance-specific)
        linked_rule2idx: List[Dict[str, int]]
            Dict from instance-specific-actions to idx into linking_score matrix (action2ques_linkingscore) and
            SpanField indices (ques_str_action_spans)
            Currently this is possible since both instance-specific-action types (QSTR & QENT)
        action2ques_linkingscore: torch.FloatTensor
            (B, A, Q_len): Foe each instance, for each instance-specific action, the linking score for LinkingTF.
            For each action (ques_span for us) a binary vector indicating presence of q_token in A span
        ques_str_action_spans: torch.LongTensor
            (B, A, 2): For each instance_specific action (ques span for us) it's span in Q to make action embeddings
        gold_ans_type: List[str]=None
            List of Ans's types from datasets.hotpotqa.constants
        kwargs: ``Dict``
            Is a dictionary containing datatypes for answer_groundings of different types, key being ans_grounding_TYPE
            Each key has a batched_tensor of ground_truth for the instances.
            Since each instance only has a single correct type, the values for the incorrect types are empty vectors ..
            to make batching easier. See the reader code for more details.
        """
        # pylint: disable=arguments-differ

        # snli_output = self.snli_model.forward(premise=snli_ques, hypothesis=snli_ques)
        # snli_probs = snli_output['label_probs']
        # print(snli_probs)
        # exit()
        batch_size = len(languages)
        if 'metadata' in kwargs:
            metadata = kwargs['metadata']
        else:
            metadata = None

        if epoch_num is not None:
            # epoch_num in allennlp starts from 0
            epoch = epoch_num[0] + 1
        else:
            epoch = None

        # print()
        # for i in range(batch_size):
        #     ques = metadata[i]['question']
        #     print(ques)
        #     context_text = metadata[i]['contexts']
        #     for c in context_text:
        #         print(c)
        # print()

        if self._bidafutils is not None:
            (ques_encoded_final_state,
             encoded_ques_tensor, questions_mask_tensor,
             embedded_questions, questions_mask,
             embedded_contexts, contexts_mask,
             encoded_questions,
             encoded_contexts, modeled_contexts) = self._bidafutils.bidaf_reprs(question, contexts)
        elif self._qencoder is not None:
            (embedded_questions, encoded_questions,
             questions_mask, encoded_ques_tensor,
             questions_mask_tensor, ques_encoded_final_state,
             embedded_contexts, contexts_mask) = genutils.embed_and_encode_ques_contexts(
                                                            text_field_embedder=self._text_field_embedder,
                                                            qencoder=self._qencoder,
                                                            batch_size=batch_size,
                                                            question=question,
                                                            contexts=contexts)
            encoded_contexts, modeled_contexts = embedded_contexts, embedded_contexts
        else:
            raise NotImplementedError

        device_id = allenutil.get_device_of(ques_encoded_final_state)

        if gold_ans_type is not None:
            (firststep_action_ids,
             ans_grounding_dict,
             ans_grounding_mask) = self._get_firstStepActionIdxs_GoldAnsDict(gold_ans_type, actions, languages,
                                                                             **kwargs)
        else:
            (firststep_action_ids, ans_grounding_dict, ans_grounding_mask) = (None, None, None)

        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for _ in range(batch_size)]

        if self._use_quesspan_actionemb:
            qspan_action_embeddings = self._questionspan_action_embeddings(
                encoded_question_tensor=encoded_ques_tensor, quesspanactions2spanfield=quesspanactions2spanfield)
        else:
            qspan_action_embeddings = None

        # For each instance, create a grammar statelet containing the valid_actions and their representations
        initial_grammar_statelets = []
        for i in range(batch_size):
            initial_grammar_statelets.append(self._create_grammar_statelet(languages[i],
                                                                           actions[i],
                                                                           linked_rule2idx[i],
                                                                           quesspanaction2linkingscore[i],
                                                                           qspan_action_embeddings[i]))

        # Initial RNN state for the decoder
        initial_rnn_states = self._get_initial_rnn_state(ques_repr=encoded_ques_tensor,
                                                        ques_mask=questions_mask_tensor,
                                                        question_final_repr=ques_encoded_final_state,
                                                        ques_encoded_list=encoded_questions,
                                                        ques_mask_list=questions_mask)

        # Initial side_args list to make the GrammarBasedState if using Language with sideargs
        initial_side_args = None
        if self._wsideargs:
            initial_side_args = [[] for _ in range(0, batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_states,
                                          grammar_state=initial_grammar_statelets,
                                          possible_actions=actions,
                                          debug_info=initial_side_args)

        outputs: Dict[str, torch.Tensor] = {}

        ''' Parsing the question to get action sequences'''
        # Dict[batch_idx(int): List[StateType]]
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             firststep_allowed_actions=firststep_action_ids,
                                                             keep_final_unfinished_states=False)

        # instanceidx2actionseq_idxs: Dict[int, List[List[int]]]
        # instanceidx2actionseq_scores: Dict[int, List[torch.Tensor]]
        # instanceidx2actionseq_sideargs: Dict[int, List[List[Dict]]]
        (instanceidx2actionseq_idxs,
         instanceidx2actionseq_scores,
         instanceidx2actionseq_sideargs) = self._get_actionseq_idxs_and_scores(best_final_states,
                                                                               batch_size,
                                                                               self._wsideargs)

        # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
        # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
        # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (batch_actionseqs,
         batch_actionseq_scores,
         batch_actionseq_sideargs) = self._get_actionseq_strings(actions,
                                                                 instanceidx2actionseq_idxs,
                                                                 instanceidx2actionseq_scores,
                                                                 instanceidx2actionseq_sideargs)
        batch_actionseq_probs = self._convert_actionscores_to_probs(batch_actionseq_scores)

        beta = 0.5
        epoch = float('inf') if epoch is None else epoch
        beta = math.pow(beta, 1.0/epoch)
        batch_actionseq_probs = self._meritocratic_program_prob(batch_actionseq_probs, beta)

        ''' THE PROGRAMS ARE EXECUTED HERE '''
        ''' First set the instance-spcific data (question, contexts, entity_mentions, etc.) to their 
            respective worlds (executors) so that execution can be performed.
        '''
        # This is a list of attention/action tuples for bool_qent_qstr supervision being used for diagnostics
        for i in range(0, len(languages)):
            languages[i].set_execution_parameters(execution_parameters=self.executor_parameters)
            languages[i].set_arguments(ques_embedded=embedded_questions[i],
                                       ques_encoded=encoded_questions[i],
                                       ques_mask=questions_mask[i],
                                       contexts_embedded=embedded_contexts[i],
                                       contexts_encoded=encoded_contexts[i],
                                       contexts_modeled=modeled_contexts[i],
                                       contexts_vec=None,   # context_vec_list[i],
                                       contexts_mask=contexts_mask[i],
                                       ne_ent_mens=ent_mens[i],
                                       num_ent_mens=num_mens[i],
                                       date_ent_mens=date_mens[i],
                                       q_nemenspan2entidx=q_nemenspan2entidx[i],
                                       bool_qstr_qent_func=self._bool_qstrqent_func)
                                       # snli_ques=embedded_snliques[i],
                                       # snli_contexts=embedded_snlicontexts[i],
                                       # snliques_mask=snliques_mask[i],
                                       # snlicontexts_mask=snlicontexts_mask[i])
            languages[i].preprocess_arguments()

        # Find the gold attention/action tuples for bool_qent_qstr supervision being used for diagnostics, if needed
        batch_gold_actions = []
        if self._goldactions or self._aux_goldprog_loss:
            for i in range(0, len(languages)):
                qent1, qent2, qstr = languages[i]._get_gold_actions()
                batch_gold_actions.append((qent1, qent2, qstr))

        if self._goldactions:
            if self._wsideargs:
                self.add_goldatt_to_sideargs(batch_actionseqs, batch_actionseq_sideargs, batch_gold_actions)
            else:
                self.replace_qentqstr_actions_w_gold(batch_actionseqs, batch_gold_actions)

        # List[List[denotation]], List[List[str]]: Denotations and their types for all instances
        batch_denotations, batch_denotation_types = self._get_denotations(batch_actionseqs, languages,
                                                                          batch_actionseq_sideargs)

        (best_predicted_anstypes,
         best_predicted_denotations,
         best_predicted_actionseq_probs,
         predicted_expected_denotations,
         topk_predicted_denotations) = self._expected_best_denotations(batch_denotation_types=batch_denotation_types,
                                                                       batch_action_probs=batch_actionseq_probs,
                                                                       batch_denotations=batch_denotations,
                                                                       gold_ans_type=gold_ans_type)

        if ans_grounding_dict is not None:
            loss = 0.0
            denotation_loss = self._compute_loss(gold_ans_type=gold_ans_type,
                                                 ans_grounding_dict=ans_grounding_dict,
                                                 ans_grounding_mask=ans_grounding_mask,
                                                 predicted_expected_denotations=predicted_expected_denotations,
                                                 predicted_best_denotations=best_predicted_denotations,
                                                 topk_predicted_denotations=topk_predicted_denotations)
            loss += denotation_loss
            outputs["denotation_loss"] = denotation_loss

            # Auxiliary loss encouraging that find_Qent action attends to one of the q_ent spans
            # Since the attention is normalized, this leads to no-attention on non-entity tokens
            if self._entityspan_qatt_loss:
                entityspan_ques_att_loss = self._entity_attention_loss(languages=languages,
                                                                       batch_actionseqs=batch_actionseqs,
                                                                       batch_side_args=batch_actionseq_sideargs)

                # entityspan_ques_att_loss = 0.1 * entityspan_ques_att_loss
                loss += entityspan_ques_att_loss
                outputs["qent_loss"] = entityspan_ques_att_loss
                self.qent_loss(entityspan_ques_att_loss.detach().cpu().numpy().tolist())

            # Auxiliary loss against gold-actions (attentions for w/wideargs, or MML loss with gold parse for w/oside)
            if self._aux_goldprog_loss is True:
                assert self._goldactions is False, "GoldActions cannot be true with auxiliary loss"
                if self._wsideargs:
                    ques_att_loss = self._ques_attention_loss(languages=languages,
                                                              batch_actionseqs=batch_actionseqs,
                                                              batch_side_args=batch_actionseq_sideargs,
                                                              batch_gold_ques_attns=batch_gold_actions)
                    loss += ques_att_loss
                    outputs["aux_goldparse_loss"] = ques_att_loss
                    self.aux_goldparse_loss(ques_att_loss.detach().cpu().numpy().tolist())

                if self._wsideargs is False:
                    (batch_goldactionseq_tensor,
                     batch_goldactionseq_mask) = self._get_gold_actionseq_forMML(batch_gold_qentqstr=batch_gold_actions,
                                                                                 actions=actions,
                                                                                 languages=languages,
                                                                                 device_id=device_id)

                    mml_loss = self._mml.decode(initial_state=initial_state,
                                                transition_function=self._decoder_step,
                                                supervision=(batch_goldactionseq_tensor, batch_goldactionseq_mask))['loss']
                    loss += mml_loss
                    outputs["aux_goldparse_loss"] = mml_loss
                    self.aux_goldparse_loss(mml_loss.detach().cpu().numpy().tolist())

            # Auxiliary loss encouraging aggregate attention on all tokens for find_* actions
            if self._qattn_coverage_loss is True:
                attn_coverage_loss = self._attention_coverage_loss(ques_masks=questions_mask,
                                                                   batch_actionseqs=batch_actionseqs,
                                                                   batch_side_args=batch_actionseq_sideargs)
                # attn_coverage_loss *= 0.01
                loss += attn_coverage_loss
                self.qattn_cov_loss_metric(myutils.tocpuNPList(attn_coverage_loss))

            outputs["loss"] = loss

        if metadata is not None:
            outputs["metadata"] = metadata
        outputs["best_action_strings"] = batch_actionseqs
        outputs["batch_actionseq_probs"] = batch_actionseq_probs
        outputs["batch_actionseq_sideargs"] = batch_actionseq_sideargs
        outputs["denotations"] = batch_denotations
        outputs["best_denotations"] = best_predicted_denotations
        outputs["languages"] = languages

        return outputs

    def _average_ques_repr(self, ques_reprs, ques_masks):
        """ Average question reprs from multiple runs of bidaf

        Parameters:
        -----------
        ques_reprs: `List[torch.FloatTensor]`
            A list of (B, T, D) sized tensors from multiple runs of bidaf for different contexts
        """
        avg_ques_repr = sum(ques_reprs)/float(len(ques_reprs))
        # Since mask for each run should be the same
        avg_ques_mask = ques_masks[0]

        return (avg_ques_repr, avg_ques_mask)

    def _concatenate_context_reprs(self, batch_size, *args):
        def unsqueeze_dim_1(*args):
            return [[t.unsqueeze(1) for t in a] for a in args]
        def concat_dim_1(*args):
            return [torch.cat(t, dim=1) for t in args]
        def slice_tensor_to_list(batch_size, *args):
            return [[t[i] for i in range(batch_size)] for t in args]

        unsqueezed_tensors = unsqueeze_dim_1(*args)
        concatenated_tensors = concat_dim_1(*unsqueezed_tensors)
        sliced_tensors = slice_tensor_to_list(batch_size, *concatenated_tensors)

        return sliced_tensors


    def _get_mens_mask(self, mention_spans: torch.LongTensor) -> torch.LongTensor:
        """ Get mask for spans from a SpanField.

        Parameters:
        -----------
        mention_spans : ``torch.LongTensor``
            Spanfield tensor of arbitrary size but with last dim of size=2.

        Returns:
        --------
        mask: ``torch.LongTensor``
            A tensor of size `dim - 1` with 1 for valid spans, and 0 for padded spans
        """
        # Ellipsis
        span_mask = (mention_spans[..., 0] >= 0).long()
        return span_mask


    def _get_firstStepActionIdxs_GoldAnsDict(self,
                                             gold_ans_type: List[str],
                                             actions: List[List[ProductionRule]],
                                             languages: List[HotpotQALanguage],
                                             **kwargs):
        """ If gold answer types are given, then make a dict of answers and equivalent masks based on type.
        Also give a list of possible first actions when decode to make programs of valid types only.

        :param gold_ans_types:
        :return:
        """

        if gold_ans_type is None:
            return None, None, None

        # Field_name containing the grounding for type T is "ans_grounding_T"
        ans_grounding_prefix = "ans_grounding_"

        ans_grounding_dict = {}
        firststep_action_ids = []

        for k, v in kwargs.items():
            if k.startswith(ans_grounding_prefix):
                anstype = k[len(ans_grounding_prefix):]
                ans_grounding_dict[anstype] = v

        for instance_actions, ans_type, language in zip(actions, gold_ans_type, languages):
            # Converting the gold ans_type to the name used in our language
            langtype_name = language.typename_to_langtypename(ans_type)
            instance_allowed_actions = []
            for action_idx, action in enumerate(instance_actions):
                if action[0] == f"{alcommon_util.START_SYMBOL} -> {langtype_name}":
                    instance_allowed_actions.append(action_idx)
            firststep_action_ids.append(set(instance_allowed_actions))

        ans_grounding_mask = {}
        # Create masks for different types
        for ans_type, ans_grounding in ans_grounding_dict.items():
            if ans_type in types.ANS_TYPES:
                mask = (ans_grounding >= 0.0).float()
                ans_grounding_mask[ans_type] = mask

        return firststep_action_ids, ans_grounding_dict, ans_grounding_mask

    def _questionspan_action_embeddings(self, encoded_question_tensor, quesspanactions2spanfield):
        """ Get input_action_embeddings for question_span actions

        Encoded question is an encoding of the Question.

        Parameters:
        ------------
        encoded_question_tensor: ``torch.FloatTensor``  (B, ques_length, encoded_dim)
            Encoded question whose final state is also used for InitialRNNStatelet
        quesspanactions2spanfield: ``torch.LongTensor`` (B, A, 2)
            Spans of linked rule question spans to SpanField
        """
        # (B, A) -- A is the number of ques_span actions (includes QSTR and QENT actions)
        span_mask = (quesspanactions2spanfield[:, :, 0] >= 0).squeeze(-1).long()
        # [B, A, 2 * encoded_dim]
        quesspan_actions_encoded = self._quesspan_extractor(sequence_tensor=encoded_question_tensor,
                                                            span_indices=quesspanactions2spanfield,
                                                            span_indices_mask=span_mask)
        # Shape: (B, A, Action_emb/2)
        quesspan_action_embeddings = self._quesspan2actionemb(quesspan_actions_encoded)

        return quesspan_action_embeddings


    def _expected_best_denotations(self,
                                   batch_denotation_types: List[List[str]],
                                   batch_action_probs: List[torch.FloatTensor],
                                   batch_denotations: List[List[Any]],
                                   gold_ans_type: List[str]=None,
                                   top_k: int=5):
        """ Returns the best denotation and it's type. Also returns expected denotation if gold_ans_type is known.
        This is based on the assumption that all denotations are of the gold_type in this case.

        Parameters:
        ------------
        batch_denotation_types: ``List[List[str]]``
            For each instance, for each program it's return type
        batch_action_probs: ``torch.FloatTensor``
            For each instance, a tensor of shape (A) containing the probability for each predicted program
        batch_denotations: List[List[Any]]
            For each instance, the denotation (ideally, tensor) for each predicted program
        gold_ans_type: ``List[str]``, optional
            Gold answer type for each instance. If given, the function assumes (and checks) that all programs for an
            instance result in the gold type.
        ans_grounding_dict: ``Dict[str, Tensor]``, optional
            Dictionary from ans_type to a Tensor of shape (B, *) denoting the gold denotation for each instance.
            Since all instances don't belong to the same type, these tensors are padded and masked
        ans_grounding_mask: ``Dict[str, Tensor]``, optional
            Dictionary from ans_type to a Tensor of shape (B, *) denoting the mask for each instance if
            it doesn't belong to the key's answer type
        top_k: ``int`` * only if gold_ans_type* is given
            Along with top-1 and expected, also return top_k denotations for analysis purposes

        Returns:
        --------
        best_predicted_anstypes: List[str]
            Ans-Type from the best predicted program
        best_predicted_denotations: List[Any]
            Denotation from the best predicted program
        best_predicted_actionprobs: List[torch.FloatTensor]
            Probability of the best predicted program
        predicted_expected_denotations: ``List[Any]`` - * Only if gold_ans_type is provided *
            Expected denotation of the programs decoded by the beam-search
        topk_predicted_denotations: ``List[List[Any]]`` - * Only if gold_ans_type is provided *
            Top K predicted denotations from the beam search
        """

        all_predanstype_match = False

        # Since the batch_action_probs is sorted, the first instance is the best predicted program
        # These lists are of length = batch_size
        best_predicted_anstypes = [x[0] for x in batch_denotation_types]
        best_predicted_denotations = [x[0] for x in batch_denotations]
        best_predicted_actionprobs = [x[0] for x in batch_action_probs]

        predicted_expected_denotations = None
        topk_predicted_denotations = None

        if gold_ans_type is not None:
            predicted_expected_denotations = []
            # assert(ans_grounding_dict is not None), "Ans grounding dict is None"
            # assert (ans_grounding_mask is not None), "Ans grounding mask is None"
            type_check = [all([ptype == gtype for ptype in ins_types]) for gtype, ins_types in
                          zip(gold_ans_type, batch_denotation_types)]
            assert all(
                type_check), f"Program types mismatch gold type. \n GoldTypes:\n{gold_ans_type}" \
                             f"PredictedTypes:\n{batch_denotation_types}"

            topk_predicted_denotations = [x[0:top_k] for x in batch_denotations]

            for ins_idx in range(0, len(batch_denotations)):
                instance_denotations = batch_denotations[ins_idx]
                instance_action_probs = batch_action_probs[ins_idx]
                gold_type = gold_ans_type[ins_idx]

                num_actionseqs = len(instance_denotations)
                # Making a single tensor of (A, *d) by stacking all denotations
                # Size of all denotations for the same instance should be the same, hence no padding should be required.
                # Shape: [A, *d]
                ins_denotations = torch.cat([single_actionseq_d.unsqueeze(0)
                                             for single_actionseq_d in instance_denotations], dim=0)
                num_dim_in_denotation = len(instance_denotations[0].size())
                # [A, 1,1, ...], dim=1 onwards depends on the size of the denotation
                # This view allows broadcast when computing the expected denotation
                instance_action_probs_ex = instance_action_probs.view(num_actionseqs, *([1]*num_dim_in_denotation))

                # Shape: [*d]
                expected_denotation = (ins_denotations * instance_action_probs_ex).sum(0)
                predicted_expected_denotations.append(expected_denotation)
        return (best_predicted_anstypes, best_predicted_denotations, best_predicted_actionprobs,
                predicted_expected_denotations, topk_predicted_denotations)


    def _compute_loss(self, gold_ans_type: List[str],
                            ans_grounding_dict: Dict,
                            ans_grounding_mask: Dict,
                            predicted_expected_denotations: List[torch.Tensor],
                            predicted_best_denotations: List[torch.Tensor],
                            topk_predicted_denotations: List[List[torch.Tensor]]=None):
        """ Compute the loss given the gold type, grounding, and predicted expected denotation of that type. """

        loss = 0.0

        # All answer denotations will comes as tensors
        # For each instance, compute expected denotation based on the prob of the action-seq.
        for ins_idx in range(0, len(predicted_expected_denotations)):
            gold_type = gold_ans_type[ins_idx]
            gold_denotation = ans_grounding_dict[gold_type][ins_idx]
            mask = ans_grounding_mask[gold_type][ins_idx]
            expected_denotation = predicted_expected_denotations[ins_idx]
            best_denotation = predicted_best_denotations[ins_idx]
            if topk_predicted_denotations is not None:
                topk_denotations = topk_predicted_denotations[ins_idx]
            else:
                topk_denotations = None

            gold_denotation = gold_denotation * mask
            expected_denotation = expected_denotation * mask
            best_denotation = best_denotation * mask
            if topk_denotations is not None:
                topk_denotations = [d * mask for d in topk_denotations]

            if gold_type == hpcons.BOOL_TYPE:
                instance_loss = F.binary_cross_entropy(input=expected_denotation, target=gold_denotation)
                loss += instance_loss

                # Top1 prediction
                bool_pred = (best_denotation >= 0.5).float()
                correct = 1.0 if (bool_pred == gold_denotation) else 0.0
                self.top1_acc_metric(float(correct))

                # ExptecDenotation Acc
                bool_pred = (expected_denotation >= 0.5).float()
                correct = 1.0 if (bool_pred == gold_denotation) else 0.0
                self.expden_acc_metric(float(correct))

                if topk_denotations is not None:
                    # List of torch.Tensor([0.0]) or torch.Tensor([1.0])
                    topk_preds = [(d >= 0.5).float() for d in topk_denotations]
                    pred_correctness = [1.0 if (x == gold_denotation) else 0.0 for x in topk_preds]
                    correct = max(pred_correctness)
                    self.topk_acc_metric(float(correct))
                else:
                    self.topk_acc_metric(0.0)

        return loss


    def _attention_coverage_loss(self,
                                 ques_masks: List[torch.FloatTensor],
                                 batch_actionseqs: List[List[List[str]]],
                                 batch_side_args: List[List[List[Dict]]]):
        """ Encourage the question_attentions in a program to attend to all tokens.
            Every program attends to the question for terminal-find_Qent, find_Qstr, etc. functions.
            We want that in aggregate all tokens are attended to.

            For each token, We maximize the sum of token-attention from different relevant-actions
        """
        relevant_actions = ['Qent -> find_Qent', 'Qstr -> find_Qstr']
        loss = 0.0
        normalizer = 0
        batch_size = len(batch_actionseqs)
        for i in range(0, batch_size):
            instance_action_seqs: List[List[str]] = batch_actionseqs[i]
            instance_sideargs: List[List[Dict]] = batch_side_args[i]
            ques_mask = ques_masks[i]
            for program, side_args in zip(instance_action_seqs, instance_sideargs):
                # List of (Qlen,) shaped attention vectors
                relevant_ques_attns = []
                for action, side_arg in zip(program, side_args):
                    if action in relevant_actions:
                        question_attention = side_arg['question_attention']
                        relevant_ques_attns.append(question_attention)

                # Concatenating ques_attns into (Qlen, R) - R is the num. of relevant actions
                relevant_ques_attns = torch.cat([x.unsqueeze(1) for x in relevant_ques_attns], dim=1)

                # Idea here is to minimize (1 - max(token_attn)) for every token
                '''
                # Making a single (Qlen,) tensor containing the token-wise max across all relevant attns
                # Shape: (Qlen,)
                max_ques_attn, _ = torch.max(relevant_ques_attns, dim=1)
                # The idea is that all elements of this tensor should be low, meaning the token was attended to atleast
                # by one question attention
                negative_coverage = 1 - max_ques_attn
                log_negative_coverage = torch.log(negative_coverage + 1e-45) * ques_mask
                program_coverage_loss = torch.sum(log_negative_coverage)
                '''
                # Idea here is to maximize \prod_tokens (\sum att_i) - i.e. maximize the sum of attns for each token
                # Shape = (Qlen,)
                tokenwise_sum_attns = torch.sum(relevant_ques_attns, dim=1)
                log_sum_attns = torch.log(tokenwise_sum_attns + 1e-45) * ques_mask
                program_coverage_loss = -1 * torch.sum(log_sum_attns)
                normalizer += 1
                loss += program_coverage_loss

        return loss/normalizer


    def _entity_attention_loss(self,
                               languages: List[HotpotQALanguageWSideArgs],
                               batch_actionseqs: List[List[List[str]]],
                               batch_side_args: List[List[List[Dict]]]):
        ''' Auxiliary loss to encourage the model to attend to entity-spans when predicting 'Qent -> findQent' action
            Compute a maximizing-objective, the sum of entity_probs (across q-ents) based on the attention.
            The entity_prob can be computed in two ways (controlled by the variable sum
                1. Sum of entity_token_prob - The model can choose to divide the attention assymetrically across tokens
                2. Prob of entity_token_prob - Encourages the model to divide the attention uniformly across tokens
        '''

        relevant_action = 'Qent -> find_Qent'
        batch_size = len(languages)
        loss = 0.0
        normalizer = 0

        sum_entitytokenprob_bool = True
        max_entitydist_entropy = True

        sigmoid_att = False if (self._transitionfunc_att._normalize is True) else True

        for i in range(0, batch_size):
            # Shape: (Qlen)
            entidx2spanvecs = languages[i].entidx2spanvecs
            entity_span_vec_gold = languages[i].entitymention_idxs_vec
            instance_action_seqs: List[List[str]] = batch_actionseqs[i]
            instance_sideargs: List[List[Dict]] = batch_side_args[i]
            all_ent_vecs = [vec for _, vecs in entidx2spanvecs.items() for vec in vecs]

            # For each attention action, one of many entity_spans is correct. Hence we try a MML kind-of loss.
            # Maximize the log of prod instance-likelihood (reward), i.e. sum of log expected-reward(LH) each instance.
            # The expected reward in our case is the reward for each gold entity_span where,
            # the reward for a single span is the prod-of-token-attn-prob for entity-tokens in that span

            for program, side_args in zip(instance_action_seqs, instance_sideargs):
                for action, side_arg in zip(program, side_args):
                    if action == relevant_action:
                        question_attention = side_arg['question_attention']
                        log_question_attention = torch.log(question_attention + 1e-40)
                        sum_sum_attn_probs = 0
                        sum_prod_attn_probs = 0
                        entropy_ent_attns = 0
                        # sum_log_prod_attn_probs = 0
                        all_entity_vec = 0
                        for vec in all_ent_vecs:
                            all_entity_vec = all_entity_vec + vec
                            if not sum_entitytokenprob_bool:
                                # prod_attn_probs = exp(\sum log(attn_probs))
                                prod_attn_probs = torch.exp(torch.sum(vec * log_question_attention))
                                # Sum of prod_attn_probs for different entities
                                sum_prod_attn_probs = sum_prod_attn_probs + prod_attn_probs
                            else:
                                # sum_attn_probs = \sum attn_probs
                                sum_attn_probs = torch.sum(vec * question_attention)
                                # Sum of sum_attn_probs for different entities
                                sum_sum_attn_probs = sum_sum_attn_probs + sum_attn_probs

                            if max_entitydist_entropy:
                                # Computing entity dist entropy
                                entity_probs = vec * question_attention
                                sum_entity_probs = torch.sum(entity_probs) + 1e-20
                                entity_prob_dist = (entity_probs/sum_entity_probs) * vec
                                log_entity_prob_dist = torch.log(entity_prob_dist + 1e-40) * vec
                                ent_entropy = -1 * torch.sum(vec * entity_prob_dist * log_entity_prob_dist)
                                entropy_ent_attns += ent_entropy

                        if sigmoid_att is True:
                            # This is additionally used for sigmoid attention
                            non_entity_tokens = (all_entity_vec <= 0.0).float()
                            log_one_minus_p = torch.log((non_entity_tokens - question_attention) \
                                                        *non_entity_tokens  + 1e-5) * non_entity_tokens
                            sum_log_one_minus_p = torch.sum(log_one_minus_p)
                        else:
                            sum_log_one_minus_p = 0.0

                        if sum_entitytokenprob_bool:
                            loss += torch.log(sum_sum_attn_probs)
                        else:
                            loss += torch.log(sum_prod_attn_probs)

                        # if max_entitydist_entropy:
                        loss += entropy_ent_attns

                        loss += sum_log_one_minus_p
                        normalizer += 1

        return -1 * (loss/normalizer)


    def _ques_attention_loss(self,
                             languages: List[HotpotQALanguageWSideArgs],
                             batch_actionseqs: List[List[List[str]]],
                             batch_side_args: List[List[List[Dict]]],
                             batch_gold_ques_attns: List[Tuple[torch.FloatTensor,
                                                               torch.FloatTensor,
                                                               torch.FloatTensor]]):
        """ W/SideArgs: This can be used as an auxiliary loss for qent1, qent2, qstr ques_atten for boolwosame ques"""
        # batch_gold_ques_attns contains gold (qent1, qent2, qstr) attention vectors for each instance
        qent_action = 'Qent -> find_Qent'
        qstr_action = 'Qstr -> find_Qstr'
        batch_size = len(languages)
        loss = 0.0
        normalizer = 0
        for i in range(0, batch_size):
            # Shape: (Qlen)
            instance_action_seqs: List[List[str]] = batch_actionseqs[i]
            instance_sideargs: List[List[Dict]] = batch_side_args[i]
            qent1, qent2, qstr = batch_gold_ques_attns[i]
            first_qent = True

            # Maximize the log of prod of instance-likelihood, i.e. sum of log likelihood for each instance.
            # The likelihood for each instance is the prod-of-token-attn-prob for for gold-tokens
            for program, side_args in zip(instance_action_seqs, instance_sideargs):
                for action, side_arg in zip(program, side_args):
                    question_attention = side_arg['question_attention']
                    log_question_attention = torch.log(question_attention + 1e-5)
                    if action == qent_action:
                        if first_qent:
                            l = torch.sum(log_question_attention * qent1)
                            loss += l
                            normalizer += 1
                            first_qent = False
                        else:
                            l = torch.sum(log_question_attention * qent2)
                            loss += l
                            normalizer += 1
                    if action == qstr_action:
                        l = torch.sum(log_question_attention * qstr)
                        loss += l
                        normalizer += 1
        return -1 * (loss/normalizer)


    def add_goldatt_to_sideargs(self,
                                batch_actionseqs: List[List[List[str]]],
                                batch_actionseq_sideargs: List[List[List[Dict]]],
                                batch_gold_attentions: List[Tuple]):
        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            instance_gold_attentions = batch_gold_attentions[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                first_entity = True   # This tells model which qent attention to use
                # print(side_args)
                # print()
                for action, sidearg_dict in zip(program, side_args):
                    if action == 'Qent -> find_Qent':
                        if first_entity:
                            sidearg_dict['question_attention'] = instance_gold_attentions[0]
                            first_entity = False
                        else:
                            sidearg_dict['question_attention'] = instance_gold_attentions[1]
                    elif action == 'Qstr -> find_Qstr':
                        sidearg_dict['question_attention'] = instance_gold_attentions[2]

    def replace_qentqstr_actions_w_gold(self,
                                        batch_actionseqs: List[List[List[str]]],
                                        batch_gold_actions: List[Tuple]):
        # For each instance, batch_gold_actions contains a tuple of (qent1, qent2, qstr) action.
        # From each instances' program, replace the first 'Qent -> xxx' with qent1, second with qent2 and 'Qstr -> ..'
        for ins_idx in range(len(batch_actionseqs)):
            instance_programs: List[List[str]] = batch_actionseqs[ins_idx]
            instance_gold_actions: Tuple[str, str, str] = batch_gold_actions[ins_idx]

            for program in instance_programs:
                first_entity = True   # This tells model which qent attention to use
                for idx, action in enumerate(program):
                    if action.startswith('Qent ->'):
                        if first_entity:
                            program[idx] = instance_gold_actions[0]
                            first_entity = False
                        else:
                            program[idx] = instance_gold_actions[1]
                    elif action.startswith('Qstr ->'):
                        program[idx] = instance_gold_actions[2]


    def _get_gold_actionseq_forMML(self,
                                   batch_gold_qentqstr: List[Tuple[str, str, str]],
                                   actions: List[List[ProductionRule]],
                                   languages: List[HotpotQALanguageWOSideArgs],
                                   device_id: int):
        """ For BoolWOSame questions, make gold_actionseq_idxs_tensor for MML loss. """
        action_dicts = [{} for _ in range(len(actions))]
        for idx, ins_actions in enumerate(actions):
            for action_index, action in enumerate(ins_actions):
                action_string = action[0]
                if action_string:
                    # print("{} {}".format(action_string, action))
                    action_dicts[idx][action_string] = action_index

        # The middle list here will be of size 1 since there is one gold_action_seq per instance.
        batch_actionseq_idxs: List[List[List[int]]] = []
        for idx, gold_qent_qstr in enumerate(batch_gold_qentqstr):
            gold_logicalform = getGoldLF(gold_qent_qstr[0], gold_qent_qstr[1], gold_qent_qstr[2])
            gold_action_seq: List[str] = languages[idx].logical_form_to_action_sequence(gold_logicalform)
            gold_actionseq_idxs = [action_dicts[idx][a] for a in gold_action_seq]
            batch_actionseq_idxs.append([gold_actionseq_idxs])

        if device_id > -1:
            batch_goldactions_tensor = torch.cuda.LongTensor(batch_actionseq_idxs)
            batch_goldactions_mask = torch.cuda.LongTensor(*batch_goldactions_tensor.size(), device=device_id).fill_(1)

            return (batch_goldactions_tensor, batch_goldactions_mask)
        else:
            raise NotImplementedError


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self.top1_acc_metric.get_metric(reset),
                'expden_acc': self.expden_acc_metric.get_metric(reset),
                'topk_acc': self.topk_acc_metric.get_metric(reset),
                'aux_gp_l': self.aux_goldparse_loss.get_metric(reset),
                'qent_l': self.qent_loss.get_metric(reset),
                'attcov_l': self.qattn_cov_loss_metric.get_metric(reset)
        }
