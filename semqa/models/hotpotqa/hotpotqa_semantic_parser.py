import logging
from typing import List, Dict, Any, Tuple

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
from allennlp.modules.span_extractors import SpanExtractor
import allennlp.nn.util as allenutil
import allennlp.common.util as alcommon_util
from allennlp.models.archival import load_archive
from allennlp.models.decomposable_attention import DecomposableAttention

import semqa.type_declarations.semqa_type_declaration_wques as types
from semqa.domain_languages.hotpotqa.hotpotqa_language_w_sideargs import HotpotQALanguageWSideArgs
from semqa.models.hotpotqa.hotpotqa_parser_base import HotpotQAParserBase
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters

from semqa.data.datatypes import DateField, NumberField
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp.training.metrics import Average
from semqa.models.utils.bidaf_utils import PretrainedBidafModelUtils
from semqa.models.utils import generic_utils as genutils

import datasets.hotpotqa.utils.constants as hpcons

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
                 wsideargs: bool = True,
                 goldactions: bool = False,
                 bidafutils: PretrainedBidafModelUtils = None,
                 question_token_repr_key: str = 'embed',
                 context_token_repr_key: str = 'embed',
                 bool_qstrqent_func: str = 'mentions',
                 entityspan_qatt_loss: bool = False,
                 dropout: float = 0.0,
                 text_field_embedder: TextFieldEmbedder = None,
                 qencoder: Seq2SeqEncoder = None,
                 quesspan_extractor: SpanExtractor = None,
                 quesspan2actionemb: FeedForward = None,
                 use_quesspan_actionemb: bool = False,
                 debug: bool=False) -> None:

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

        if self._qencoder is not None:
            encoder_output_dim = self._qencoder.get_output_dim()
        elif bidafutils is not None:
            encoder_output_dim = bidafutils._bidaf_encoded_dim
        else:
            raise NotImplementedError

        self._decoder_step = LinkingTransitionFunction(encoder_output_dim=encoder_output_dim,
                                                       action_embedding_dim=action_embedding_dim,
                                                       input_attention=attention,
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
                languages: List[HotpotQALanguageWSideArgs],
                actions: List[List[ProductionRule]],
                linked_rule2idx: List[Dict],
                quesspanaction2linkingscore: torch.FloatTensor,
                quesspanactions2spanfield: torch.LongTensor,
                gold_ans_type: List[str]=None,
                snli_ques=None,
                snli_contexts=None,
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
        initial_rnn_state = self._get_initial_rnn_state(ques_repr=encoded_ques_tensor,
                                                        ques_mask=questions_mask_tensor,
                                                        question_final_repr=ques_encoded_final_state,
                                                        ques_encoded_list=encoded_questions,
                                                        ques_mask_list=questions_mask)

        # Initial side_args list to make the GrammarBasedState if using Language with sideargs
        initial_side_args = None
        if self._wsideargs:
            initial_side_args = [[] for i in range(0, batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
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

        instanceidx2actionseq_idxs: Dict[int, List[List[int]]] = {}
        instanceidx2actionseq_scores: Dict[int, List[torch.Tensor]] = {}
        instanceidx2actionseq_sideargs: Dict[int, List[List[Dict]]] = {} if self._wsideargs else None

        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            if i in best_final_states:
                # Since the group size for any state is 1, action_history[0] can be used.
                instance_actionseq_idxs = [final_state.action_history[0] for final_state in best_final_states[i]]
                instance_actionseq_scores = [final_state.score[0] for final_state in best_final_states[i]]
                instanceidx2actionseq_idxs[i] = instance_actionseq_idxs
                instanceidx2actionseq_scores[i] = instance_actionseq_scores
                if self._wsideargs:
                    instance_actionseq_sideargs = [final_state.debug_info[0] for final_state in best_final_states[i]]
                    instanceidx2actionseq_sideargs[i] = instance_actionseq_sideargs

        # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
        # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
        # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (batch_actionseqs, batch_actionseq_scores,
         batch_actionseq_sideargs) = self._get_actionseq_strings(actions,
                                                                 instanceidx2actionseq_idxs,
                                                                 instanceidx2actionseq_scores,
                                                                 instanceidx2actionseq_sideargs)
        # Convert batch_action_scores to a single tensor the size of number of actions for each batch
        device_id = allenutil.get_device_of(batch_actionseq_scores[0][0])
        # List[torch.Tensor] : Stores probs for each action_seq. Tensor length is same as the number of actions
        # The prob is normalized across the action_seqs in the beam
        batch_actionseq_probs = []
        for score_list in batch_actionseq_scores:
            scores_astensor = allenutil.move_to_device(torch.cat([x.view(1) for x in score_list]), device_id)
            action_probs = allenutil.masked_softmax(scores_astensor, mask=None)
            batch_actionseq_probs.append(action_probs)

        # for ins in batch_actionseqs:
        #     print(f"{ins}\n")

        ''' THE PROGRAMS ARE EXECUTED HERE '''
        ''' First set the instance-spcific data (question, contexts, entity_mentions, etc.) to their 
            respective worlds (executors) so that execution can be performed.
        '''
        # This is a list of attention/action tuples for bool_qent_qstr supervision being used for diagnostics
        batch_gold_actions = []
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
            if self._goldactions:
                # Works for both with and w/o sideargs
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

            if self._entityspan_qatt_loss:
                entityspan_ques_att_loss = self._entity_attention_loss(languages=languages,
                                                                       batch_actionseqs=batch_actionseqs,
                                                                       batch_side_args=batch_actionseq_sideargs)
                loss += entityspan_ques_att_loss
                outputs["entityspan_ques_att_loss"] = entityspan_ques_att_loss

            outputs["loss"] = loss

        if metadata is not None:
            outputs["metadata"] = metadata
        outputs["best_action_strings"] = batch_actionseqs
        outputs["batch_actionseq_sideargs"] = batch_actionseq_sideargs
        outputs["denotations"] = batch_denotations
        outputs["best_denotations"] = best_predicted_denotations
        outputs["languages"] = languages

        return outputs


    def _entity_attention_loss(self,
                               languages: List[HotpotQALanguageWSideArgs],
                               batch_actionseqs: List[List[List[str]]],
                               batch_side_args: List[List[List[Dict]]]):
        relevant_action = 'Qent -> find_Qent'
        batch_size = len(languages)
        loss = 0.0

        for i in range(0, batch_size):
            # Shape: (Qlen)
            entity_span_vec_gold = languages[i].entitymention_idxs_vec
            instance_action_seqs: List[List[str]] = batch_actionseqs[i]
            instance_sideargs: List[List[Dict]] = batch_side_args[i]

            for program, side_args in zip(instance_action_seqs, instance_sideargs):
                for action, side_arg in zip(program, side_args):
                    if action == relevant_action:
                        question_attention = side_arg['question_attention']
                        loss += torch.sum(question_attention * entity_span_vec_gold)

        return -1 * loss

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

    '''
    def _concatenate_context_reprs(self, context_embeds, context_encodings,
                                   context_modelings, context_vecs, context_masks):
        def unsqueeze_dim_1(*args):
            return [[t.unsqueeze(1) for t in a] for a in args]
        def concat_dim_1(*args):
            return [torch.cat(t, dim=1) for t in args]
        def slice_tensor_to_list(batch_size, *args):
            return [[t[i] for i in range(batch_size)] for t in args]

        """ Concatenate context_repr from same instance into a single tensor, and return a list of these for the batch
        Input is list of different batched context_repr from different runs of the bidaf model. Here we extract
        different contexts for the same instance and concatenate them.

        Parameters:
        -----------
        context_embeds: `List[torch.FloatTensor]`
            A C-sized list of (B, T, D) sized tensors
        context_encodings: `List[torch.FloatTensor]`
            A C-sized list of (B, T, D) sized tensors
        context_modelings: `List[torch.FloatTensor]`
            A C-sized list of (B, T, D) sized tensors
        context_vecs: `List[torch.FloatTensor]`
            A C-sized list of (B, D) sized tensors
        context_masks: `List[torch.FloatTensor]`
            A C-sized list of (B, T) sized tensors
        """

        num_contexts = len(context_encodings)
        batch_size = context_encodings[0].size()[0]

        # We will unsqueeze dim-1 of all input tensors, concatenate it along that axis to make single tensor of
        # (B, C, T, D) and then slice instances out of it

        context_embeds_expanded = [t.unsqueeze(1) for t in context_embeds]
        context_encodings_expanded = [t.unsqueeze(1) for t in context_encodings]
        context_modelings_expanded = [t.unsqueeze(1) for t in context_modelings]
        context_vecs_expanded = [t.unsqueeze(1) for t in context_vecs]
        context_masks_expanded = [t.unsqueeze(1) for t in context_masks]

        context_embeds_concatenated = torch.cat(context_embeds_expanded, dim=1)
        context_encodings_concatenated = torch.cat(context_encodings_expanded, dim=1)
        context_modelings_concatenated = torch.cat(context_modelings_expanded, dim=1)
        context_vecs_concatenated = torch.cat(context_vecs_expanded, dim=1)
        context_masks_concatenated = torch.cat(context_masks_expanded, dim=1)

        context_embed_list = [context_embeds_concatenated[i] for i in range(batch_size)]
        context_encoding_list = [context_encodings_concatenated[i] for i in range(batch_size)]
        context_modeling_list = [context_modelings_concatenated[i] for i in range(batch_size)]
        context_vec_list = [context_vecs_concatenated[i] for i in range(batch_size)]
        context_mask_list = [context_masks_concatenated[i] for i in range(batch_size)]

        return (context_embed_list, context_encoding_list, context_modeling_list, context_vec_list, context_mask_list)
    '''


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
                                             languages: List[HotpotQALanguageWSideArgs],
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


    # def _update_metrics(self,
    #                     action_strings: List[List[List[str]]],
    #                     worlds: List[SampleHotpotWorld],
    #                     label_strings: List[List[str]]) -> None:
    #     # TODO(pradeep): Move this to the base class.
    #     # TODO(pradeep): Using only the best decoded sequence. Define metrics for top-k sequences?
    #     batch_size = len(worlds)
    #     for i in range(batch_size):
    #         instance_action_strings = action_strings[i]
    #         sequence_is_correct = [False]
    #         if instance_action_strings:
    #             instance_label_strings = label_strings[i]
    #             instance_worlds = worlds[i]
    #             # Taking only the best sequence.
    #             sequence_is_correct = self._check_denotation(instance_action_strings[0],
    #                                                          instance_label_strings,
    #                                                          instance_worlds)
    #         for correct_in_world in sequence_is_correct:
    #             self._denotation_accuracy(1 if correct_in_world else 0)
    #         self._consistency(1 if all(sequence_is_correct) else 0)


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self.top1_acc_metric.get_metric(reset),
                'expden_acc': self.expden_acc_metric.get_metric(reset),
                'topk_acc': self.topk_acc_metric.get_metric(reset)
        }
