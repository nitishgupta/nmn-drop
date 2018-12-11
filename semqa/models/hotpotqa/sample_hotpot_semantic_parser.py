import sys
import logging
from typing import List, Dict, Any

from overrides import overrides

import torch
import torch.nn.functional as nnfunc

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
# from allennlp.models.semantic_parsing.nlvr.nlvr_semantic_parser import NlvrSemanticParser
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
import allennlp.semparse.type_declarations.type_declaration as type_declr
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions import BasicTransitionFunction, LinkingTransitionFunction
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
import allennlp.nn.util as allenutil

import semqa.type_declarations.semqa_type_declaration_wques as types
from semqa.worlds.hotpotqa.sample_world import SampleHotpotWorld
from semqa.models.hotpotqa.hotpot_semantic_parser import HotpotSemanticParser
import datasets.hotpotqa.utils.constants as hpcons

from semqa.data.datatypes import DateField, NumberField
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("sample_hotpot_parser")
class SampleHotpotSemanticParser(HotpotSemanticParser):
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
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 qencoder: Seq2SeqEncoder,
                 ques2action_encoder: Seq2SeqEncoder,
                 quesspan_extractor: SpanExtractor,
                 attention: Attention,
                 decoder_beam_search: ConstrainedBeamSearch,
                 context_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 beam_size: int,
                 max_decoding_steps: int,
                 dropout: float = 0.0) -> None:
        super(SampleHotpotSemanticParser, self).__init__(vocab=vocab,
                                                         question_embedder=question_embedder,
                                                         action_embedding_dim=action_embedding_dim,
                                                         qencoder=qencoder,
                                                         ques2action_encoder=ques2action_encoder,
                                                         quesspan_extractor=quesspan_extractor,
                                                         context_embedder=context_embedder,
                                                         context_encoder=context_encoder,
                                                         dropout=dropout)

        # self._decoder_trainer = ExpectedRiskMinimization(beam_size=beam_size,
        #                                                  normalize_by_length=True,
        #                                                  max_decoding_steps=max_decoding_steps)

        # self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
        #                                              action_embedding_dim=action_embedding_dim,
        #                                              input_attention=attention,
        #                                              num_start_types=1,
        #                                              activation=Activation.by_name('tanh')(),
        #                                              predict_start_type_separately=False,
        #                                              add_action_bias=False,
        #                                              dropout=dropout)
        self._decoder_step = LinkingTransitionFunction(encoder_output_dim=self._qencoder.get_output_dim(),
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

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                contexts,  #: Dict[str, torch.LongTensor],
                ent_mens: torch.LongTensor,
                num_nents: torch.LongTensor,
                num_mens: torch.LongTensor,
                num_numents: torch.LongTensor,
                date_mens: torch.LongTensor,
                num_dateents: torch.LongTensor,
                num_normval: List[List[NumberField]],
                date_normval: List[List[DateField]],
                worlds: List[SampleHotpotWorld],
                actions: List[List[ProductionRule]],
                linked_rule2idx: List[Dict],
                action2ques_linkingscore: torch.FloatTensor,
                ques_str_action_spans: torch.LongTensor,
                gold_ans_type: List[str]=None,
                **kwargs) -> Dict[str, torch.Tensor]:

        """ Forward call to the parser.

        Parameters:
        -----------
        kwargs: ``Dict``
            Is a dictionary containing datatypes for answer_groundings of different types, key being ans_grounding_TYPE
            Each key has a batched_tensor of ground_truth for the instances.
            Since each instance only has a single correct type, the values for the incorrect types are empty vectors ..
            to make batching easier. See the reader code for more details.
        """
        # pylint: disable=arguments-differ

        batch_size = len(worlds)

        device_id = allenutil.get_device_of(ent_mens)

        # Gold truth dict of {type: grounding}.
        # The reader passes datatypes with name "ans_grounding_TYPE" where prefix needs to be cleaned.
        ans_grounding_dict = None

        # List[Set[int]] --- Ids of the actions allowable at the first time step if the ans-type supervision is given.
        firststep_action_ids = None

        (firststep_action_ids,
         ans_grounding_dict,
         ans_grounding_mask) = self._get_FirstSteps_GoldAnsDict_and_Masks(gold_ans_type, actions, **kwargs)

        # Initial log-score list for the decoding, List of zeros.
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for i in range(batch_size)]

        # All Question_Str_Action representations
        # TODO(nitish): Matt gave 2 ideas to try:
        # (1) Same embedding for all actions
        # (2) Compose a QSTR_action embedding with span_embedding
        # Shape: (B, A, A_d)
        quesstr_action_reprs = self._questionstr_action_embeddings(question=question,
                                                                   ques_str_action_spans=ques_str_action_spans)

        # For each instance, create a grammar statelet containing the valid_actions and their representations
        initial_grammar_statelets = []
        for i in range(batch_size):
            initial_grammar_statelets.append(self._create_grammar_statelet(worlds[i],
                                                                           actions[i],
                                                                           linked_rule2idx[i],
                                                                           action2ques_linkingscore[i],
                                                                           quesstr_action_reprs[i]))

        # Initial RNN state for the decoder
        initial_rnn_state = self._get_initial_rnn_state(question)

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_statelets,
                                          possible_actions=actions)

        outputs: Dict[str, torch.Tensor] = {}

        # TODO(nitish): Before execution, encode the contexts, and get repr of mentions for all entities of all types
        # TODO: Context is encoded, but still need mention spans, and mentions for all entities.
        # Shape: (B, C, T, D)
        contexts_encoded = self._encode_contexts(contexts)
        print(f"Encoded contexts size: {contexts_encoded.size()}")

        # ent_mens_field, num_mens_field, date_mens_field -- Size is (B, E, C, M, 2)
        # For each batch, for each entity of that type, for each context, the mentions in this context


        print(f"Ent mens field size: {ent_mens.size()}")
        print(f"Num mens field size: {num_mens.size()}")
        print(f"Date mens field size: {date_mens.size()}")

        print(f"{num_nents} {num_numents} {num_dateents}")

        ent_mens_mask = self._get_mens_mask(ent_mens)
        num_mens_mask = self._get_mens_mask(ent_mens)
        date_mens_mask = self._get_mens_mask(ent_mens)

        ''' Parsing the question to get action sequences'''
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             firststep_allowed_actions=firststep_action_ids,
                                                             keep_final_unfinished_states=False)



        best_action_sequences: Dict[int, List[List[int]]] = {}
        best_action_seqscores: Dict[int, List[torch.Tensor]] = {}
        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            if i in best_final_states:
                # TODO(nitish): best_final_states[i] is a list of final states, change this to reflect that.
                # Since the group size for any state is 1, action_history[0] can be used.
                best_action_indices = [final_state.action_history[0] for final_state in best_final_states[i]]
                best_action_scores = [final_state.score[0] for final_state in best_final_states[i]]
                best_action_sequences[i] = best_action_indices
                best_action_seqscores[i] = best_action_scores

        # List[List[List[str]]]: For each instance in batch, List of action sequences (ac_seq is List[str])
        # List[List[torch.Tensor]]: For each instance in batch, score for each action sequence
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (batch_action_strings, batch_action_scores) = self._get_action_strings(actions,
                                                                               best_action_sequences,
                                                                               best_action_seqscores)

        print(batch_action_strings)

        # Convert batch_action_scores to a single tensor the size of number of actions for each batch
        device_id = allenutil.get_device_of(batch_action_scores[0][0])
        # List[torch.Tensor] : Stores probs for each action_seq. Tensor length is same as the number of actions
        # The prob is normalized across the action_seqs in the beam
        batch_action_probs = []
        for score_list in batch_action_scores:
            scores_astensor = allenutil.move_to_device(torch.cat([x.view(1) for x in score_list]), device_id)
            action_probs = allenutil.masked_softmax(scores_astensor, mask=None)
            batch_action_probs.append(action_probs)

        # List[List[denotation]], List[List[str]]: For each instance, denotations by executing the action_seqs and its type
        batch_denotations, batch_denotation_types = self._get_denotations(batch_action_strings, worlds)

        # # Get probability for different denotation types by marginalizing the prob for action_seqs of same type
        # batch_type2prob: List[Dict] = []
        # for instance_denotation_types, instance_action_probs in zip(batch_denotation_types, batch_action_probs):
        #     type2indices = {}
        #     type2probs = {}
        #     for i, t in enumerate(instance_denotation_types):
        #         if t not in type2indices:
        #             type2indices[t] = []
        #         type2indices[t].append(i)
        #
        #     for t, action_indices in type2indices.items():
        #         indices = allenutil.move_to_device(torch.tensor(action_indices), device_id)
        #         type_prob = torch.sum(instance_action_probs.index_select(0, indices))
        #         type2probs[t] = type_prob
        #     batch_type2prob.append(type2probs)

        # print(batch_denotations)
        # print(batch_denotation_types)
        # print(batch_action_scores)
        # print(batch_action_probs)
        #
        # print(gold_ans_type)

        if ans_grounding_dict is not None:
            outputs["loss"] = self._compute_loss(gold_ans_type=gold_ans_type,
                                                 ans_grounding_dict=ans_grounding_dict,
                                                 ans_grounding_mask=ans_grounding_mask,
                                                 batch_denotations=batch_denotations,
                                                 batch_action_probs=batch_action_probs,
                                                 batch_denotation_types=batch_denotation_types)

        # if target_action_sequences is not None:
        #     self._update_metrics(action_strings=batch_action_strings,
        #                          worlds=worlds,
        #                          label_strings=label_strings)
        # else:

        outputs["best_action_strings"] = batch_action_strings
        outputs["denotations"] = batch_denotations
        return outputs


    def _get_mens_mask(self, mention_spans):
        """ Get mask for entity_mention spans.

        Parameters:
        -----------
        mention_spans: ``torch.LongTensor``
            Mention spans for each entity of shape (B, E, C, M, 2)

        Returns:
        --------
        mask: ``torch.LongTensor``
            Shape: (B, E, C, M)

        """
        # Shape: (B, E, C, M)
        span_mask = (mention_spans[:, :, :, :, 0] >= 0).squeeze(-1).long()
        return span_mask


    def _get_FirstSteps_GoldAnsDict_and_Masks(self,
                                              gold_ans_type: List[str],
                                              actions: List[List[ProductionRule]],
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

        for instance_actions, ans_type in zip(actions, gold_ans_type):
            # Answer types are BASIC_TYPES in our domain.
            # Converting the gold basic_type to nltk's naming convention
            ans_type_nltkname = ans_type.lower()[0]
            instance_allowed_actions = []
            for action_idx, action in enumerate(instance_actions):
                if action[0] == f"{type_declr.START_SYMBOL} -> {ans_type_nltkname}":
                    instance_allowed_actions.append(action_idx)
            firststep_action_ids.append(set(instance_allowed_actions))


        ans_grounding_mask = {}
        # Create masks for different types
        for ans_type, ans_grounding in ans_grounding_dict.items():
            if ans_type in types.ANS_TYPES:
                mask = (ans_grounding >= 0.0).float()
                ans_grounding_mask[ans_type] = mask

        return firststep_action_ids, ans_grounding_dict, ans_grounding_mask

    def _encode_contexts(self, contexts):
        """ Encode the contexts for each instance using the context_encoder RNN.

        Params:
        -------
        contexts: ``Dict[str, torch.LongTensor]``
            Since there are multiple contexts per instance, the contexts tensor are wrapped as (B, C, T)
            where C is the number of contexts

        Returns:
        --------
        contexts_encoded: ``torch.FloatTensor``
            Tensor of shape (B, C, T, D) after encoding all the contexts for each instance in the batch
        """

        # Shape: (B, C, T, W_d)
        embedded_contexts = self._context_embedder(contexts)
        embcontext_size = embedded_contexts.size()

        # Shape: (B, C, T)
        # Since multiple contexts per instance, give num_wrapping_dims
        contexts_mask = allenutil.get_text_field_mask(contexts, num_wrapping_dims=1).float()
        conmask_size = contexts_mask.size()

        (embedded_contexts_flat, contexts_mask_flat) = (
        embedded_contexts.view(-1, embcontext_size[2], embcontext_size[3]),
        contexts_mask.view(-1, conmask_size[2]))

        # Shape: (B*C, T, D)
        contexts_encoded_flat = self._context_encoder(embedded_contexts_flat, contexts_mask_flat)
        conenc_size = contexts_encoded_flat.size()
        # View such that get B, C, T from embedded context, and D from encoded contexts
        # Shape: (B, C, T, D)
        contexts_encoded = contexts_encoded_flat.view(*embcontext_size[0:3], conenc_size[-1])

        return contexts_encoded

    def _questionstr_action_embeddings(self, question, ques_str_action_spans):
        """ Get input_action_embeddings for question_str_span actions

        The idea is to run a RNN over the question to get a hidden-state-repr.
        Then for each question_span_action, get it's repr by extracting the end-point-reprs.

        Parameters:
        ------------
        question: Input to the forward from the question TextField
        action2span
        """

        embedded_input = self._question_embedder(question)
        # Shape: (B, Qlen)
        question_mask = allenutil.get_text_field_mask(question).float()
        # (B, Qlen, encoder_output_dim)
        quesaction_encoder_outputs = self._dropout(self._ques2action_encoder(embedded_input, question_mask))
        # (B, A) -- A is the number of ques_str actions
        span_mask = (ques_str_action_spans[:, :, 0] >= 0).squeeze(-1).long()
        # [B, A, action_dim]
        quesstr_action_reprs = self._quesspan_extractor(sequence_tensor=quesaction_encoder_outputs,
                                                        span_indices=ques_str_action_spans,
                                                        span_indices_mask=span_mask)
        return quesstr_action_reprs

    def _compute_loss(self, gold_ans_type: List[str],
                            ans_grounding_dict: Dict,
                            ans_grounding_mask: Dict,
                            batch_denotation_types: List[List[str]],
                            batch_action_probs: List[torch.FloatTensor],
                            batch_denotations: List[List[Any]]):

        # All action_sequences and denotations should be of the gold-type
        # (we only decode action-seqs that lead to the correct type)

        type_check = [all([ptype == gtype for ptype in ins_types]) for gtype, ins_types in zip(gold_ans_type, batch_denotation_types)]
        assert all(type_check), f"Program types mismatch gold type. \n GT:{gold_ans_type} BT: {batch_denotation_types}"
        # print(f"Type _check:{type_check}")

        loss = 0.0

        # All answer denotations will comes as tensors
        # For each instance, compute expected denotation based on the prob of the action-seq.
        for ins_idx, (instance_denotations,
                      instance_action_probs,
                      gold_type) in enumerate(zip(batch_denotations,
                                                  batch_action_probs,
                                                  gold_ans_type)):
            # print("Instance denotation and probs")
            # print(instance_denotations)
            # print(instance_action_probs)

            num_actionseqs = len(instance_denotations)
            # Size of all denotations for the same instance should be the same, hence no padding should be required.
            # [A, *d]
            ins_denotations = torch.cat([single_actionseq_d.unsqueeze(0) for single_actionseq_d in instance_denotations], dim=0)
            num_dim_in_denotation = len(instance_denotations[0].size())
            # [A, 1,1, ...], dim=1 onwards depends on the size of the denotation
            instance_action_probs_ex = instance_action_probs.view(num_actionseqs, *([1]*num_dim_in_denotation))

            # Shape: [*d]
            expected_denotation = (ins_denotations * instance_action_probs_ex).sum(0)

            gold_denotation = ans_grounding_dict[gold_type][ins_idx]
            mask = ans_grounding_mask[gold_type][ins_idx]

            expected_denotation = expected_denotation * mask

            if gold_type == hpcons.BOOL_TYPE:
                loss += torch.nn.functional.binary_cross_entropy(expected_denotation, gold_denotation)


        return loss


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

    '''
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'denotation_accuracy': self._denotation_accuracy.get_metric(reset),
                'consistency': self._consistency.get_metric(reset)
        }
    '''