import logging
from typing import Dict, List, Tuple, Any

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn import util
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
import allennlp.common.util as alcommon_utils

import datasets.hotpotqa.utils.constants as hpcons
from semqa.domain_languages.hotpotqa.hotpotqa_language import HotpotQALanguage
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters
import semqa.domain_languages.domain_language_utils as dl_utils

from allennlp.pretrained import PretrainedModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SPAN_DELIM = hpcons.SPAN_DELIM
START_SYMBOL = alcommon_utils.START_SYMBOL

class HotpotQAParserBase(Model):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain. This is an
    abstract class and does not have a ``forward`` method implemented. Classes that inherit from
    this class are expected to define their own logic depending on the kind of supervision they
    use.  Accordingly, they should use the appropriate ``DecoderTrainer``. This class provides some
    common functionality for things like defining an initial ``RnnStatelet``, embedding actions,
    evaluating the denotations of completed logical forms, etc.  There is a lot of overlap with
    ``WikiTablesSemanticParser`` here. We may want to eventually move the common functionality into
    a more general transition-based parsing class.

    Parameters
    ----------
    vocab : ``Vocabulary``
    sentence_embedder : ``TextFieldEmbedder``
        Embedder for sentences.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    dropout : ``float``, optional (default=0.0)
        Dropout on the encoder outputs.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 action_embedding_dim: int,
                 executor_parameters: ExecutorParameters,
                 wsideargs: bool,
                 text_field_embedder: TextFieldEmbedder = None,
                 qencoder: Seq2SeqEncoder = None,
                 ques2action_encoder: Seq2SeqEncoder = None,
                 quesspan_extractor: SpanExtractor = None,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super(HotpotQAParserBase, self).__init__(vocab=vocab)

        # using langauge with or without sideargs
        self._wsideargs = wsideargs

        self._denotation_accuracy = Average()
        self._consistency = Average()

        # Don't need these now that we're using bidaf for reprs.
        self._text_field_embedder = text_field_embedder
        self._qencoder = qencoder
        # self._ques2action_encoder = ques2action_encoder
        # self._quesspan_extractor = quesspan_extractor

        # self._context_embedder = context_embedder
        # self._context_encoder = context_encoder
        self.executor_parameters = executor_parameters
        # self.executor_parameters._text_field_embedder = self._text_field_embedder

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        # ques2action_encoder_outdim = self._ques2action_encoder.get_output_dim()
        # ques2action_encoder_outdim = 2*ques2action_encoder_outdim if self._ques2action_encoder.is_bidirectional() \
        #                                 else ques2action_encoder_outdim
        # quesspan_output_dim = 2 * ques2action_encoder_outdim   # For span concat
        #
        # assert quesspan_output_dim == action_embedding_dim

        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        self._action_embedding_dim = action_embedding_dim
        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

        self._qspan_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

    @overrides
    def forward(self, **kwargs):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    # def _get_initial_rnn_state(self, question: Dict[str, torch.LongTensor]):
    #     embedded_input = self._text_field_embedder(question)
    #     # (batch_size, sentence_length)
    #     question_mask = util.get_text_field_mask(question).float()
    #
    #     batch_size = embedded_input.size(0)
    #
    #     # (B, ques_length, encoder_output_dim)
    #     encoder_outputs = self._dropout(self._qencoder(embedded_input, question_mask))
    #
    #     # (B, encoder_output_dim)
    #     final_encoder_output = util.get_final_encoder_states(encoder_outputs,
    #                                                          question_mask,
    #                                                          self._qencoder.is_bidirectional())
    #     memory_cell = encoder_outputs.new_zeros(batch_size, self._qencoder.get_output_dim())
    #     # TODO(nitish): Why does WikiTablesParser use '_first_attended_question' embedding and not this
    #     attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
    #                                                                  encoder_outputs, question_mask)
    #     encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
    #     question_mask_list = [question_mask[i] for i in range(batch_size)]
    #     initial_rnn_state = []
    #     for i in range(batch_size):
    #         initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
    #                                              memory_cell[i],
    #                                              self._first_action_embedding,
    #                                              attended_sentence[i],
    #                                              encoder_outputs_list,
    #                                              question_mask_list))
    #     return initial_rnn_state, embedded_input, encoder_outputs_list, question_mask_list

    def _get_initial_rnn_state(self,
                               ques_repr: torch.FloatTensor,
                               ques_mask: torch.LongTensor,
                               question_final_repr: torch.FloatTensor,
                               ques_encoded_list: List[torch.FloatTensor],
                               ques_mask_list: List[torch.LongTensor]):
        """ Get the initial RnnStatelet for the decoder based on the question encoding

        Parameters:
        -----------
        ques_repr: (B, T, D)
        ques_mask: (B, T)
        question_final_repr: (B, D)
        ques_encoded_list: [(T, D)]
        ques_mask_list: [(T)]



        """

        batch_size = question_final_repr.size(0)
        ques_encoded_dim = question_final_repr.size()[-1]

        # Shape: (B, D)
        memory_cell = question_final_repr.new_zeros(batch_size, ques_encoded_dim)
        # TODO(nitish): Why does WikiTablesParser use '_first_attended_question' embedding and not this
        attended_sentence, _ = self._decoder_step.attend_on_question(question_final_repr,
                                                                     ques_repr, ques_mask)

        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(question_final_repr[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 attended_sentence[i],
                                                 ques_encoded_list,
                                                 ques_mask_list))
        return initial_rnn_state

    def _create_grammar_statelet(self,
                                 language: HotpotQALanguage,
                                 possible_actions: List[ProductionRule],
                                 linked_rule2idx: Dict = None,
                                 action2ques_linkingscore: torch.FloatTensor = None,
                                 quesspan_action_repr: torch.FloatTensor = None) -> GrammarStatelet:
        """ Make grammar state for a particular instance in the batch using the global and instance-specific actions.
        For each instance-specific action we have a linking_score vector (size:ques_tokens), and an action embedding

        Parameters:
        ------------
        world: `SampleHotpotWorld` The world for this instance
        possible_actions: All possible actions, global and instance-specific
        linked_rule2idx: Dict from action_rule to idx used for the next two members
        action2ques_linkingscore: Linking score matrix of size (instance-specific_actions, num_ques_tokens)
            The indexing is based on the linked_rule2idx dict. The num_ques_tokens is to a padded length
        quesspan_action_repr: Similarly, a (instance-specific_actions, action_embedding_dim) matrix.
            The indexing is based on the linked_rule2idx dict. The num_ques_tokens is to a padded length


        """
        # ProductionRule: (rule, is_global_rule, rule_id, nonterminal)
        action2actionidx = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            if action_string:
                # print("{} {}".format(action_string, action))
                action2actionidx[action_string] = action_index

        valid_actions = language.get_nonterminal_productions()
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}

        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action2actionidx[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]

            # For global_actions: (rule_vocab_id_tensor, action_index)
            global_actions = []
            # For linked_actions: (action_string, action_index)
            linked_actions = []

            for production_rule_array, action_index in production_rule_arrays:
                # production_rule_array: ProductionRule
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            if linked_actions:
                if linked_rule2idx is None:
                    raise RuntimeError('Linked rule info not given')

            # First: Get the embedded representations of the global actions
            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0)
                # TODO(nitish): Figure out if need action_bias and separate input/output action embeddings
                # if self._add_action_bias:
                #     global_action_biases = self._action_biases(global_action_tensor)
                #     global_input_embeddings = torch.cat([global_input_embeddings, global_action_biases], dim=-1)
                global_output_embeddings = self._action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_output_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))

            # Second: Get the representations of the linked actions
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                ques_spans_idxs = [linked_rule2idx[linked_rule] for linked_rule in linked_rules]
                # Scores and embedding should not be batched and
                # should be equal to the number of actions in this instance
                # (num_linked_actions, num_question_tokens)
                linked_action_scores = action2ques_linkingscore[ques_spans_idxs]
                # (num_linked_actions, action_embedding_dim)
                action_embedding_mat = self._qspan_action_embedding.unsqueeze(0).expand(len(linked_rules),
                                                                                        self._action_embedding_dim)
                linked_action_embeddings = action_embedding_mat   # quesspan_action_repr[ques_spans_idxs]

                translated_valid_actions[key]['linked'] = (linked_action_scores,
                                                           linked_action_embeddings,
                                                           list(linked_action_ids))

        # print(translated_valid_actions)

        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               language.is_nonterminal)

    def _get_label_strings(self, labels):
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = labels.detach().cpu()
        label_strings: List[List[str]] = []
        for instance_labels_data in labels_data:
            label_strings.append([])
            for label in instance_labels_data:
                label_int = int(label)
                if label_int == -1:
                    # Padding, because not all instances have the same number of labels.
                    continue
                label_strings[-1].append(self.vocab.get_token_from_index(label_int, "denotations"))
        return label_strings

    @classmethod
    def _get_actionseq_strings(cls,
                               possible_actions: List[List[ProductionRule]],
                               b2actionindices: Dict[int, List[List[int]]],
                               b2actionscores: Dict[int, List[torch.Tensor]],
                               b2debuginfos: Dict[int, List[List[Dict]]] = None) -> Tuple[List[List[List[str]]],
                                                                                          List[List[torch.Tensor]],
                                                                                          List[List[List[Dict]]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        all_action_scores: List[List[torch.Tensor]] = []
        all_debuginfos: List[List[List[Dict]]] = [] if b2debuginfos is not None else None
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            instance_actionindices = b2actionindices[i] if i in b2actionindices else []
            instance_actionscores = b2actionscores[i] if i in b2actionscores else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [[batch_actions[rule_id][0] for rule_id in sequence]
                              for sequence in instance_actionindices]
            all_action_strings.append(action_strings)
            all_action_scores.append(instance_actionscores)
            if b2debuginfos is not None:
                instance_debuginfos = b2debuginfos[i] if i in b2debuginfos else []
                all_debuginfos.append(instance_debuginfos)
        return all_action_strings, all_action_scores, all_debuginfos

    @staticmethod
    def _get_denotations(action_strings: List[List[List[str]]],
                         languages: List[HotpotQALanguage],
                         sideargs: List[List[List[Dict]]]=None) -> Tuple[List[List[Any]], List[List[str]]]:
        """ Get denotations for all action-sequences for  every instance in a batch.

        Parameters:
        -----------
        action_strings: ``List[List[List[str]]]``
            Each program represented as a list of actions(str),  for all decoded programs for each instance
        languages: ``List[HotpotQALanguage]``
            Language instance for each instance
        sideargs: ``List[List[List[Dict]]]``
            Required for languages that use side_args
            Debug-info as List[Dict] for each program. This list should be the same size as the number of actions in
            the program. This debug-info is present for each program, for each decoded program for each instance.
        """
        all_denotations: List[List[Any]] = []
        all_denotation_types: List[List[str]] = []
        wsideargs = True if sideargs else False
        for insidx in range(len(languages)):
            instance_language: HotpotQALanguage = languages[insidx]
            instance_action_sequences = action_strings[insidx]
            instance_sideargs = sideargs[insidx] if wsideargs else None
            instance_denotations: List[Any] = []
            instance_denotation_types: List[str] = []
            for pidx in range(len(instance_action_sequences)):
                action_sequence = instance_action_sequences[pidx]
                program_sideargs = instance_sideargs[pidx] if wsideargs else None
                # print(instance_action_strings)
                if not action_sequence:
                    continue
                # logical_form = instance_language.action_sequence_to_logical_form(action_sequence)
                # print(logical_form)
                actionseq_denotation = instance_language.execute_action_sequence(action_sequence, program_sideargs)
                # instance_actionseq_denotation = instance_language.execute(logical_form)
                instance_denotations.append(actionseq_denotation._value)
                instance_actionseq_type = instance_language.typeobj_to_typename(actionseq_denotation)
                instance_denotation_types.append(instance_actionseq_type)

            all_denotations.append(instance_denotations)
            all_denotation_types.append(instance_denotation_types)
        return all_denotations, all_denotation_types

    @staticmethod
    def _check_denotation(action_sequence: List[str],
                          labels: List[str],
                          languages: List[HotpotQALanguage]) -> List[bool]:
        is_correct = []
        for language, label in zip(languages, labels):
            logical_form = language.action_sequence_to_logical_form(action_sequence)
            denotation = language.execute(logical_form)
            is_correct.append(str(denotation).lower() == label)
        return is_correct

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. We only transform the action string sequences into logical
        forms here.
        """
        best_action_strings = output_dict["best_action_strings"]
        batch_actionseq_sideargs = output_dict["batch_actionseq_sideargs"] if self._wsideargs else None
        languages = output_dict["languages"]
        # This currectly works because there aren't any instance-specific arguments to the language.
        # language = HotpotQALanguage(qstr_qent_spans=[]) # NlvrWorld([])
        logical_forms = []
        execution_vals = []
        for insidx in range(len(languages)):
        # for instance_action_sequences, instance_action_sideargs, l in zip(best_action_strings,
        #                                                                   batch_actionseq_sideargs,
        #                                                                   languages):
            l: HotpotQALanguage = languages[insidx]
            instance_action_sequences = best_action_strings[insidx]
            instance_action_sideargs = batch_actionseq_sideargs[insidx] if self._wsideargs else None

            instance_logical_forms = []
            instance_execution_vals = []
            for pidx in range(len(instance_action_sequences)):
            # for action_strings, side_args in zip(instance_action_sequences, instance_action_sideargs):
                action_strings = instance_action_sequences[pidx]
                side_args = instance_action_sideargs[pidx] if instance_action_sideargs else None
                if action_strings:
                    instance_logical_forms.append(l.action_sequence_to_logical_form(action_strings))
                    # Custom function that copies the execution from domain_languages, but is used for debugging
                    denotation, ex_vals = dl_utils.execute_action_sequence(l, action_strings, side_args)
                    instance_execution_vals.append(ex_vals)
                else:
                    instance_logical_forms.append('')
                    instance_execution_vals.append([])
            logical_forms.append(instance_logical_forms)
            execution_vals.append(instance_execution_vals)

        # print(logical_forms[0][0])
        # print('\n')
        # print(execution_vals[0][0])
        # print('\n')

        output_dict["logical_forms"] = logical_forms
        output_dict["execution_vals"] = execution_vals
        output_dict.pop('languages', None)
        if not self._wsideargs:
            output_dict.pop('batch_actionseq_sideargs', None)

        # print('\n\n')
        # print(output_dict)

        return output_dict

    def _check_state_denotations(self, state: GrammarBasedState, language: HotpotQALanguage) -> List[bool]:
        """
        Returns whether action history in the state evaluates to the correct denotations over all
        worlds. Only defined when the state is finished.
        """
        assert state.is_finished(), "Cannot compute denotations for unfinished states!"
        # Since this is a finished state, its group size must be 1.
        batch_index = state.batch_indices[0]
        instance_label_strings = state.extras[batch_index]
        history = state.action_history[0]
        all_actions = state.possible_actions[0]
        action_sequence = [all_actions[action][0] for action in history]
        # This needs to change to a single world, and remove list
        return self._check_denotation(action_sequence, instance_label_strings, [language])