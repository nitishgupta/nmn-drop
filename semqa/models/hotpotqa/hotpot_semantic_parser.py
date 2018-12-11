import logging
from typing import Dict, List, Tuple, Any

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn import util
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average
from allennlp.modules.span_extractors.span_extractor import SpanExtractor

from semqa.worlds.hotpotqa.sample_world import SampleHotpotWorld
from semqa.worlds.hotpotqa.sample_world import QSTR_PREFIX

from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.models.semantic_parsing.wikitables.wikitables_mml_semantic_parser import WikiTablesMmlSemanticParser


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class HotpotSemanticParser(Model):
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
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 qencoder: Seq2SeqEncoder,
                 ques2action_encoder: Seq2SeqEncoder,
                 quesspan_extractor: SpanExtractor,
                 context_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super(HotpotSemanticParser, self).__init__(vocab=vocab)

        self._question_embedder = question_embedder
        self._denotation_accuracy = Average()
        self._consistency = Average()
        self._qencoder = qencoder
        self._ques2action_encoder = ques2action_encoder
        self._quesspan_extractor = quesspan_extractor

        self._context_embedder = context_embedder
        self._context_encoder = context_encoder

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        ques2action_encoder_outdim = self._ques2action_encoder.get_output_dim()
        ques2action_encoder_outdim = 2*ques2action_encoder_outdim if self._ques2action_encoder.is_bidirectional() \
                                        else ques2action_encoder_outdim
        quesspan_output_dim = 2 * ques2action_encoder_outdim   # For span concat

        assert quesspan_output_dim == action_embedding_dim

        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

    @overrides
    def forward(self, **kwargs):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(self, question: Dict[str, torch.LongTensor]):
        embedded_input = self._question_embedder(question)
        # (batch_size, sentence_length)
        question_mask = util.get_text_field_mask(question).float()

        batch_size = embedded_input.size(0)

        # (B, ques_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._qencoder(embedded_input, question_mask))

        # (B, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             question_mask,
                                                             self._qencoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._qencoder.get_output_dim())
        # TODO(nitish): Why does WikiTablesParser use '_first_attended_question' embedding and not this
        attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                     encoder_outputs, question_mask)
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 attended_sentence[i],
                                                 encoder_outputs_list,
                                                 question_mask_list))
        return initial_rnn_state

    def _create_grammar_statelet(self,
                                 world: SampleHotpotWorld,
                                 possible_actions: List[ProductionRule],
                                 linked_rule2idx: Dict,
                                 action2ques_linkingscore: torch.FloatTensor,
                                 quesspan_action_repr: torch.FloatTensor) -> GrammarStatelet:
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
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.get_valid_actions()
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}

        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action_map[action_string] for action_string in action_strings]
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
                # ques_spans = [rule.split(' -> ')[1] for rule in linked_rules]
                # ques_spans = [ques_span[len(QSTR_PREFIX):] for ques_span in ques_spans]
                ques_spans_idxs = [linked_rule2idx[linked_rule] for linked_rule in linked_rules]
                # Scores and embedding should not be batched and should be equal to the number of actions in this instance
                # (num_linked_actions, num_question_tokens)
                linked_action_scores = action2ques_linkingscore[ques_spans_idxs]
                # (num_linked_actions, action_embedding_dim)
                linked_action_embeddings = quesspan_action_repr[ques_spans_idxs]

                # print(f"NumLinkedActions: {len(linked_rule2idx)}\n")
                # print(f"ActionLinkingScore size: {action2ques_linkingscore.size()}\n")
                # print(f"LinkedActionRepr size: {quesspan_action_repr.size()}\n")
                # print("After Indexing")
                # print(f"ActionLinkingScore size: {linked_action_scores.size()}\n")
                # print(f"LinkedActionRepr size: {linked_action_embeddings.size()}\n")

                translated_valid_actions[key]['linked'] = (linked_action_scores,
                                                           linked_action_embeddings,
                                                           list(linked_action_ids))
        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               type_declaration.is_nonterminal)

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
    def _get_action_strings(cls,
                            possible_actions: List[List[ProductionRule]],
                            action_indices: Dict[int, List[List[int]]],
                            action_scores: Dict[int, List[torch.Tensor]]) -> Tuple[List[List[List[str]]],
                                                                                  List[List[torch.Tensor]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        all_action_scores: List[List[torch.Tensor]] = []
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            batch_best_sequences = action_indices[i] if i in action_indices else []
            batch_best_seqscores = action_scores[i] if i in action_scores else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [[batch_actions[rule_id][0] for rule_id in sequence]
                              for sequence in batch_best_sequences]
            all_action_strings.append(action_strings)
            all_action_scores.append(batch_best_seqscores)
        return all_action_strings, all_action_scores

    @staticmethod
    def _get_denotations(action_strings: List[List[List[str]]],
                         worlds: List[SampleHotpotWorld]) -> Tuple[List[List[Any]], List[List[str]]]:
        all_denotations: List[List[Any]] = []
        all_denotation_types: List[List[str]] = []

        for instance_world, instance_action_sequences in zip(worlds, action_strings):
            instance_world: SampleHotpotWorld = instance_world
            instance_denotations: List[Any] = []
            instance_denotation_types: List[str] = []
            for instance_action_strings in instance_action_sequences:
                # print(instance_action_strings)
                if not instance_action_strings:
                    continue

                logical_form = instance_world.get_logical_form(instance_action_strings)
                instance_actionseq_denotation, instance_actionseq_type = instance_world.execute(logical_form)
                instance_denotations.append(instance_actionseq_denotation)
                instance_denotation_types.append(instance_actionseq_type)

            all_denotations.append(instance_denotations)
            all_denotation_types.append(instance_denotation_types)
        return all_denotations, all_denotation_types

    @staticmethod
    def _check_denotation(action_sequence: List[str],
                          labels: List[str],
                          worlds: List[SampleHotpotWorld]) -> List[bool]:
        is_correct = []
        for world, label in zip(worlds, labels):
            logical_form = world.get_logical_form(action_sequence)
            denotation = world.execute(logical_form)
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
        # Instantiating an empty world for getting logical forms from action strings.
        world = SampleHotpotWorld() # NlvrWorld([])
        logical_forms = []
        for instance_action_sequences in best_action_strings:
            instance_logical_forms = []
            for action_strings in instance_action_sequences:
                if action_strings:
                    instance_logical_forms.append(world.get_logical_form(action_strings))
                else:
                    instance_logical_forms.append('')
            logical_forms.append(instance_logical_forms)
        output_dict["logical_form"] = logical_forms
        return output_dict

    def _check_state_denotations(self, state: GrammarBasedState, worlds: SampleHotpotWorld) -> List[bool]:
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
        return self._check_denotation(action_sequence, instance_label_strings, [worlds])