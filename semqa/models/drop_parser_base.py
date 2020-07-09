import logging
from typing import Dict, List, Tuple, Any, TypeVar, Optional
from overrides import overrides
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.training.metrics import Average
import allennlp.common.util as alcommon_utils
from allennlp.nn import RegularizerApplicator

from allennlp_semparse.state_machines.states import GrammarStatelet, RnnStatelet, State
from allennlp_semparse.fields.production_rule_field import ProductionRule

import semqa.domain_languages.domain_language_utils as dl_utils
from semqa.domain_languages.drop_language import DropLanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

StateType = TypeVar("StateType", bound=State)

START_SYMBOL = alcommon_utils.START_SYMBOL


class DROPParserBase(Model):
    """ DROP Parser BaseClass """

    def __init__(
        self,
        vocab: Vocabulary,
        action_embedding_dim: int,
        dropout: float = 0.0,
        rule_namespace: str = "rule_labels",
        debug: bool = False,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(DROPParserBase, self).__init__(vocab=vocab, regularizer=regularizer)

        # To call garbage collection frequently
        self.num_forward_calls = 0

        self._denotation_accuracy = Average()
        self._consistency = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        # This flag turns on the debugging mode which prints a bunch of stuff in self.decode (inside functions as well)
        self._debug = debug

        self._action_embedder = Embedding(
            num_embeddings=vocab.get_vocab_size(self._rule_namespace),
            embedding_dim=action_embedding_dim,
            vocab_namespace=self._rule_namespace,
        )

        self._action_embedding_dim = action_embedding_dim
        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding, mean=0.0, std=0.001)

    @overrides
    def forward(self, **kwargs):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(
        self,
        question_encoded: torch.FloatTensor,
        question_mask: torch.Tensor,
        question_encoded_finalstate: torch.FloatTensor,
        question_encoded_aslist: List[torch.Tensor],
        question_mask_aslist: List[torch.Tensor],
    ):
        """ Get the initial RnnStatelet for the decoder based on the question encoding

        Parameters:
        -----------
        ques_repr: (B, question_length, D)
        ques_mask: (B, question_length)
        question_final_repr: (B, D)
        ques_encoded_list: [(question_length, D)] - List of length B
        ques_mask_list: [(question_length)] - List of length B
        """

        batch_size = question_encoded_finalstate.size(0)
        ques_encoded_dim = question_encoded_finalstate.size()[-1]

        # Shape: (B, D)
        memory_cell = question_encoded_finalstate.new_zeros(batch_size, ques_encoded_dim)
        # TODO(nitish): Why does WikiTablesParser use '_first_attended_question' embedding and not this
        attended_sentence, _ = self._decoder_step.attend_on_question(
            question_encoded_finalstate, question_encoded, question_mask
        )

        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(
                RnnStatelet(
                    question_encoded_finalstate[i],
                    memory_cell[i],
                    self._first_action_embedding,
                    attended_sentence[i],
                    question_encoded_aslist,
                    question_mask_aslist,
                )
            )
        return initial_rnn_state

    def _create_grammar_statelet(
        self, language: DropLanguage, possible_actions: List[ProductionRule]
    ) -> Tuple[GrammarStatelet, Dict[str, int], List[str]]:
        # linked_rule2idx: Dict = None,
        # action2ques_linkingscore: torch.FloatTensor = None,
        # quesspan_action_emb: torch.FloatTensor = None) -> GrammarStatelet:
        """ Make grammar state for a particular instance in the batch using the global and instance-specific actions.
        For each instance-specific action we have a linking_score vector (size:ques_tokens), and an action embedding

        Parameters:
        ------------
        world: `SampleHotpotWorld` The world for this instance
        possible_actions: All possible actions, global and instance-specific

        linked_rule2idx: Dict from linked_action to idx used for the next two members
        action2ques_linkingscore: Linking score matrix of size (instance-specific_actions, num_ques_tokens)
            The indexing is based on the linked_rule2idx dict. The num_ques_tokens is to a padded length
            The num_ques_tokens is to a padded length, because of which not using a dictionary but a tensor.
        quesspan_action_emb: Similarly, a (instance-specific_actions, action_embedding_dim) matrix.
            The indexing is based on the linked_rule2idx dict.
        """
        # ProductionRule: (rule, is_global_rule, rule_id, nonterminal)
        action2actionidx = {}
        actionidx2actionstr: List[str] = []
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action2actionidx[action_string] = action_index
            actionidx2actionstr.append(action_string)

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

            for production_rule_array, action_index in production_rule_arrays:
                # production_rule_array: ProductionRule
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    raise NotImplementedError

            # First: Get the embedded representations of the global actions
            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0)
                # TODO(nitish): Figure out if need action_bias and separate input/output action embeddings
                # if self._add_action_bias:
                #     global_action_biases = self._action_biases(global_action_tensor)
                #     global_input_embeddings = torch.cat([global_input_embeddings, global_action_biases], dim=-1)
                global_output_embeddings = self._action_embedder(global_action_tensor)
                translated_valid_actions[key]["global"] = (
                    global_output_embeddings,
                    global_output_embeddings,
                    list(global_action_ids),
                )

        return (
            GrammarStatelet([START_SYMBOL], translated_valid_actions, language.is_nonterminal),
            action2actionidx,
            actionidx2actionstr,
        )

    @staticmethod
    def _get_denotations(
        action_strings: List[List[List[str]]], languages: List[DropLanguage], sideargs: List[List[List[Dict]]] = None
    ) -> Tuple[List[List[Any]], List[List[str]]]:
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
        for insidx in range(len(languages)):
            instance_language: DropLanguage = languages[insidx]
            instance_action_sequences = action_strings[insidx]
            instance_sideargs = sideargs[insidx]
            instance_denotations: List[Any] = []
            instance_denotation_types: List[str] = []
            for pidx in range(len(instance_action_sequences)):
                action_sequence = instance_action_sequences[pidx]
                program_sideargs = instance_sideargs[pidx]
                instance_language.modules_debug_info.append([])
                if not action_sequence:
                    continue
                actionseq_denotation = instance_language.execute_action_sequence(action_sequence, program_sideargs)
                # instance_actionseq_denotation = instance_language.execute(logical_form)
                instance_denotations.append(actionseq_denotation)
                instance_actionseq_type = (
                    actionseq_denotation.__class__.__name__
                )  # instance_language.typeobj_to_typename(actionseq_denotation)
                instance_denotation_types.append(instance_actionseq_type)

            all_denotations.append(instance_denotations)
            all_denotation_types.append(instance_denotation_types)
        return all_denotations, all_denotation_types
