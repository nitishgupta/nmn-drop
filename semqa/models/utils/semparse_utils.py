from typing import Dict, List, Tuple, Any, TypeVar

import torch

from allennlp_semparse.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet, State
from allennlp_semparse.fields.production_rule_field import ProductionRule
import allennlp.nn.util as allenutil

StateType = TypeVar("StateType", bound=State)


def _convert_finalstates_to_actions(
    best_final_states: Dict[int, List[StateType]], possible_actions: List[List[ProductionRule]], batch_size: int
) -> Tuple[List[List[List[int]]], List[List[List[str]]], List[List[torch.Tensor]], List[List[List[Dict]]]]:

    (
        instanceidx2actionseq_idxs,
        instanceidx2actionseq_scores,
        instanceidx2actionseq_sideargs,
    ) = _get_actionseq_idxs_and_scores(best_final_states, batch_size)

    return _get_actionseq_strings(
        possible_actions, instanceidx2actionseq_idxs, instanceidx2actionseq_scores, instanceidx2actionseq_sideargs
    )


def _get_actionseq_idxs_and_scores(best_final_states: Dict[int, List[StateType]], batch_size: int):
    instanceidx2actionseq_idxs = {}
    instanceidx2actionseq_scores = {}
    instanceidx2actionseq_sideargs = {}

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
            instance_actionseq_sideargs = [final_state.debug_info[0] for final_state in best_final_states[i]]
            instanceidx2actionseq_sideargs[i] = instance_actionseq_sideargs

    return (instanceidx2actionseq_idxs, instanceidx2actionseq_scores, instanceidx2actionseq_sideargs)


def _get_actionseq_strings(
    possible_actions: List[List[ProductionRule]],
    b2actionindices: Dict[int, List[List[int]]],
    b2actionscores: Dict[int, List[torch.Tensor]],
    b2debuginfos: Dict[int, List[List[Dict]]] = None,
) -> Tuple[List[List[List[int]]], List[List[List[str]]], List[List[torch.Tensor]], List[List[List[Dict]]]]:
    """
    Takes a list of possible actions and indices of decoded actions into those possible actions
    for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
    mapping batch indices to k-best decoded sequence lists.
    """
    all_action_indices: List[List[List[int]]] = []
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
        action_strings = [[batch_actions[rule_id][0] for rule_id in sequence] for sequence in instance_actionindices]

        all_action_indices.append(instance_actionindices)
        all_action_strings.append(action_strings)
        all_action_scores.append(instance_actionscores)
        if b2debuginfos is not None:
            instance_debuginfos = b2debuginfos[i] if i in b2debuginfos else []
            all_debuginfos.append(instance_debuginfos)

    # batch_actionseq_probs = _convert_actionscores_to_probs(batch_actionseq_scores=all_action_scores)
    # return all_action_indices, all_action_strings, all_action_scores, batch_actionseq_probs, all_debuginfos
    return all_action_indices, all_action_strings, all_action_scores, all_debuginfos


def _convert_actionscores_to_probs(batch_actionseq_scores: List[List[torch.Tensor]]) -> List[torch.FloatTensor]:
    """ Normalize program scores in a beam for an instance to get probabilities

    Returns:
    ---------
    List[torch.FloatTensor]:
        For each instance, a tensor the size of number of predicted programs
        containing normalized probabilities
    """
    # Convert batch_action_scores to a single tensor the size of number of programs for each instance
    device_id = allenutil.get_device_of(batch_actionseq_scores[0][0])
    # Inside List[torch.Tensor] is a list of scalar-tensor with prob of each program for this instance
    # The prob is normalized across the programs in the beam
    batch_actionseq_probs = []
    for score_list in batch_actionseq_scores:
        scores_astensor = allenutil.move_to_device(torch.cat([x.view(1) for x in score_list]), device_id)
        # allenutil.masked_softmax(scores_astensor, mask=None)
        action_probs = torch.nn.functional.softmax(scores_astensor, dim=-1)
        batch_actionseq_probs.append(action_probs)

    return batch_actionseq_probs
