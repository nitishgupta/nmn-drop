"""
We define a simple deterministic decoder here, that takes steps to add integers to list. At
each step, the decoder takes the last integer in the list, and adds either 1 or 2 to produce the
next element that will be added to the list. We initialize the list with the value 0 (or whatever
you pick), and we say that a sequence is finished when the last element is 4. We define the score
of a state as the negative of the number of elements (excluding the initial value) in the action
history.
"""
from collections import defaultdict
from typing import List, Set, Dict

from overrides import overrides
import torch

from allennlp_semparse.state_machines import State, TransitionFunction, BeamSearch
from semqa.state_machines.prefixed_beam_search import PrefixedConstrainedBeamSearch


class SimpleState(State["SimpleState"]):
    def __init__(
        self,
        batch_indices: List[int],
        action_history: List[List[int]],
        score: List[torch.Tensor],
        start_values: List[int] = None,
    ) -> None:
        super().__init__(batch_indices, action_history, score)
        self.start_values = start_values or [0] * len(batch_indices)

    def is_finished(self) -> bool:
        return self.action_history[0][-1] == 4

    @classmethod
    def combine_states(cls, states) -> "SimpleState":
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [
            action_history for state in states for action_history in state.action_history
        ]
        scores = [score for state in states for score in state.score]
        start_values = [start_value for state in states for start_value in state.start_values]
        return SimpleState(batch_indices, action_histories, scores, start_values)

    def __repr__(self):
        return f"{self.action_history}"


class SimpleTransitionFunction(TransitionFunction[SimpleState]):
    def __init__(
        self, valid_actions: Set[int] = None, include_value_in_score: bool = False
    ) -> None:
        # The default allowed actions are adding 1 or 2 to the last element.
        self._valid_actions = valid_actions or {1, 2}
        # If True, we will add a small multiple of the action take to the score, to encourage
        # getting higher numbers first (and to differentiate action sequences).
        self._include_value_in_score = include_value_in_score

    @overrides
    def take_step(
        self, state: SimpleState, max_actions: int = None, allowed_actions: List[Set] = None
    ) -> List[SimpleState]:
        indexed_next_states: Dict[int, List[SimpleState]] = defaultdict(list)
        if not allowed_actions:
            allowed_actions = [None] * len(state.batch_indices)
        for batch_index, action_history, score, start_value, actions in zip(
            state.batch_indices,
            state.action_history,
            state.score,
            state.start_values,
            allowed_actions,
        ):
            prev_action = action_history[-1] if action_history else start_value
            for action in self._valid_actions:
                next_item = int(prev_action + action)
                if actions and next_item not in actions:
                    continue
                new_history = action_history + [next_item]
                # For every action taken, we reduce the score by 1.
                new_score = score - 1
                if self._include_value_in_score:
                    new_score += 0.01 * next_item
                new_state = SimpleState([batch_index], [new_history], [new_score])
                indexed_next_states[batch_index].append(new_state)
        next_states: List[SimpleState] = []
        for batch_next_states in indexed_next_states.values():
            sorted_next_states = [(-state.score[0].data[0], state) for state in batch_next_states]
            sorted_next_states.sort(key=lambda x: x[0])
            if max_actions is not None:
                sorted_next_states = sorted_next_states[:max_actions]
            next_states.extend(state[1] for state in sorted_next_states)
        return next_states

def test_constraints():
    # The simple transition system starts at some number, adds one or two at each state, and
    # tries to get to 4.  The highest scoring path has the shortest length and the highest
    # numbers (so always add two, unless you're at 3).  From -3, there are lots of possible
    # sequences: [-2, -1, 0, 1, 2, 3, 4], [-1, 1, 3, 4], ...  We'll specify a few of those up
    # front as "allowed", and use that to test the constrained beam search implementation.
    initial_state = SimpleState(
        batch_indices=[0, 1, 2],
        action_history=[[], [], []],
        score=[torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0])],
        start_values=[-3, 0, -1]
    )

    beam_size = 5
    # initial_sequence = torch.Tensor([-2, -1, 0, 1])
    initial_sequence = [[[-2, -1], [-1, 0, 1]], [], [[1], [0, 2, 4]]]

    # bm = BeamSearch(beam_size, initial_sequence=torch.Tensor([-2, -1, 0, 1]))
    # print(bm._allowed_transitions)

    beam_search = PrefixedConstrainedBeamSearch(beam_size, allowed_sequences=initial_sequence)
    print(beam_search._allowed_transitions)

    decoder_step = SimpleTransitionFunction(include_value_in_score=True)
    # best_states = beam_search.search(7, initial_state, decoder_step)
    best_states = beam_search.search(num_steps=7, initial_state=initial_state, transition_function=decoder_step,
                                     keep_final_unfinished_states=True)

    print("\nBest States after Beam Search")
    for k, states in best_states.items():
        print(f"BatchIndex: {k}")
        for st in states:
            print(f"\t Action Seq: {st.action_history[0]}")
    print("----\n")
    exit()

    # assert len(best_states) == 1

    # After the constraint runs out, we generate [3], [2],
    # then we generate [3, 5], [3, 4], [2, 4], the latter two of which are finished,
    # then we generate [3, 5, 7], [3, 5, 6], and we're out of steps, so we keep the former
    assert best_states[0][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
    assert best_states[0][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
    assert best_states[0][2].action_history[0] == [-2, -1, 0, 1, 3, 5, 7]

    assert best_states[1][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
    assert best_states[1][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
    assert best_states[1][2].action_history[0] == [-2, -1, 0, 1, 3, 5, 7]


    # Now set the beam size to 6, we generate [3], [2]
    # then [3, 5], [2, 3], [3, 4], [2, 4] (the latter two of which are finished)
    # then [3, 5, 6], [3, 5, 7], [2, 3, 5], [2, 3, 4] (the last is finished)
    beam_size = 6
    beam_search = PrefixedBeamSearch(
        beam_size, initial_sequence=initial_sequence, keep_beam_details=True
    )
    decoder_step = SimpleTransitionFunction(include_value_in_score=True)
    best_states = beam_search.search(
        7, initial_state, decoder_step, keep_final_unfinished_states=False
    )

    assert len(best_states) == 2
    assert len(best_states[0]) == 3
    assert len(best_states[1]) == 3
    assert best_states[0][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
    assert best_states[0][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
    assert best_states[0][2].action_history[0] == [-2, -1, 0, 1, 2, 3, 4]

    assert best_states[1][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
    assert best_states[1][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
    assert best_states[1][2].action_history[0] == [-2, -1, 0, 1, 2, 3, 4]

    # Check that beams are correct
    best_action_sequence = best_states[0][0].action_history[0]

    beam_snapshots = beam_search.beam_snapshots
    assert len(beam_snapshots) == 2

    beam_snapshots0 = beam_snapshots.get(0)
    assert beam_snapshots0 is not None

    for i, beam in enumerate(beam_snapshots0):
        assert all(len(sequence) == i + 1 for _, sequence in beam)
        if i < len(best_action_sequence):
            assert any(sequence[-1] == best_action_sequence[i] for _, sequence in beam)


def main():
    test_constraints()


if __name__=="__main__":
    main()
