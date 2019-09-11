from typing import Callable, List, Dict, Tuple, Any, Type, Set

import inspect
import torch

import utils.util as myutils

from allennlp.semparse.domain_languages.domain_language import DomainLanguage, ExecutionError
import allennlp.nn.util as allenutil


def execute_action_sequence(language, action_sequence: List[str], side_arguments: List[Dict] = None):
    """
    Executes the program defined by an action sequence directly, without needing the overhead
    of translating to a logical form first.  For any given program, :func:`execute` and this
    function are equivalent, they just take different representations of the program, so you
    can use whichever is more efficient.

    Also, if you have state or side arguments associated with particular production rules
    (e.g., the decoder's attention on an input utterance when a predicate was predicted), you
    `must` use this function to execute the logical form, instead of :func:`execute`, so that
    we can match the side arguments with the right functions.
    """

    # We'll strip off the first action, because it doesn't matter for execution.
    first_action = action_sequence[0]
    left_side, right_side = first_action.split(' -> ')
    if left_side != '@start@':
        raise ExecutionError('invalid action sequence')
    remaining_side_args = side_arguments[1:] if side_arguments else None

    execution_vals = []

    execution_value, _, _, execution_vals = _execute_sequence(language, action_sequence[1:],
                                                              remaining_side_args, execution_vals)

    return execution_value, execution_vals[0]


def _execute_sequence(language,
                      action_sequence: List[str],
                      side_arguments: List[Dict],
                      execution_vals: List[Any]) -> Tuple[Any, List[str], List[Dict], List[Any]]:
    """
    This does the bulk of the work of :func:`execute_action_sequence`, recursively executing
    the functions it finds and trimming actions off of the action sequence.  The return value
    is a tuple of (execution, remaining_actions), where the second value is necessary to handle
    the recursion.
    """
    first_action = action_sequence[0]
    remaining_actions = action_sequence[1:]
    remaining_side_args = side_arguments[1:] if side_arguments else None
    left_side, right_side = first_action.split(' -> ')
    if right_side in language._functions:
        function = language._functions[right_side]
        # mypy doesn't like this check, saying that Callable isn't a reasonable thing to pass
        # here.  But it works just fine; I'm not sure why mypy complains about it.
        if isinstance(function, Callable):  # type: ignore
            function_arguments = inspect.signature(function).parameters
            if not function_arguments:
                # This was a zero-argument function / constant that was registered as a lambda
                # function, for consistency of execution in `execute()`.
                execution_value = function()
                execution_vals.append([(function.__name__, execution_value.debug_value)])
            elif side_arguments:
                kwargs = {}
                non_kwargs = []
                for argument_name in function_arguments:
                    if argument_name in side_arguments[0]:
                        kwargs[argument_name] = side_arguments[0][argument_name]
                    else:
                        non_kwargs.append(argument_name)
                if kwargs and non_kwargs:
                    # This is a function that has both side arguments and logical form
                    # arguments - we curry the function so only the logical form arguments are
                    # left.
                    def curried_function(*args):
                        return function(*args, **kwargs)
                    execution_value = curried_function
                elif kwargs:
                    # This is a function that _only_ has side arguments - we just call the
                    # function and return a value.
                    execution_value = function(**kwargs)
                    execution_vals.append([(function.__name__, execution_value.debug_value)])
                else:
                    # This is a function that has logical form arguments, but no side arguments
                    # that match what we were given - just return the function itself.
                    execution_value = function
            else:
                execution_value = function
        return execution_value, remaining_actions, remaining_side_args, execution_vals
    else:
        # This is a non-terminal expansion, like 'int -> [<int:int>, int, int]'.  We need to
        # get the function and its arguments, then call the function with its arguments.
        # Because we linearize the abstract syntax tree depth first, left-to-right, we can just
        # recursively call `_execute_sequence` for the function and all of its arguments, and
        # things will just work.
        right_side_parts = right_side.split(', ')

        # We don't really need to know what the types are, just how many of them there are, so
        # we recurse the right number of times.
        function, remaining_actions, remaining_side_args, execution_vals = _execute_sequence(language,
                                                                                             remaining_actions,
                                                                                             remaining_side_args,
                                                                                             execution_vals)

        args_exval_list = []
        arguments = []
        for _ in right_side_parts[1:]:
            argument, remaining_actions, remaining_side_args, args_exval_list_i = _execute_sequence(language,
                                                                                                    remaining_actions,
                                                                                                    remaining_side_args,
                                                                                                    [])
            arguments.append(argument)
            args_exval_list.append(args_exval_list_i[0])

        execution_value = function(*arguments)
        args_exval_list.insert(0, (function.__name__, execution_value.debug_value))

        execution_vals.insert(0, args_exval_list)

        return execution_value, remaining_actions, remaining_side_args, execution_vals


def masking_blockdiagonal(passage_length, window, device_id):
    """ Make a (passage_length, passage_length) tensor M of 1 and -1 in which for each row x,
        M[:, x, y] = -1 if y < x - window or y > x + window, else it is 1.
        Basically for the x-th row, the [x-win, x+win] columns should be 1, and rest -1
    """

    lower_limit = [max(0, i - window) for i in range(passage_length)]
    upper_limit = [min(passage_length, i + window) for i in range(passage_length)]

    # Tensors of lower and upper limits for each row
    lower = allenutil.move_to_device(torch.LongTensor(lower_limit), cuda_device=device_id)
    upper = allenutil.move_to_device(torch.LongTensor(upper_limit), cuda_device=device_id)
    lower_un = lower.unsqueeze(1)
    upper_un = upper.unsqueeze(1)

    # Range vector for each row
    lower_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)
    upper_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)

    # Masks for lower and upper limits of the mask
    lower_mask = lower_range_vector >= lower_un
    upper_mask = upper_range_vector <= upper_un

    # Final-mask that we require
    # Shape: (passage_length, passage_length); (passage_length, passage_length)
    inwindow_mask = (lower_mask == upper_mask).float()
    outwindow_mask = (lower_mask != upper_mask).float()

    return inwindow_mask, outwindow_mask


def aux_window_loss(ptop_attention, passage_mask, inwindow_mask):
    """Auxiliary loss to encourage p-to-p attention to be within a certain window.

    Args:
        ptop_attention: (passage_length, passage_length)
        passage_mask: (passage_length)
        inwindow_mask: (passage_length, passage_length)

    Returns:
        inwindow_aux_loss: ()
    """
    inwindow_mask = inwindow_mask * passage_mask.unsqueeze(0)
    inwindow_mask = inwindow_mask * passage_mask.unsqueeze(1)
    inwindow_probs = ptop_attention * inwindow_mask
    # Sum inwindow_probs for each token, signifying the token can distribute its alignment prob in any way
    # Shape: (passage_length)
    sum_inwindow_probs = inwindow_probs.sum(1)
    # Shape: (passage_length) -- mask for tokens that have empty windows
    mask_sum = (inwindow_mask.sum(1) > 0).float()
    masked_sum_inwindow_probs = allenutil.replace_masked_values(sum_inwindow_probs, mask_sum, replace_with=1e-40)
    log_sum_inwindow_probs = torch.log(masked_sum_inwindow_probs + 1e-40) * mask_sum
    inwindow_likelihood = torch.sum(log_sum_inwindow_probs)
    inwindow_likelihood_avg = inwindow_likelihood / torch.sum(inwindow_mask)

    inwindow_aux_loss = -1.0 * inwindow_likelihood_avg

    return inwindow_aux_loss


def mostAttendedSpans(attention_vec: torch.FloatTensor, tokens: List[str], span_length=5):
    """ Visualize an attention vector for a list of tokens

        Parameters:
        ----------
        attention_vec: Shape: (sequence_length, )
            Padded vector containing attention over a sequence
        question_tokens: List[str]
            List of tokens in the sequence

        Returns:
        --------
    """

    attention_aslist: List[float] = myutils.round_all(myutils.tocpuNPList(attention_vec), 3)
    tokens_len = len(tokens)
    # To remove padded elements
    attention_aslist: List[float] = attention_aslist[:tokens_len]

    span2atten = {}
    for start in range(0, len(tokens) - span_length + 1):
        end = start + span_length
        attention_sum = sum(attention_aslist[start:end])
        span2atten[(start, end)] = attention_sum

    sorted_spanattn = myutils.sortDictByValue(span2atten, decreasing=True)

    top_spans = [sorted_spanattn[0][0]]
    idx = 1
    while len(top_spans) < 5 and idx < len(sorted_spanattn):
        span = sorted_spanattn[idx][0]
        keep_span = True
        for in_span in top_spans:
            if myutils.isSpanOverlap(span, in_span):
                keep_span = False

        if keep_span:
            top_spans.append(span)
        idx += 1

    most_attention_spans = [" ".join(tokens[start:end]) for (start, end) in top_spans]
    attention_values = [span2atten[span] for span in top_spans]
    out_str = ""
    for span, attn in zip(most_attention_spans, attention_values):
        out_str += "{}:{} | ".format(span, attn)
    out_str = out_str.strip()

    return out_str


def listTokensVis(attention_vec: torch.FloatTensor, tokens: List[str]):
    """ Visualize an attention vector for a list of tokens

        Parameters:
        ----------
        attention_vec: Shape: (sequence_length, )
            Padded vector containing attention over a sequence
        question_tokens: List[str]
            List of tokens in the sequence

        Returns:
        --------
        complete_attention_vis: str
        most_attended_vis: String visualization of question attention
    """

    attention_aslist: List[float] = myutils.round_all(myutils.tocpuNPList(attention_vec), 3)
    tokens_len = len(tokens)
    # To remove padded elements
    attention_aslist: List[float] = attention_aslist[:tokens_len]

    complete_attention_vis = ""
    for token, attn in zip(tokens, attention_aslist):
        complete_attention_vis += f"{token}|{attn} "

    # List[(token, attn)]
    sorted_token_attn = sorted([(x, y)for x, y in zip(tokens, attention_aslist)], key=lambda x: x[1], reverse=True)
    most_attended_token_attn = sorted_token_attn[:10]
    most_attended_vis = "Most attended: "
    for token, attn in most_attended_token_attn:
        most_attended_vis += f"{token}|{attn} "

    return complete_attention_vis.strip(), most_attended_vis.strip()


