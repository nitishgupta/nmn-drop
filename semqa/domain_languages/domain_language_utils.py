from typing import Callable, List, Dict, Tuple, Any, Type, Set

import inspect

from allennlp.semparse.domain_languages.domain_language import DomainLanguage, ExecutionError


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
                execution_vals.append([(function.__name__, execution_value._value)])
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
                    execution_vals.append([(function.__name__, execution_value._value)])
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

        args_list = []
        arguments = []
        for _ in right_side_parts[1:]:
            argument, remaining_actions, remaining_side_args, args_list_i = _execute_sequence(language,
                                                                                             remaining_actions,
                                                                                             remaining_side_args,
                                                                                             [])
            arguments.append(argument)
            args_list.append(args_list_i[0])



        execution_value = function(*arguments)
        args_list.insert(0, (function.__name__, execution_value._value))

        execution_vals.insert(0, args_list)


        return execution_value, remaining_actions, remaining_side_args, execution_vals

