from typing import List, Set, Dict, Union, TypeVar, Callable, Tuple, Any
from collections import defaultdict
import operator
import logging

from allennlp.semparse import util as semparse_util

from semqa.executors.hotpotqa.executor_functions import ExecutorFunctions

import datasets.hotpotqa.utils.constants as hpconstants


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


class SampleHotpotExecutor:
    """
    Copied from the NLVR Executor
    (https://github.com/allenai/allennlp/blob/master/allennlp/semparse/executors/nlvr_executor.py)
    """

    # pylint: disable=too-many-public-methods
    def __init__(self) -> None:

        self.function_mappings = {
                                    'number_greater': ExecutorFunctions.number_greater,
                                    'scalar_mult': ExecutorFunctions.scalar_mult,
                                    'multiply': ExecutorFunctions.multiply,
                                    'ground_num': ExecutorFunctions.ground_num,
                                    'number_threshold': ExecutorFunctions.number_threshold,
                                    'two_ques_bool': ExecutorFunctions.two_ques_bool,
                                    'ques_bool': ExecutorFunctions.ques_bool
                                }

        self.func2returntype_mappings = {
            'number_greater': hpconstants.BOOL_TYPE,
            'scalar_mult': hpconstants.NUM_TYPE,
            'multiply': hpconstants.NUM_TYPE,
            'ground_num': hpconstants.NUM_TYPE,
            'number_threshold': hpconstants.BOOL_TYPE,
            'two_ques_bool': hpconstants.BOOL_TYPE,
            'ques_bool': hpconstants.BOOL_TYPE
        }


    def grounded_argument(self, arg):
        """
        This function takes an arg and returns whether it is grounded or not.
        For eg. 5 is grounded, but an expression list ['scalar_mult', ['scalar_mult', 'ground_num']] is not.

        The logic here can be tricky and we'll update it as we go.
        Basics:
            List: are not grounded if the first element is a function (expr. list), else assume grounded.
            Strings: Grounded: if the string is such that it can be a linked action (example copying question strings)
                Other check could be to maintain a list of non-terminal functions, but this requires bookkeeping.
            Other types: Assume grounded

        Returns:
        --------
        is_grounded: Bool
        """

        is_grounded = True
        if isinstance(arg, list):
            is_grounded = False

        if isinstance(arg, str):
            if arg in self.function_mappings:
                is_grounded = False

        return is_grounded

    def _is_expr_list(self, arg):
        """ Returns true if the arg is an expression_list.

        Simple implementation currently just checks if the arg is a list.
        Will need sophisticated logic if one of the groundings could be a list
        """
        if isinstance(arg, list):
            return True
        else:
            return False

    def execute(self, logical_form: str) -> Tuple[Any, Any]:

        if not logical_form.startswith("("):
            logical_form = f"({logical_form})"
        logical_form = logical_form.replace(",", " ")

        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)

        # print(expression_as_list[0])

        denotation = self._handle_expression(expression_as_list[0])

        outer_most_function = expression_as_list[0][0]
        denotation_type = self.func2returntype_mappings[outer_most_function]

        # print(f"Denotation: {denotation} with type: {denotation_type}")

        return denotation, denotation_type

    # def _handle_expression(self, expression_list):
    #     print(expression_list)
    #
    #     exp_list_stack = Stack()
    #     arg_cache = []
    #
    #     for item in expression_list:
    #         exp_list_stack.push(item)
    #
    #     while not exp_list_stack.isEmpty():
    #         top_item = exp_list_stack.pop()
    #
    #         if self.grounded_argument(top_item):
    #             arg_cache.append(top_item)
    #
    #         elif self._is_expr_list(top_item):
    #             for item in top_item:
    #                 exp_list_stack.push(item)
    #
    #         elif top_item in self.function_mappings:
    #             print(top_item)
    #             print(arg_cache)
    #
    #             # Arguments are inserted in the arg_cache in reverse order
    #             func_args = arg_cache[::-1]
    #             if len(func_args) > 0:
    #                 return_val = self.function_mappings[top_item](*func_args)
    #             else:
    #                 return_val = self.function_mappings[top_item]()
    #             exp_list_stack.push(return_val)
    #             arg_cache = []
    #
    #         else:
    #             print("Shouldn't be here")
    #             print(exp_list_stack.items)
    #             print(arg_cache)
    #
    #     return arg_cache[0]


    def _handle_expression(self, expression_list):
        assert isinstance(expression_list, list), f"Expression list is not a list: {expression_list}"

        assert expression_list[0] in self.function_mappings, f"Expression_list[0] not implemented: {expression_list[0]}"

        args = []

        function_name = expression_list[0]

        for item in expression_list[1:]:
            if self.grounded_argument(item):
                args.append(item)
            elif self._is_expr_list(item):
                args.append(self._handle_expression(item))
            elif (not isinstance(item, list)) and (item in self.function_mappings):
                # Function with zero arguments
                args.append(self.function_mappings[item]())
            else:
                print(f"Why am I here. Item: {item}")

        return self.function_mappings[function_name](*args)


if __name__=='__main__':
    executor = SampleHotpotExecutor()
    ans = executor.execute("(stack)")

    print(ans)