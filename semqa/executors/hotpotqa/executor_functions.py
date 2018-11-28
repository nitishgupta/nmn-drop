from typing import List, Set, Dict, Union, TypeVar, Callable
from collections import defaultdict
import operator
import logging

from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.worlds.nlvr_object import Object
from allennlp.semparse.worlds.nlvr_box import Box


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class ExecutorFunctions:

    @staticmethod
    def number_threshold(arg1):
        return arg1 > 10.0

    @staticmethod
    def number_greater(arg1, arg2):
        return arg1 > arg2

    @staticmethod
    def scalar_mult(arg1):
        return 5.0 * arg1


    @staticmethod
    def ground_num():
        return 10.0

    @staticmethod
    def two_ques_bool(arg1, arg2):
        arg1, arg2 = arg1[5:], arg2[5:]

        return len(arg1) > len(arg2)



