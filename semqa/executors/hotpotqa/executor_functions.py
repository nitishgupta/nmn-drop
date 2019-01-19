from typing import List, Set, Dict, Union, TypeVar, Callable
import torch
import logging


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


def convert_float_to_tensor(val):
    if val == 1.0:
        val = 0.9
    else:
        val = 0.1
    return torch.Tensor([val]).cuda()

class ExecutorFunctions:

    @staticmethod
    def number_threshold(arg1):
        return convert_float_to_tensor((arg1 > 10.0) * 1.0)

    @staticmethod
    def number_greater(arg1, arg2):
        return convert_float_to_tensor((arg1 > arg2) * 1.0)

    @staticmethod
    def scalar_mult(arg1):
        return convert_float_to_tensor(5.0 * arg1)

    @staticmethod
    def multiply(arg1, arg2):
        return convert_float_to_tensor(arg1 * arg2)


    @staticmethod
    def ground_num():
        return convert_float_to_tensor(10.0)

    @staticmethod
    def two_ques_bool(arg1, arg2):
        arg1, arg2 = arg1[5:], arg2[5:]

        return convert_float_to_tensor((len(arg1) > len(arg2)) * 1.0)

    @staticmethod
    def ques_bool(arg1):
        if len(arg1) > 5:
            return convert_float_to_tensor(1.0)
        else:
            return convert_float_to_tensor(0.0)

    @staticmethod
    def ques_ent_bool(arg1):
        if len(arg1) > 5:
            return convert_float_to_tensor(1.0)
        else:
            return convert_float_to_tensor(0.0)







