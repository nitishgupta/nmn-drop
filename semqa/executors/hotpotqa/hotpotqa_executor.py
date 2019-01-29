import copy
from typing import List, Set, Dict, Union, TypeVar, Callable, Tuple, Any
from collections import defaultdict
import operator
import logging

import torch
import torch.nn.functional
from allennlp.semparse import util as semparse_util

from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
import allennlp.nn.util as allenutil
from allennlp.common.registrable import Registrable


import datasets.hotpotqa.utils.constants as hpconstants


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

AttributeType = TypeVar('AttributeType', str, int)  # pylint: disable=invalid-name


class ExecutorParameters(torch.nn.Module, Registrable):
    """
    Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 ques_encoder: Seq2SeqEncoder,
                 context_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder):
        super(ExecutorParameters, self).__init__()
        self._ques_encoder = ques_encoder
        self._context_embedder = context_embedder
        self._context_encoder = context_encoder
        self._span_extractor = EndpointSpanExtractor(input_dim=self._context_encoder.get_output_dim())


    def _encode_contexts(self, contexts: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        """ Encode the contexts for each instance using the context_encoder RNN.

        Params:
        -------
        contexts: ``Dict[str, torch.LongTensor]``
            Since there are multiple contexts per instance, the contexts tensor are wrapped as (B, C, T)
            where C is the number of contexts

        Returns:
        --------
        contexts_encoded: ``torch.FloatTensor``
            Tensor of shape (B, C, T, D) after encoding all the contexts for each instance in the batch
        """

        # Shape: (B, C, T, W_d)
        embedded_contexts = self._context_embedder(contexts)
        embcontext_size = embedded_contexts.size()

        # Shape: (B, C, T)
        # Since multiple contexts per instance, give num_wrapping_dims
        contexts_mask = allenutil.get_text_field_mask(contexts, num_wrapping_dims=1).float()
        conmask_size = contexts_mask.size()

        (embedded_contexts_flat, contexts_mask_flat) = (
        embedded_contexts.view(-1, embcontext_size[2], embcontext_size[3]),
        contexts_mask.view(-1, conmask_size[2]))

        # Shape: (B*C, T, D)
        contexts_encoded_flat = self._context_encoder(embedded_contexts_flat, contexts_mask_flat)
        conenc_size = contexts_encoded_flat.size()
        # View such that get B, C, T from embedded context, and D from encoded contexts
        # Shape: (B, C, T, D)
        contexts_encoded = contexts_encoded_flat.view(*embcontext_size[0:3], conenc_size[-1])

        return contexts_encoded, contexts_mask


# class ExecutorArguments():



class SampleHotpotExecutor:
    """
    Copied from the NLVR Executor
    (https://github.com/allenai/allennlp/blob/master/allennlp/semparse/executors/nlvr_executor.py)
    """

    # pylint: disable=too-many-public-methods
    def __init__(self) -> None:

        self._executor_parameters: ExecutorParameters = None
        self.function_mappings = {
                                    # 'number_greater': self..number_greater,
                                    # 'scalar_mult': ExecutorFunctions.scalar_mult,
                                    # 'multiply': ExecutorFunctions.multiply,
                                    # 'ground_num': ExecutorFunctions.ground_num,
                                    # 'number_threshold': ExecutorFunctions.number_threshold,
                                    # 'two_ques_bool': ExecutorFunctions.two_ques_bool,
                                    # 'ques_bool': ExecutorFunctions.ques_bool,
                                    # 'ques_ent_bool': ExecutorFunctions.ques_ent_bool,
                                    'bool_qent_qstr': self.bool_qent_qstr,
                                    'bool_and': self.bool_and,
                                    'bool_or': self.bool_or
                                }

        self.func2returntype_mappings = {
            # 'number_greater': hpconstants.BOOL_TYPE,
            # 'scalar_mult': hpconstants.NUM_TYPE,
            # 'multiply': hpconstants.NUM_TYPE,
            # 'ground_num': hpconstants.NUM_TYPE,
            # 'number_threshold': hpconstants.BOOL_TYPE,
            # 'two_ques_bool': hpconstants.BOOL_TYPE,
            # 'ques_bool': hpconstants.BOOL_TYPE,
            # 'ques_ent_bool': hpconstants.BOOL_TYPE,
            'bool_qent_qstr': hpconstants.BOOL_TYPE,
            'bool_and': hpconstants.BOOL_TYPE,
            'bool_or': hpconstants.BOOL_TYPE
        }
        self.ques_embedded = None
        self.ques_mask = None
        self.contexts = None
        self.ne_ent_mens = None
        self.num_ent_mens = None
        self.date_ent_mens = None
        self.q_qstr2idx = None
        self.q_qstr_spans = None
        self.q_nemens2groundingidx = None
        self.q_nemens_grounding = None
        self.q_nemenspan2entidx = None

    def set_arguments(self, **kwargs):
        self.ques_embedded = kwargs["ques_embedded"]
        self.ques_mask = kwargs["ques_mask"]
        self.contexts = kwargs["contexts"]
        self.contexts_mask = kwargs["contexts_mask"]
        self.ne_ent_mens = kwargs["ne_ent_mens"]
        self.num_ent_mens = kwargs["num_ent_mens"]
        self.date_ent_mens = kwargs["date_ent_mens"]
        self.q_qstr2idx= kwargs["q_qstr2idx"]
        self.q_qstr_spans = kwargs["q_qstr_spans"]
        self.q_nemens2groundingidx = kwargs["q_nemens2groundingidx"]
        self.q_nemens_grounding = kwargs["q_nemens_grounding"]
        self.q_nemenspan2entidx = kwargs["q_nemenspan2entidx"]


    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """

        # For all QSTR, computing repr by running ques_encoder, and concatenating repr of end-points.
        self.qstr2repr = {}
        for qstr, qstr_idx in self.q_qstr2idx.items():
            # Shape: [2] denoting the span in the question
            qstr_span = self.q_qstr_spans[qstr_idx]
            # Shape: (QSTR_len, emb_dim)
            qstr_embedded = self.ques_embedded[qstr_span[0]:qstr_span[1] + 1]
            qstr_mask = self.ques_mask[qstr_span[0]:qstr_span[1] + 1]
            # [1, QSTR_len, emb_dim]
            qstr_encoded_ex = self._executor_parameters._ques_encoder(qstr_embedded.unsqueeze(0), qstr_mask.unsqueeze(0))
            # Shape: [QSTR_len, emb_dim]
            qstr_encoded = qstr_encoded_ex.squeeze(0)
            # Concatenating first and last time step of encoded qstr
            # Shape: (2 * Qd)
            qstr_repr = torch.cat([qstr_encoded[0].unsqueeze(0), qstr_encoded[-1].unsqueeze(0)], 1).squeeze(0)
            self.qstr2repr[qstr] = qstr_repr






    def set_executor_parameters(self, executor_parameters: ExecutorParameters):
        self._executor_parameters: ExecutorParameters = executor_parameters


    def is_argument_grounded(self, arg):
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

        denotation = self._handle_expression(expression_as_list)

        outer_most_function = expression_as_list[0]
        denotation_type = self.func2returntype_mappings[outer_most_function]

        return denotation, denotation_type


    def _handle_expression(self, expression_list):
        assert isinstance(expression_list, list), f"Expression list is not a list: {expression_list}"

        assert expression_list[0] in self.function_mappings, f"Expression_list[0] not implemented: {expression_list[0]}"

        args = []

        function_name = expression_list[0]

        for item in expression_list[1:]:
            if self.is_argument_grounded(item):
                args.append(item)
            elif self._is_expr_list(item):
                args.append(self._handle_expression(item))
            elif (not isinstance(item, list)) and (item in self.function_mappings):
                # Function with zero arguments
                args.append(self.function_mappings[item]())
            else:
                print(f"Why am I here. Item: {item}")

        return self.function_mappings[function_name](*args)

    def convert_float_to_tensor(self, val):
        if val == 1.0:
            val = 0.9
        else:
            val = 0.1
        return torch.Tensor([val]).cuda()

    ''' Predicates' functional definitions '''

    def bool_qent_qstr(self, qent: str, qstr: str):
        # Get Q_ent string, and Q_str and map to boolean.

        entity_grounding_idx = self.q_nemenspan2entidx[qent]

        # Shape: (2*D)
        qstr_repr = self.qstr2repr[qstr]

        # qstr_span = self.q_qstr_spans[self.q_qstr2idx[qstr]]
        # # [QSTR_len, emb_dim]
        # qstr_embedded = self.ques_embedded[qstr_span[0]:qstr_span[1]+1]
        # qstr_mask = self.ques_mask[qstr_span[0]:qstr_span[1]+1]
        # # [1, QSTR_len, emb_dim]
        # qstr_encoded_ex = self._executor_parameters._ques_encoder(qstr_embedded.unsqueeze(0), qstr_mask.unsqueeze(0))
        # # Shape: [QSTR_len, emb_dim]
        # qstr_encoded = qstr_encoded_ex.squeeze(0)
        # # Concatenating first and last time step of encoded qstr
        # # Shape: (2 * Qd)
        # qstr_repr = torch.cat([qstr_encoded[0].unsqueeze(0), qstr_encoded[-1].unsqueeze(0)], 1).squeeze(0)

        # Shape: (C, M, 2)
        qent_mens = self.ne_ent_mens[entity_grounding_idx]
        qent_mens_mask = (qent_mens[..., 0] >= 0).squeeze(-1).long()
        # Shape: (C, M, 2*D)
        qent_men_repr = self._executor_parameters._span_extractor(self.contexts, qent_mens,
                                                                  self.contexts_mask, qent_mens_mask)
        qstr_repr_ex = qstr_repr.unsqueeze(0).unsqueeze(1)
        scores = torch.sum(qstr_repr_ex * qent_men_repr, 2)
        probs = torch.sigmoid(scores) * qent_mens_mask.float()

        boolean_prob = torch.max(probs) #.unsqueeze(0)

        return boolean_prob


    def bool_and(self, arg1, arg2):
        # AND operation between two booleans
        return arg1 * arg2


    def bool_or(self, arg1, arg2):
        # AND operation between two booleans
        return torch.max(torch.cat([arg1.unsqueeze(0), arg2.unsqueeze(0)], 0))
        # return self.convert_float_to_tensor(1.0)
