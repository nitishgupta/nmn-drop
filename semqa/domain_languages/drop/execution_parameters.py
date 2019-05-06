from typing import Dict, Optional, List, Any

import copy

import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder

from allennlp.modules.attention import Attention, DotProductAttention
from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.modules.matrix_attention import (MatrixAttention, BilinearMatrixAttention,
                                               DotProductMatrixAttention, LinearMatrixAttention)

class ExecutorParameters(torch.nn.Module, Registrable):
    """
        Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 question_encoding_dim: int,
                 passage_encoding_dim: int,
                 passage_attention_to_span: Seq2SeqEncoder,
                 question_attention_to_span: Seq2SeqEncoder,
                 passage_attention_to_count: Seq2VecEncoder,
                 passage_count_predictor=None,
                 dropout: float = 0.0):
        super().__init__()

        self.num_counts = 10

        self.passage_attention_scalingvals = [1, 2, 5, 10]

        self.passage_attention_to_span = passage_attention_to_span
        self.passage_startend_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(), 2)

        self.question_attention_to_span = question_attention_to_span
        self.question_startend_predictor = torch.nn.Linear(self.question_attention_to_span.get_output_dim(), 2)

        self.passage_attention_to_count = passage_attention_to_count
        # self.passage_count_predictor = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
        #                                                self.num_counts)
        self.passage_count_predictor = passage_count_predictor

        self.dotprod_matrix_attn = DotProductMatrixAttention()

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_date_attention: MatrixAttention = BilinearMatrixAttention(matrix_1_dim=passage_encoding_dim,
                                                                                  matrix_2_dim=passage_encoding_dim)
        # self.passage_to_date_attention: MatrixAttention = LinearMatrixAttention(tensor_1_dim=passage_encoding_dim,
        #                                                                         tensor_2_dim=passage_encoding_dim,
        #                                                                         combination='x,y,x*y')

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_num_attention: MatrixAttention = BilinearMatrixAttention(matrix_1_dim=passage_encoding_dim,
                                                                                 matrix_2_dim=passage_encoding_dim)
        # self.passage_to_num_attention: MatrixAttention = LinearMatrixAttention(tensor_1_dim=passage_encoding_dim,
        #                                                                        tensor_2_dim=passage_encoding_dim,
        #                                                                        combination='x,y,x*y')



        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x


# Deprecated
"""
@predicate_with_side_args(['question_attention', 'passage_attention'])
def find_PassageAttention(self, question_attention: Tensor,
                          passage_attention: Tensor = None) -> PassageAttention:

    # The passage attention is only used as supervision for certain synthetic questions
    if passage_attention is not None:
        return PassageAttention(passage_attention, debug_value="Supervised-Pattn-Used")

    question_attention = question_attention * self.question_mask
    # Shape: (encoding_size)
    weighted_question_vector = torch.sum(self.encoded_question * question_attention.unsqueeze(1), 0)

    # Shape: (passage_length, encoded_dim)
    passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage

    # Shape: (1, 1, 10)
    passage_logits_unsqueezed = self.parameters.q2p_matrix_attention(
                                                            weighted_question_vector.unsqueeze(0).unsqueeze(1),
                                                            passage_repr.unsqueeze(0))

    passage_logits = passage_logits_unsqueezed.squeeze(0).squeeze(0)

    passage_attention = allenutil.masked_softmax(passage_logits, mask=self.passage_mask, memory_efficient=True)

    passage_attention = passage_attention * self.passage_mask

    debug_value = ""
    if self._debug:
        qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(question_attention, self.metadata["question_tokens"])
        debug_value += f"Qattn: {qattn_vis_complete}"
        pattn_vis_complete, pattn_vis_most = dlutils.listTokensVis(passage_attention, self.metadata["passage_tokens"])
        debug_value += f"\nPattn: {pattn_vis_complete}"

    return PassageAttention(passage_attention, debug_value=debug_value)
"""

"""
# self.q2p_matrix_attention = LinearMatrixAttention(tensor_1_dim=question_encoding_dim,
#                                                   tensor_2_dim=passage_encoding_dim,
#                                                   combination="x,y,x*y")

self.q2p_matrix_attention = BilinearMatrixAttention(matrix_1_dim=question_encoding_dim,
                                                    matrix_2_dim=passage_encoding_dim)
"""