from typing import Dict, Optional, List, Any

import copy

import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder

from allennlp.modules.attention import Attention, DotProductAttention
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

