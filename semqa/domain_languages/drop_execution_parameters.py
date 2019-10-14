from typing import Dict, Optional, List, Any

import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Seq2SeqEncoder

from allennlp.modules.matrix_attention import (
    MatrixAttention,
    BilinearMatrixAttention,
    DotProductMatrixAttention,
    LinearMatrixAttention,
)


class ExecutorParameters(torch.nn.Module, Registrable):
    """
        Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """

    def __init__(
        self,
        question_encoding_dim: int,
        passage_encoding_dim: int,
        passage_attention_to_span: Seq2SeqEncoder,
        question_attention_to_span: Seq2SeqEncoder,
        passage_attention_to_count: Seq2SeqEncoder,
        num_implicit_nums: int = None,
        passage_count_predictor=None,
        passage_count_hidden2logits=None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_counts = 10

        self.passage_attention_scalingvals = [1, 2, 5, 10]

        # Parameters for answer start/end prediction from PassageAttention
        self.passage_attention_to_span = passage_attention_to_span
        self.passage_startend_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(), 2)

        # Parameters for answer start/end pred directly from passage encoding (direct PassageSpanAnswer from 1step prog)
        self.oneshot_psa_startend_predictor = torch.nn.Linear(passage_encoding_dim, 2)

        self.question_attention_to_span = question_attention_to_span
        self.question_startend_predictor = torch.nn.Linear(self.question_attention_to_span.get_output_dim(), 2)

        self.passage_attention_to_count = passage_attention_to_count
        # self.passage_count_predictor = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
        #                                                self.num_counts)
        self.passage_count_predictor = passage_count_predictor
        # Linear from self.passage_attention_to_count.output_dim --> 1
        self.passage_count_hidden2logits = passage_count_hidden2logits

        self.dotprod_matrix_attn = DotProductMatrixAttention()

        self.implicit_num_embeddings = torch.nn.Parameter(torch.FloatTensor(num_implicit_nums, passage_encoding_dim))
        torch.nn.init.normal_(self.implicit_num_embeddings, mean=0.0, std=0.001)
        self.implicitnum_bilinear_attention = BilinearMatrixAttention(
            matrix_1_dim=passage_encoding_dim, matrix_2_dim=passage_encoding_dim
        )

        self.filter_matrix_attention = LinearMatrixAttention(
            tensor_1_dim=question_encoding_dim, tensor_2_dim=passage_encoding_dim, combination="x,y,x*y"
        )

        # We will sum the passage-token-repr to the weighted-q-repr - to use x*y combination
        self.relocate_matrix_attention = LinearMatrixAttention(
            tensor_1_dim=passage_encoding_dim, tensor_2_dim=passage_encoding_dim, combination="x,y,x*y"
        )

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_date_attention: MatrixAttention = BilinearMatrixAttention(
            matrix_1_dim=passage_encoding_dim, matrix_2_dim=passage_encoding_dim
        )

        self.passage_to_start_date_attention: MatrixAttention = BilinearMatrixAttention(
            matrix_1_dim=passage_encoding_dim, matrix_2_dim=passage_encoding_dim
        )

        self.passage_to_end_date_attention: MatrixAttention = BilinearMatrixAttention(
            matrix_1_dim=passage_encoding_dim, matrix_2_dim=passage_encoding_dim
        )

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_num_attention: MatrixAttention = BilinearMatrixAttention(
            matrix_1_dim=passage_encoding_dim, matrix_2_dim=passage_encoding_dim
        )

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
