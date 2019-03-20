from typing import Dict, Optional, List, Any

import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder, TimeDistributed, FeedForward
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.attention import Attention, DotProductAttention
from allennlp.modules.matrix_attention import MatrixAttention, BilinearMatrixAttention, DotProductMatrixAttention

from allennlp.models.decomposable_attention import DecomposableAttention
from allennlp.nn.util import masked_softmax, weighted_sum
from semqa.domain_languages.hotpotqa.decompatt import DecompAtt

class ExecutorParameters(torch.nn.Module, Registrable):
    """
        Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 hidden_dim: int):
        super().__init__()

        encoding_in_dim = phrase_layer.get_input_dim()
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)
        self._phrase_layer: Seq2SeqEncoder = phrase_layer
        self._matrix_attention: MatrixAttention = matrix_attention_layer
        self._modeling_layer: Seq2SeqEncoder = modeling_layer

        passage_encoding_dim = self._phrase_layer.get_output_dim()
        question_encoding_dim = self._phrase_layer.get_output_dim()


        self.find_attention: Attention = DotProductAttention()

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_date_attention: MatrixAttention = BilinearMatrixAttention(matrix_1_dim=passage_encoding_dim,
                                                                                  matrix_2_dim=passage_encoding_dim)

        self.relocate_linear1 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.relocate_linear2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear3 = torch.nn.Linear(passage_encoding_dim, 1)
        self.relocate_linear4 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.passage_span_start_predictor = FeedForward(passage_encoding_dim * 2,
                                                        activations=[Activation.by_name('relu')(),
                                                                     Activation.by_name('linear')()],
                                                        hidden_dims=[passage_encoding_dim, 1],
                                                        num_layers=2)

        self.passage_span_end_predictor = FeedForward(passage_encoding_dim * 2,
                                                      activations=[Activation.by_name('relu')(),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[passage_encoding_dim, 1],
                                                      num_layers=2)
        self.question_span_start_predictor = FeedForward(question_encoding_dim * 2,
                                                         activations=[Activation.by_name('relu')(),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[passage_encoding_dim, 1],
                                                         num_layers=2)
        self.question_span_end_predictor = FeedForward(question_encoding_dim * 2,
                                                       activations=[Activation.by_name('relu')(),
                                                                    Activation.by_name('linear')()],
                                                       hidden_dims=[passage_encoding_dim, 1],
                                                       num_layers=2)

