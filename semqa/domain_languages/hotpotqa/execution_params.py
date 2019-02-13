import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention


class ExecutorParameters(torch.nn.Module, Registrable):
    """
    Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 bool_bilinear: SimilarityFunction = None,
                 bidafmodel: BidirectionalAttentionFlow = None,
                 dropout: float = 0.0):
        super(ExecutorParameters, self).__init__()
        # TODO(nitish): Figure out a way to pass this programatically from bidaf
        self._span_extractor = EndpointSpanExtractor(input_dim=200)
        self._bool_bilinear = bool_bilinear
        self._bidafmodel = bidafmodel

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        # Set this in the model init -- same as the model's text_field_embedder
        self._text_field_embedder: TextFieldEmbedder = None