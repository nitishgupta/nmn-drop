import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor


class ExecutorParameters(torch.nn.Module, Registrable):
    """
    Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 ques_encoder: Seq2SeqEncoder = None,
                 context_encoder: Seq2SeqEncoder = None,
                 bool_bilinear: SimilarityFunction = None,
                 dropout: float = 0.0):
        super(ExecutorParameters, self).__init__()
        # self._ques_encoder = ques_encoder
        # self._context_encoder = context_encoder
        # TODO(nitish): Figure out a way to pass this programatically from bidaf
        self._span_extractor = EndpointSpanExtractor(input_dim=200)
        self._bool_bilinear = bool_bilinear

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x



        # Set this in the model init -- same as the model's text_field_embedder
        self._text_field_embedder: TextFieldEmbedder = None