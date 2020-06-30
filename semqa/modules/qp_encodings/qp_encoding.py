from typing import Dict, Any

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.common import Registrable


class QPEncoding(Registrable):
    def __init__(self, text_field_embedder: TextFieldEmbedder):
        self._text_field_embedder = text_field_embedder

    def get_representation(self,
                           question_passage: TextFieldTensors = None,
                           max_ques_len: int = None,
                           question: TextFieldTensors = None,
                           passage: TextFieldTensors = None):
        raise NotImplementedError
