from typing import Dict, Any

import torch

from allennlp.common import Registrable

class SpanAnswer(Registrable):

    def gold_log_marginal_likelihood(self, **kwargs):
        raise NotImplementedError

    def decode_answer(self, **kwargs):
        raise NotImplementedError