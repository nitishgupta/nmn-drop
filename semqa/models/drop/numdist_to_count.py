import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import copy

from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

from allennlp.training.metrics import Average


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("drop_numdist2count")
class NumDistToCount(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 passage_attention_to_count: Seq2VecEncoder,
                 dropout: float = 0.2,
                 initializers: InitializerApplicator = InitializerApplicator()) -> None:

        super(NumDistToCount, self).__init__(vocab=vocab)

        self.scaling_vals = [1, 2, 5, 10]

        self.passage_attention_to_count = passage_attention_to_count

        assert len(self.scaling_vals) == self.passage_attention_to_count.get_input_dim()

        self.num_counts = 10
        self.passage_count_predictor = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
                                                       self.num_counts, bias=False)

        self.count_acc = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializers(self)

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                number_dist: torch.Tensor,
                count_answer: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, max_dist_length = number_dist.size()

        num_dist_mask = (number_dist > -1).float()

        # List of (B, P) shaped tensors
        scaled_attentions = [number_dist * sf for sf in self.scaling_vals]
        # Shape: (B, passage_length, num_scaling_factors)
        scaled_num_dists = torch.stack(scaled_attentions, dim=2)

        # Shape: (B, hidden_dim)
        count_hidden_repr = self.passage_attention_to_count(scaled_num_dists,
                                                            num_dist_mask)

        # Shape: (B, num_counts)
        count_logits = self.passage_count_predictor(count_hidden_repr)
        # count_distribution = torch.softmax(passage_span_logits, dim=1)

        # Loss computation
        output_dict = {}
        log_likelihood = 0.0

        loss = 0.0
        if count_answer is not None:

            loss = F.cross_entropy(input=count_logits, target=count_answer)

            # List of predicted count idxs, Shape: (B,)
            count_idx = torch.argmax(count_logits, 1)
            correct_vec = (count_idx == count_answer).float()
            correct_perc = torch.sum(correct_vec) / batch_size

            self.count_acc(correct_perc.item())

        batch_loss = loss / batch_size
        output_dict["loss"] = batch_loss

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        count_acc = self.count_acc.get_metric(reset)
        metric_dict.update({'acc': count_acc})

        return metric_dict


