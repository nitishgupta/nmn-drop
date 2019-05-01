import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import copy

from overrides import overrides

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

from allennlp.training.metrics import Average
import datasets.drop.constants as dropconstants
import utils.util as myutils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("drop_pattn2count")
class PassageAttnToCount(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 passage_attention_to_count: Seq2VecEncoder,
                 dropout: float = 0.2,
                 initializers: InitializerApplicator = InitializerApplicator()) -> None:

        super(PassageAttnToCount, self).__init__(vocab=vocab)

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
                passage_attention: torch.Tensor,
                passage_lengths: List[int],
                count_mask: torch.Tensor,
                answer_as_count: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, max_passage_length = passage_attention.size()
        passage_mask = passage_attention.new_zeros(batch_size, max_passage_length)
        for i, passage_length in enumerate(passage_lengths):
            passage_mask[i, 0:passage_length] = 1.0

        # (B, num_counts)
        answer_as_count = answer_as_count.float()
        count_mask = count_mask.float()

        # List of (B, P) shaped tensors
        scaled_attentions = [passage_attention * sf for sf in self.scaling_vals]
        # Shape: (B, passage_length, num_scaling_factors)
        scaled_passage_attentions = torch.stack(scaled_attentions, dim=2)

        # Shape: (B, hidden_dim)
        count_hidden_repr = self.passage_attention_to_count(scaled_passage_attentions,
                                                            passage_mask)

        # Shape: (B, num_counts)
        passage_span_logits = self.passage_count_predictor(count_hidden_repr)
        count_distribution = torch.softmax(passage_span_logits, dim=1)

        # Loss computation
        output_dict = {}
        log_likelihood = 0.0

        num_masked_instances = torch.sum(count_mask)

        if answer_as_count is not None:

            count_log_probs = torch.log(count_distribution + 1e-40)
            log_likelihood = torch.sum(count_log_probs * answer_as_count * count_mask.unsqueeze(1))

            # List of predicted count idxs, Shape: (B,)
            count_idx = torch.argmax(count_distribution, 1)
            gold_count_idxs = torch.argmax(answer_as_count, 1)
            correct_vec = ((count_idx == gold_count_idxs).float() * count_mask)
            if num_masked_instances > 0:
                correct_perc = torch.sum(correct_vec) / num_masked_instances
            else:
                correct_perc = torch.sum(correct_vec)
            self.count_acc(correct_perc.item())

        loss = -1.0 * log_likelihood

        batch_loss = loss / batch_size
        output_dict["loss"] = batch_loss

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        count_acc = self.count_acc.get_metric(reset)
        metric_dict.update({'acc': count_acc})

        return metric_dict


