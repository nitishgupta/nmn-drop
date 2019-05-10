import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import copy
import numpy as np
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from allennlp.training.metrics import Average
import datasets.drop.constants as dropconstants
import utils.util as myutils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("drop_pattn2count")
class PassageAttnToCount(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 passage_attention_to_count: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 initializers: InitializerApplicator = InitializerApplicator()) -> None:

        super(PassageAttnToCount, self).__init__(vocab=vocab)

        self.scaling_vals = [1, 2, 5, 10]

        self.passage_attention_to_count = passage_attention_to_count

        assert len(self.scaling_vals) == self.passage_attention_to_count.get_input_dim()

        self.num_counts = 10
        # self.passage_count_predictor = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
        #                                                self.num_counts, bias=False)

        # We want to predict a score for each passage token
        self.passage_count_hidden2logits = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
                                                           1, bias=True)

        self.passagelength_to_bias = torch.nn.Linear(1, 1, bias=True)

        self.count_acc = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializers(self)
        # self.passage_count_hidden2logits.bias.data.fill_(-1.0)
        # self.passage_count_hidden2logits.bias.requires_grad = False

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                passage_attention: torch.Tensor,
                passage_lengths: List[int],
                count_answer: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        device_id = allenutil.get_device_of(passage_attention)

        batch_size, max_passage_length = passage_attention.size()

        # Shape: (B, passage_length)
        passage_mask = (passage_attention >= 0).float()

        # List of (B, P) shaped tensors
        scaled_attentions = [passage_attention * sf for sf in self.scaling_vals]
        # Shape: (B, passage_length, num_scaling_factors)
        scaled_passage_attentions = torch.stack(scaled_attentions, dim=2)

        # Shape (batch_size, 1)
        passage_len_bias = self.passagelength_to_bias(passage_mask.sum(1, keepdim=True))

        scaled_passage_attentions = scaled_passage_attentions * passage_mask.unsqueeze(2)

        # Shape: (B, passage_length, hidden_dim)
        count_hidden_repr = self.passage_attention_to_count(scaled_passage_attentions,
                                                            passage_mask)

        # Shape: (B, passage_length, 1) -- score for each token
        passage_span_logits = self.passage_count_hidden2logits(count_hidden_repr)
        # Shape: (B, passage_length) -- sigmoid on token-score
        token_sigmoids = torch.sigmoid(passage_span_logits.squeeze(2))
        token_sigmoids = token_sigmoids * passage_mask

        # Shape: (B, 1) -- sum of sigmoids. This will act as the predicted mean
        # passage_count_mean = torch.sum(token_sigmoids, dim=1, keepdim=True) + passage_len_bias
        passage_count_mean = torch.sum(token_sigmoids, dim=1, keepdim=True)

        # Shape: (1, count_vals)
        self.countvals = allenutil.get_range_vector(10, device=device_id).unsqueeze(0).float()

        variance = 0.2

        # Shape: (batch_size, count_vals)
        l2_by_vsquared = torch.pow(self.countvals - passage_count_mean, 2) / (2 * variance * variance)
        exp_val = torch.exp(-1 * l2_by_vsquared) + 1e-30
        # Shape: (batch_size, count_vals)
        count_distribution = exp_val / (torch.sum(exp_val, 1, keepdim=True))

        # Loss computation
        output_dict = {}
        loss = 0.0
        pred_count_idx = torch.argmax(count_distribution, 1)
        if count_answer is not None:
            # L2-loss
            passage_count_mean = passage_count_mean.squeeze(1)
            L2Loss = F.mse_loss(input=passage_count_mean, target=count_answer.float())
            loss = L2Loss
            predictions = passage_count_mean.detach().cpu().numpy()
            predictions = np.round_(predictions)

            gold_count = count_answer.detach().cpu().numpy()
            correct_vec = (predictions == gold_count)
            correct_perc = sum(correct_vec)/batch_size
            # print(f"{correct_perc} {predictions} {gold_count}")
            self.count_acc(correct_perc)

            # loss = F.cross_entropy(input=count_distribution, target=count_answer)
            # List of predicted count idxs, Shape: (B,)
            # correct_vec = (pred_count_idx == count_answer).float()
            # correct_perc = torch.sum(correct_vec) / batch_size
            # self.count_acc(correct_perc.item())

        batch_loss = loss / batch_size
        output_dict["loss"] = batch_loss
        output_dict["passage_attention"] = passage_attention
        output_dict["passage_sigmoid"] = token_sigmoids
        output_dict["count_mean"] = passage_count_mean
        output_dict["count_distritbuion"] = count_distribution
        output_dict["count_answer"] = count_answer
        output_dict["pred_count"] = pred_count_idx


        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        count_acc = self.count_acc.get_metric(reset)
        metric_dict.update({'acc': count_acc})

        return metric_dict


