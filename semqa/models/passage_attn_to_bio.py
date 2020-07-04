import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

from semqa.modules.spans import MultiSpanAnswer

from allennlp.training.metrics import Average


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("drop_pattn2bio")
class PassageAttnToBio(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        passage_attention_to_span: Seq2SeqEncoder,
        bio_label_scheme: str = "IO",
        joint_count: bool = True,
        dropout: float = 0.2,
        initializers: InitializerApplicator = InitializerApplicator(),
    ) -> None:

        super(PassageAttnToBio, self).__init__(vocab=vocab)

        self.joint_count = joint_count

        self.scaling_vals = [1, 2, 5, 10]

        self.passage_attention_to_span = passage_attention_to_span

        assert len(self.scaling_vals) == self.passage_attention_to_span.get_input_dim()

        self.bio_label_scheme = bio_label_scheme
        self.bio_labels = None
        if self.bio_label_scheme == "BIO":
            self.bio_labels = {'O': 0, 'B': 1, 'I': 2}
        elif self.bio_label_scheme == "IO":
            self.bio_labels = {'O': 0, 'I': 1}
        else:
            raise Exception("bio_label_scheme not supported: {}".format(self.bio_label_scheme))

        self.span_answer = MultiSpanAnswer(ignore_question=True,
                                           prediction_method="viterbi",
                                           decoding_style="at_least_one",
                                           training_style="",
                                           labels=self.bio_labels,
                                           empty_decoding=True)

        self.passage_bio_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(),
                                                     len(self.bio_labels))
        self.bio_acc = Average()
        self.total_acc = Average()

        if self.joint_count:
            self.num_counts = 10
            # We want to predict a score for each passage token
            self.passage_count_hidden2logits = torch.nn.Linear(
                self.passage_attention_to_span.get_output_dim(), 1, bias=True
            )
            self.variance = 0.2
            self.count_acc = Average()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializers(self)

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(
        self,
        passage_attention: torch.Tensor,
        passage_lengths: List[int],
        all_taggings_spans: List[List[List[Tuple[int, int]]]],
        bios_label_field: torch.LongTensor,   # (batch_size, num_tag_seqs, passage_length)
        bios_label_mask: torch.LongTensor,    # (batch_size, num_tag_seqs)
        gold_spans: List[List[Tuple[int, int]]],
        count_answer: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        passage_attention: `torch.FloatTensor`
            (batch_size, passage_length) -- passage attention for a gold-span tagging
        passage_lengths: `List[int]`
            True length of passages without masking
        all_taggings_spans: `List[List[List[Tuple[int, int]]]]`
            For each instance, List of all possible tagging combinations, each taggins is a List[Span]
        bios_label_field: `torch.LongTensor` (batch_size, num_tag_seqs, passage_length)
            All possible taggings as a BIO tag-index tensor. This will be used as weak supervision for marginalization.
            Equivalent to `passage_span_answer` in drop_parser_bert.py`
        bios_label_mask: `torch.LongTensor` (batch_size, num_tag_seqs)
            Mask for BIO seqs above since different instances would have different number of gold-bio tag-sequences
        gold_spans: `List[List[Tuple[int, int]]]`
            For each instance, the gold-spans for which we provide passage attention. This is one of the possible
            tagging amongst all possible. This will be used to measure accuracy.
        metadata: `List[Dict[str, Any]]`
        """

        device_id = allenutil.get_device_of(passage_attention)

        batch_size, max_passage_length = passage_attention.size()

        # Shape: (B, passage_length)
        passage_mask = (passage_attention >= 0).float()

        # List of (B, P) shaped tensors
        scaled_attentions = [passage_attention * sf for sf in self.scaling_vals]
        # Shape: (B, passage_length, num_scaling_factors)
        scaled_passage_attentions = torch.stack(scaled_attentions, dim=2)

        scaled_passage_attentions = scaled_passage_attentions * passage_mask.unsqueeze(2)

        # Shape: (B, passage_length, hidden_dim)
        pattn_hidden_repr = self.passage_attention_to_span(scaled_passage_attentions, passage_mask)
        # Shape: (B, passage_length, num_tags)
        passage_bio_logits = self.passage_bio_predictor(pattn_hidden_repr)

        # (batch_size, pasage_len, num_bio_tags)
        passage_bio_logprobs = allenutil.masked_log_softmax(passage_bio_logits, dim=-1, mask=None)

        bio_loss = 0.0
        batch_predicted_spans = []
        batch_bio_correct = []

        for i in range(0, batch_size):
            marginal_ll = self.span_answer.gold_log_marginal_likelihood(passage_span_answer=bios_label_field[i],
                                                                        passage_span_answer_mask=bios_label_mask[i],
                                                                        log_probs=passage_bio_logprobs[i],
                                                                        passage_mask=passage_mask[i])

            bio_loss += -1.0 * marginal_ll

            if not self.training:
                if bios_label_mask[i].sum() > 0:   # Non-masked BIO instance;
                    masked_predicted_tags = self.span_answer.get_predicted_tags(log_probs=passage_bio_logprobs[i],
                                                                                passage_mask=passage_mask[i])

                    predicted_token_tags = masked_predicted_tags.cpu().tolist()

                    predicted_spans: List[Tuple[int, int]] = self.span_answer.convert_tags_to_spans(predicted_token_tags)

                    instance_gold_spans = gold_spans[i]
                    instance_gold_spans_set = set(instance_gold_spans)
                    predicted_spans_set = set(predicted_spans)
                    correct = int(predicted_spans_set == instance_gold_spans_set)
                    batch_bio_correct.append(correct)
                    self.bio_acc(correct)
                    if not self.joint_count:
                        self.total_acc(correct)
                else:
                    # synthetic count instance w/o BIO supervision
                    predicted_spans = []
                    batch_bio_correct.append(0)
                batch_predicted_spans.append(predicted_spans)

        loss = bio_loss / batch_size
        output_dict = {}
        output_dict["passage_attention"] = passage_attention
        output_dict["predicted_spans"] = batch_predicted_spans
        output_dict["gold_spans"] = gold_spans
        output_dict["gold_count"] = count_answer

        """ Count value """
        if self.joint_count:
            # Shape: (B, passage_length, 1) -- score for each token
            passage_span_logits = self.passage_count_hidden2logits(pattn_hidden_repr)
            # Shape: (B, passage_length) -- sigmoid on token-score
            token_sigmoids = torch.sigmoid(passage_span_logits.squeeze(2))
            token_sigmoids = token_sigmoids * passage_mask
            # Shape: (B, 1) -- sum of sigmoids. This will act as the predicted mean
            # passage_count_mean = torch.sum(token_sigmoids, dim=1, keepdim=True) + passage_len_bias
            passage_count_mean = torch.sum(token_sigmoids, dim=1, keepdim=True)
            # Shape: (1, count_vals)
            self.countvals = allenutil.get_range_vector(10, device=device_id).unsqueeze(0).float()
            # Shape: (batch_size, count_vals)
            l2_by_vsquared = torch.pow(self.countvals - passage_count_mean, 2) / (2 * self.variance * self.variance)
            exp_val = torch.exp(-1 * l2_by_vsquared) + 1e-30
            # Shape: (batch_size, count_vals)
            count_distribution = exp_val / (torch.sum(exp_val, 1, keepdim=True))
            """ /end count value """

            # Count Loss and prediction computation
            pred_count_idx = torch.argmax(count_distribution, 1)
            # L2-loss
            passage_count_mean = passage_count_mean.squeeze(1)
            count_loss = F.mse_loss(input=passage_count_mean, target=count_answer.float())
            count_predictions = pred_count_idx.detach().cpu().numpy()
            gold_counts = count_answer.detach().cpu().numpy()
            batch_count_correct = np.array(count_predictions == gold_counts, dtype=np.int32).tolist()

            for bio_correct, count_correct in zip(batch_bio_correct, batch_count_correct):
                self.count_acc(count_correct)
                self.total_acc(bio_correct & count_correct)
            loss += count_loss
            output_dict["predicted_count"] = count_predictions
            output_dict["count_mean"] = passage_count_mean
            output_dict["count_distribution"] = count_distribution

        output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        bio_acc = self.bio_acc.get_metric(reset)
        total_acc = self.total_acc.get_metric(reset)
        metric_dict.update({"total_acc": total_acc})
        metric_dict.update({"bio_acc": bio_acc})
        if self.joint_count:
            count_acc = self.count_acc.get_metric(reset)
            metric_dict.update({"count_acc": count_acc})


        return metric_dict
