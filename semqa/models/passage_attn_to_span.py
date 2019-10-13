import logging
from typing import List, Dict, Any, Tuple

from overrides import overrides

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator
from allennlp.models.reading_comprehension.util import get_best_span

from allennlp.training.metrics import Average


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("drop_pattn2span")
class PassageAttnToSpan(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        passage_attention_to_span: Seq2SeqEncoder,
        scaling: bool = False,
        dropout: float = 0.0,
        initializers: InitializerApplicator = InitializerApplicator(),
    ) -> None:

        super(PassageAttnToSpan, self).__init__(vocab=vocab)

        self._scaling = scaling
        self.scaling_vals = [1, 2, 5, 10]

        self._passage_attention_to_span = passage_attention_to_span

        if self._scaling:
            assert len(self.scaling_vals) == self._passage_attention_to_span.get_input_dim()

        if self._passage_attention_to_span.get_output_dim() > 1:
            self._mapping = torch.nn.Linear(self._passage_attention_to_span.get_output_dim(), 2)
        else:
            self._mapping = None

        self._span_rnn_hsize = self._passage_attention_to_span.get_output_dim()

        self.start_ff = torch.nn.Linear(self._passage_attention_to_span.get_output_dim(), 1)
        self.end_ff = torch.nn.Linear(self._passage_attention_to_span.get_output_dim(), 1)

        self.start_acc_metric = Average()
        self.end_acc_metric = Average()
        self.span_acc_metric = Average()

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
        answer_as_passage_spans: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, max_passage_length = passage_attention.size()
        passage_mask = passage_attention.new_zeros(batch_size, max_passage_length)
        for i, passage_length in enumerate(passage_lengths):
            passage_mask[i, 0:passage_length] = 1.0

        answer_as_passage_spans = answer_as_passage_spans.long()

        passage_attention = passage_attention * passage_mask

        if self._scaling:
            scaled_attentions = [passage_attention * sf for sf in self.scaling_vals]
            passage_attention_input = torch.stack(scaled_attentions, dim=2)
        else:
            passage_attention_input = passage_attention.unsqueeze(2)

        # Shape: (batch_size, passage_length, span_rnn_hsize)
        passage_span_logits_repr = self._passage_attention_to_span(passage_attention_input, passage_mask)

        if self._mapping:
            passage_span_logits_repr = self._mapping(passage_span_logits_repr)

        # Shape: (batch_size, passage_length)
        span_start_logits = passage_span_logits_repr[:, :, 0]
        span_end_logits = passage_span_logits_repr[:, :, 1]

        # span_start_logits = self.start_ff(passage_span_logits_repr).squeeze(2)
        # span_end_logits = self.end_ff(passage_span_logits_repr).squeeze(2)

        span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask, -1e32)

        span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, passage_mask)
        span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, passage_mask)

        span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs, passage_mask, -1e32)
        span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs, passage_mask, -1e32)

        # Loss computation
        batch_likelihood = 0
        output_dict = {}
        for i in range(batch_size):
            log_likelihood = self._get_span_answer_log_prob(
                answer_as_spans=answer_as_passage_spans[i],
                span_log_probs=(span_start_log_probs[i], span_end_log_probs[i]),
            )

            best_span = get_best_span(
                span_start_logits=span_start_log_probs[i].unsqueeze(0),
                span_end_logits=span_end_log_probs[i].unsqueeze(0),
            ).squeeze(0)

            correct_start, correct_end = False, False

            if best_span[0] == answer_as_passage_spans[i][0][0]:
                self.start_acc_metric(1)
                correct_start = True
            else:
                self.start_acc_metric(0)

            if best_span[1] == answer_as_passage_spans[i][0][1]:
                self.end_acc_metric(1)
                correct_end = True
            else:
                self.end_acc_metric(0)

            if correct_start and correct_end:
                self.span_acc_metric(1)
            else:
                self.span_acc_metric(0)

            batch_likelihood += log_likelihood

        loss = -1.0 * batch_likelihood

        batch_loss = loss / batch_size
        output_dict["loss"] = batch_loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        start_acc = self.start_acc_metric.get_metric(reset)
        end_acc = self.end_acc_metric.get_metric(reset)
        span_acc = self.span_acc_metric.get_metric(reset)
        metric_dict.update({"st": start_acc, "end": end_acc, "span": span_acc})

        return metric_dict

    @staticmethod
    def _get_span_answer_log_prob(
        answer_as_spans: torch.LongTensor, span_log_probs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """ Compute the log_marginal_likelihood for the answer_spans given log_probs for start/end
            Compute log_likelihood (product of start/end probs) of each ans_span
            Sum the prob (logsumexp) for each span and return the log_likelihood

        Parameters:
        -----------
        answer: ``torch.LongTensor`` Shape: (number_of_spans, 2)
            These are the gold spans
        span_log_probs: ``torch.FloatTensor``
            2-Tuple with tensors of Shape: (length_of_sequence) for span_start/span_end log_probs

        Returns:
        log_marginal_likelihood_for_passage_span
        """

        # Unsqueezing dim=0 to make a batch_size of 1
        answer_as_spans = answer_as_spans.unsqueeze(0)

        span_start_log_probs, span_end_log_probs = span_log_probs
        span_start_log_probs = span_start_log_probs.unsqueeze(0)
        span_end_log_probs = span_end_log_probs.unsqueeze(0)

        # (batch_size, number_of_ans_spans)
        gold_passage_span_starts = answer_as_spans[:, :, 0]
        gold_passage_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = allenutil.replace_masked_values(
            gold_passage_span_starts, gold_passage_span_mask, 0
        )
        clamped_gold_passage_span_ends = allenutil.replace_masked_values(
            gold_passage_span_ends, gold_passage_span_mask, 0
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = torch.gather(span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_span_ends = torch.gather(span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = allenutil.replace_masked_values(
            log_likelihood_for_spans, gold_passage_span_mask, -1e7
        )
        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = allenutil.logsumexp(log_likelihood_for_spans)

        return log_marginal_likelihood_for_span

    @staticmethod
    def _get_best_spans(
        batch_denotations,
        batch_denotation_types,
        question_char_offsets,
        question_strs,
        passage_char_offsets,
        passage_strs,
        *args,
    ):
        """ For all SpanType denotations, get the best span

        Parameters:
        ----------
        batch_denotations: List[List[Any]]
        batch_denotation_types: List[List[str]]
        """

        (question_num_tokens, passage_num_tokens, question_mask_aslist, passage_mask_aslist) = args

        batch_best_spans = []
        batch_predicted_answers = []

        for instance_idx in range(len(batch_denotations)):
            instance_prog_denotations = batch_denotations[instance_idx]
            instance_prog_types = batch_denotation_types[instance_idx]

            instance_best_spans = []
            instance_predicted_ans = []

            for denotation, progtype in zip(instance_prog_denotations, instance_prog_types):
                # if progtype == "QuestionSpanAnswwer":
                # Distinction between QuestionSpanAnswer and PassageSpanAnswer is not needed currently,
                # since both classes store the start/end logits as a tuple
                # Shape: (2, )
                best_span = get_best_span(
                    span_start_logits=denotation._value[0].unsqueeze(0),
                    span_end_logits=denotation._value[1].unsqueeze(0),
                ).squeeze(0)
                instance_best_spans.append(best_span)

                predicted_span = tuple(best_span.detach().cpu().numpy())
                if progtype == "QuestionSpanAnswer":
                    try:
                        start_offset = question_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = question_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = question_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"Question numtoksn: {question_num_tokens[instance_idx]}")
                        print(f"QuesMaskLen: {question_mask_aslist[instance_idx].size()}")
                        print(f"StartLogProbs:{denotation._value[0]}")
                        print(f"EndLogProbs:{denotation._value[1]}")
                        print(f"LenofOffsets: {len(question_char_offsets[instance_idx])}")
                        print(f"QuesStrLen: {len(question_strs[instance_idx])}")

                elif progtype == "PassageSpanAnswer":
                    try:
                        start_offset = passage_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = passage_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = passage_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"Passagenumtoksn: {passage_num_tokens[instance_idx]}")
                        print(f"PassageMaskLen: {passage_mask_aslist[instance_idx].size()}")
                        print(f"LenofOffsets: {len(passage_char_offsets[instance_idx])}")
                        print(f"PassageStrLen: {len(passage_strs[instance_idx])}")
                else:
                    raise NotImplementedError

                instance_predicted_ans.append(predicted_answer)

            batch_best_spans.append(instance_best_spans)
            batch_predicted_answers.append(instance_predicted_ans)

        return batch_best_spans, batch_predicted_answers

    def passageattn_to_startendlogits(self, passage_attention, passage_mask):
        span_start_logits = passage_attention.new_zeros(passage_attention.size())
        span_end_logits = passage_attention.new_zeros(passage_attention.size())

        nonzeroindcs = (passage_attention > 0).nonzero()

        startidx = nonzeroindcs[0]
        endidx = nonzeroindcs[-1]

        print(f"{startidx} {endidx}")

        span_start_logits[startidx] = 2.0
        span_end_logits[endidx] = 2.0

        span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask, -1e32)

        span_start_logits += 1e-7
        span_end_logits += 1e-7

        return (span_start_logits, span_end_logits)
