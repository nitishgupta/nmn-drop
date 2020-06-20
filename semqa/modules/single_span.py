from typing import Dict, List, Tuple

import torch
import allennlp.nn.util as allenutil
from semqa.modules.span_answer import SpanAnswer
from semqa.utils.rc_utils import get_best_span


class SingleSpanAnswer(SpanAnswer):
    def __init__(self):
        super().__init__()


    def gold_log_marginal_likelihood(self,
            **kwargs,
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

        answer_as_spans: torch.LongTensor = kwargs["answer_as_spans"]
        span_start_log_probs: torch.FloatTensor = kwargs["span_start_log_probs"]
        span_end_log_probs: torch.FloatTensor = kwargs["span_end_log_probs"]

        # Unsqueezing dim=0 to make a batch_size of 1
        answer_as_spans = answer_as_spans.unsqueeze(0)

        span_start_log_probs = span_start_log_probs.unsqueeze(0)
        span_end_log_probs = span_end_log_probs.unsqueeze(0)

        # (batch_size, number_of_ans_spans)
        gold_passage_span_starts = answer_as_spans[:, :, 0]
        gold_passage_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        gold_passage_span_mask_bool: torch.BoolTensor = gold_passage_span_mask.bool()
        clamped_gold_passage_span_starts = allenutil.replace_masked_values(
            gold_passage_span_starts, gold_passage_span_mask_bool, 0
        )
        clamped_gold_passage_span_ends = allenutil.replace_masked_values(
            gold_passage_span_ends, gold_passage_span_mask_bool, 0
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = torch.gather(span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_span_ends = torch.gather(span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = allenutil.replace_masked_values(
            log_likelihood_for_spans, gold_passage_span_mask_bool, -1e7
        )
        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = allenutil.logsumexp(log_likelihood_for_spans)

        # Squeezing the batch-size 1
        log_marginal_likelihood_for_span = log_marginal_likelihood_for_span.squeeze(-1)

        return log_marginal_likelihood_for_span

    def decode_answer(self, **kwargs):
        # Shape: (passage_length)
        span_start_logits: torch.Tensor = kwargs["span_start_logits"]
        span_end_logits: torch.Tensor = kwargs["span_end_logits"]
        passage_token_offsets: List[Tuple[int, int]] = kwargs["passage_token_offsets"]
        passage_text: List[Tuple[int, int]] = kwargs["passage_text"]

        best_span = get_best_span(
            span_start_logits=span_start_logits.unsqueeze(0),
            span_end_logits=span_end_logits.unsqueeze(0),
        ).squeeze(0)

        predicted_span = tuple(best_span.detach().cpu().numpy())
        if predicted_span[0] >= len(passage_token_offsets) or predicted_span[1] >= len(passage_token_offsets):
            import pdb
            pdb.set_trace()
        start_char_offset = passage_token_offsets[predicted_span[0]][0]
        end_char_offset = passage_token_offsets[predicted_span[1]][1]
        predicted_answer = passage_text[start_char_offset:end_char_offset]

        return predicted_answer