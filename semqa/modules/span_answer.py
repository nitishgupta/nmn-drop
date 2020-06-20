from typing import Dict, Any

import torch
from allennlp.nn.util import logsumexp, masked_log_softmax, replace_masked_values, get_range_vector, get_device_of
from allennlp.common import Registrable

class SpanAnswer(Registrable):

    def gold_log_marginal_likelihood(self, **kwargs):
        raise NotImplementedError

    def decode_answer(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def passage_attention_loss(passage_attention: torch.FloatTensor,
                               passage_mask: torch.FloatTensor,
                               answer_spans_for_possible_taggings: torch.LongTensor,
                               device_id: int):
        """Compute loss for encouraging passage-attention to attend to possible answer spans.

        We are provided with weak-supervision for possible answer-spans;
        - different possible taggings are provided
        - for each tagging, a set of spans is provided

        We would marginalize over different tag-probabilities, which are in turn computed as a joint over all spans
        for that tagging.

        Parameters:
        ----------
        passage_attention: (passage_length, )
        passage_mask: (passage_length, )
        answer_spans_for_possible_taggings: (num_taggings, num_spans, 2)
        """
        passage_len = passage_mask.size()
        # device_id = get_device_of(passage_attention)
        # replace masked values with 0+ to compute log
        log_passage_attention = torch.log(passage_attention + 1e-32)
        log_passage_attention = replace_masked_values(log_passage_attention, passage_mask.bool(), -1e32)

        # (num_taggings, num_spans)
        possible_spans_mask = (answer_spans_for_possible_taggings[:, :, 0] >= 0).long()

        answer_spans_for_possible_taggings = answer_spans_for_possible_taggings * possible_spans_mask.unsqueeze(-1)

        # Create a mask of shape (num_taggings, num_spans, passage_length) with 1s where spans are
        range_vector = get_range_vector(passage_len, device_id)

        # (num_taggings, num_spans, passage_length)
        starts = (range_vector.unsqueeze(0).unsqueeze(0) >=
                    answer_spans_for_possible_taggings[:, :, 0].unsqueeze(-1)).long()
        ends = (range_vector.unsqueeze(0).unsqueeze(0) <=
                    answer_spans_for_possible_taggings[:, :, 1].unsqueeze(-1)).long()
        spans_token_bool = starts * ends * possible_spans_mask.unsqueeze(-1)
        # (num_taggings, num_spans, passage_length)
        spans_token_logprob = log_passage_attention * spans_token_bool.float()
        # (num_taggings, num_spans)
        spans_logprob = spans_token_logprob.sum(dim=-1)
        # (num_taggings)
        taggings_logprob = spans_logprob.sum(dim=-1)
        marginal_logprob = logsumexp(taggings_logprob)

        return -1.0 * marginal_logprob













