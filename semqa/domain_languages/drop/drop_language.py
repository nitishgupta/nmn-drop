from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import LSTM

import allennlp.nn.util as allenutil
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args)

from semqa.domain_languages.drop.execution_parameters import ExecutorParameters

# # An Attention is a single tensor; we're giving this a type so that we can use it for constructing
# # predicates.
# class AttentionTensor(Tensor):
#     pass


class DateDistribution:
    def __init__(self,
                 year_distribution: Tensor,
                 month_distribution: Tensor,
                 day_distribution: Tensor) -> None:
        self.year_distribution = year_distribution
        self.month_distribution = month_distribution
        self.day_distribution = day_distribution


class DateDelta:
    def __init__(self,
                 year_delta: Tensor,
                 month_delta: Tensor,
                 day_delta: Tensor) -> None:
        self.year_delta = year_delta
        self.month_delta = month_delta
        self.day_delta = day_delta


class PassageSpanAnswer():
    def __init__(self,
                 passage_span_start_log_probs: Tensor,
                 passage_span_end_log_probs: Tensor) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self._value = (passage_span_start_log_probs, passage_span_end_log_probs)


class QuestionSpanAnswer():
    def __init__(self,
                 question_span_start_log_probs: Tensor,
                 question_span_end_log_probs: Tensor) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self._value = (question_span_start_log_probs, question_span_end_log_probs)


class QuestionAttention():
    def __init__(self, question_attention):
        self._value = question_attention


class PassageAttention():
    def __init__(self, passage_attention):
        self._value = passage_attention


# A ``NumberAnswer`` is a distribution over the possible answers from addition and subtraction.
class NumberAnswer(Tensor):
    pass


# A ``CountAnswer`` is a distribution over the possible counts in the passage.
class CountAnswer(Tensor):
    pass

class DropLanguage(DomainLanguage):
    """
    DomainLanguage for the DROP dataset based on neural module networks. This language has a `learned execution model`,
    meaning that the predicates in this language have learned parameters.

    Parameters
    ----------
    parameters : ``DropNmnParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    def __init__(self,
                 encoded_question: Tensor,
                 encoded_passage: Tensor,
                 question_mask: Tensor,
                 passage_mask: Tensor,
                 parameters: ExecutorParameters,
                 max_samples=10) -> None:
        super().__init__(start_types={QuestionSpanAnswer, PassageSpanAnswer})
        self.encoded_passage = encoded_passage
        self.encoded_question = encoded_question
        self.question_mask = question_mask
        self.passage_mask = passage_mask

        self.parameters = parameters
        self.max_samples = max_samples

    @predicate_with_side_args(['question_attention'])
    def find_QuestionAttention(self, question_attention: Tensor) -> QuestionAttention:
        return QuestionAttention(question_attention)


    @predicate
    def find_PassageAttention(self, question_attn: QuestionAttention) -> PassageAttention:
        # Shape: (1, D)
        attended_question = allenutil.weighted_sum(self.encoded_question.unsqueeze(0),
                                                   question_attn._value.unsqueeze(0))
        # Shape: (passage_length)
        passage_attention = self.parameters.find_attention(attended_question,
                                                           self.encoded_passage.unsqueeze(0)).squeeze(0)

        return PassageAttention(passage_attention=passage_attention)


    @predicate_with_side_args(['question_attention'])
    def relocate_PassageAttention(self,
                                  passage_attention: PassageAttention,
                                  question_attention: Tensor) -> PassageAttention:
        # Shape: (1, D)
        attended_question = allenutil.weighted_sum(self.encoded_question.unsqueeze(0),
                                                   question_attention.unsqueeze(0))

        # Shape: (encoding_dim, 1)
        attended_passage = (passage_attention._value.unsqueeze(-1) * self.encoded_passage).sum(0)
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        linear3 = self.parameters.relocate_linear3
        linear4 = self.parameters.relocate_linear4
        # Shape: (passage_length,)
        return_passage_attention = linear2(linear1(self.encoded_passage) * linear3(attended_passage) * linear4(attended_question)).squeeze()
        return PassageAttention(return_passage_attention)


    @predicate
    def find_passageSpanAnswer(self, passage_attention: PassageAttention) -> PassageSpanAnswer:
        # Shape: (1, D)
        attended_passage = allenutil.weighted_sum(self.encoded_passage.unsqueeze(0),
                                                   passage_attention._value.unsqueeze(0))

        # (passage_length, 2*encoding_dim)
        passage_for_span_start = torch.cat(
            [self.encoded_passage,
             attended_passage.repeat(self.encoded_passage.size(0), 1)],
            -1)

        # Shape: (passage_length)
        passage_span_start_logits = self.parameters.passage_span_start_predictor(passage_for_span_start).squeeze(1)
        passage_span_end_logits = self.parameters.passage_span_end_predictor(passage_for_span_start).squeeze(1)

        passage_span_start_log_probs = allenutil.masked_log_softmax(passage_span_start_logits, self.passage_mask)
        passage_span_end_log_probs = allenutil.masked_log_softmax(passage_span_end_logits, self.passage_mask)

        return PassageSpanAnswer(passage_span_start_log_probs=passage_span_start_log_probs,
                                 passage_span_end_log_probs=passage_span_end_log_probs)

    @predicate
    def find_questionSpanAnswer(self, passage_attention: PassageAttention) -> QuestionSpanAnswer:
        # Shape: (1, D)
        attended_passage = allenutil.weighted_sum(self.encoded_passage.unsqueeze(0),
                                                  passage_attention._value.unsqueeze(0))

        # (question_length, 2*encoding_dim)
        question_for_span_start = torch.cat(
            [self.encoded_question, attended_passage.repeat(self.encoded_question.size(0), 1)],
            -1)

        # Shape: (question_length)
        question_span_start_logits = self.parameters.question_span_start_predictor(question_for_span_start).squeeze(1)
        question_span_end_logits = self.parameters.question_span_end_predictor(question_for_span_start).squeeze(1)

        question_span_start_log_probs = allenutil.masked_log_softmax(question_span_start_logits,
                                                                     self.question_mask)
        question_span_end_log_probs = allenutil.masked_log_softmax(question_span_end_logits,
                                                                self.question_mask)

        return QuestionSpanAnswer(question_span_start_log_probs=question_span_start_log_probs,
                                  question_span_end_log_probs=question_span_end_log_probs)


    ''' NOT IMPLEMENTED AFTER THIS '''

    # @predicate_with_side_args(['attended_question'])
    # def find(self, attended_question: Tensor) -> AttentionTensor:
    #     find = self.parameters.find_attention
    #     return find(attended_question.unsqueeze(0), self.encoded_passage.unsqueeze(0)).squeeze(0)
    #
    #
    # @predicate_with_side_args(['attended_question'])
    # def relocate(self, attention: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
    #     linear1 = self.parameters.relocate_linear1
    #     linear2 = self.parameters.relocate_linear2
    #     linear3 = self.parameters.relocate_linear3
    #     linear4 = self.parameters.relocate_linear4
    #     attended_passage = (attention.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
    #     return linear2(linear1(self.encoded_passage) * linear3(attended_passage) * linear4(attended_question)).squeeze()

    # @predicate
    # def and_(self, attention1: AttentionTensor, attention2: AttentionTensor) -> AttentionTensor:
    #     return torch.max(torch.stack([attention1, attention2], dim=0), dim=0)[0]
    #
    # @predicate
    # def or_(self, attention1: AttentionTensor, attention2: AttentionTensor) -> AttentionTensor:
    #     return torch.min(torch.stack([attention1, attention2], dim=0), dim=0)[0]
    #
    # @predicate
    # def count(self, attention: AttentionTensor) -> CountAnswer:
    #     lstm = self.parameters.count_lstm
    #     linear = self.parameters.count_linear
    #
    #     # (1, passage_length, 2)
    #     hidden_states = lstm(attention.unsqueeze(-1))[0]
    #     return linear(hidden_states.squeeze()[-1])
    #
    # @predicate
    # def maximum_number(self, numbers: NumberAnswer) -> NumberAnswer:
    #     cumulative_distribution_function = numbers.cumsum(0)
    #     cumulative_distribution_function_n = cumulative_distribution_function ** self.max_samples
    #     maximum_distribution = cumulative_distribution_function_n - torch.cat(
    #         (torch.zeros(1), cumulative_distribution_function_n[:-1]))
    #     return maximum_distribution
    #
    # @predicate
    # def subtract(self,
    #              attention1: AttentionTensor,
    #              attention2: AttentionTensor,
    #              attention_map: Dict[int, List[Tuple[int, int]]]) -> NumberAnswer:
    #     attention_product = torch.matmul(attention1.unsqueeze(-1), torch.t(attention2.unsqueeze(-1)))
    #     answers = torch.zeros(len(attention_map), )
    #     for candidate_index, (candidate_subtraction, indices) in enumerate(attention_map.items()):
    #         attention_sum = 0
    #         for index1, index2 in indices:
    #             attention_sum += attention_product[index1, index2]
    #         answers[candidate_index] = attention_sum
    #     return NumberAnswer(answers)
    #
    # @predicate
    # def add(self,
    #         attention1: AttentionTensor,
    #         attention2: AttentionTensor,
    #         attention_map: Dict[int, List[Tuple[int, int]]]) -> NumberAnswer:
    #     attention_product = torch.matmul(attention1.unsqueeze(-1), torch.t(attention2.unsqueeze(-1)))
    #     answers = torch.zeros(len(attention_map), )
    #     for candidate_index, (candidate_addition, indices) in enumerate(attention_map.items()):
    #         attention_sum = 0
    #         for index1, index2 in indices:
    #             attention_sum += attention_product[index1, index2]
    #         answers[candidate_index] = attention_sum
    #     return NumberAnswer(answers)


if __name__=='__main__':
    dl = DropLanguage(None, None, None, None, None)

    spanans = PassageSpanAnswer(torch.randn(5), torch.randn(5))

    print(spanans.__class__.__name__)
