from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import LSTM

import allennlp.nn.util as allenutil
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args, ExecutionError)

from semqa.domain_languages.drop.execution_parameters import ExecutorParameters


class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, other) -> bool:
        # Note that the logic below renders equality to be non-transitive. That is,
        # Date(2018, -1, -1) == Date(2018, 2, 3) and Date(2018, -1, -1) == Date(2018, 4, 5)
        # but Date(2018, 2, 3) != Date(2018, 4, 5).
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        year_is_same = self.year == -1 or other.year == -1 or self.year == other.year
        month_is_same = self.month == -1 or other.month == -1 or self.month == other.month
        day_is_same = self.day == -1 or other.day == -1 or self.day == other.day
        return year_is_same and month_is_same and day_is_same

    def __gt__(self, other) -> bool:
        # pylint: disable=too-many-return-statements
        # The logic below is tricky, and is based on some assumptions we make about date comparison.
        # Year, month or day being -1 means that we do not know its value. In those cases, the
        # we consider the comparison to be undefined, and return False if all the fields that are
        # more significant than the field being compared are equal. However, when year is -1 for both
        # dates being compared, it is safe to assume that the year is not specified because it is
        # the same. So we make an exception just in that case. That is, we deem the comparison
        # undefined only when one of the year values is -1, but not both.
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        # We're doing an exclusive or below.
        if (self.year == -1) != (other.year == -1):
            return False  # comparison undefined
        # If both years are -1, we proceed.
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1, or both are -1.
        if self.month == -1 or other.month == -1:
            return False
        if self.month != other.month:
            return self.month > other.month
        # The months and years are equal and not -1
        if self.day == -1 or other.day == -1:
            return False
        return self.day > other.day

    def __ge__(self, other) -> bool:
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        return self > other or self == other

    def __str__(self):
        return f"{self.year}-{self.month}-{self.day}"


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
                 passage_span_end_log_probs: Tensor,
                 start_logits,
                 end_logits) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (passage_span_start_log_probs, passage_span_end_log_probs)


class QuestionSpanAnswer():
    def __init__(self,
                 question_span_start_log_probs: Tensor,
                 question_span_end_log_probs: Tensor,
                 start_logits,
                 end_logits) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
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
                 passage_tokenidx2dateidx: torch.LongTensor,
                 passage_date_values: List[Date],
                 parameters: ExecutorParameters,
                 start_types = None,
                 max_samples=10) -> None:

        if start_types is None:
            start_types = {PassageSpanAnswer, QuestionSpanAnswer}
        super().__init__(start_types=start_types)

        self.encoded_passage = encoded_passage
        self.encoded_question = encoded_question
        self.question_mask = question_mask
        self.passage_mask = passage_mask

        # Shape: (passage_length, )
        self.passage_tokenidx2dateidx = passage_tokenidx2dateidx.long() if passage_tokenidx2dateidx is not None else None
        # List[Date] - number of unique dates in the passage
        self.passage_date_values = passage_date_values
        self.num_passage_dates = len(self.passage_date_values) if self.passage_date_values is not None else 0

        self.parameters = parameters
        self.max_samples = max_samples

    @predicate_with_side_args(['question_attention'])
    def find_QuestionAttention(self, question_attention: Tensor) -> QuestionAttention:
        return QuestionAttention(question_attention)


    @predicate
    def compare_date_greater_than(self, ques_attn1: QuestionAttention, ques_attn2: QuestionAttention) -> PassageAttention:
        def get_passage_attention(find_attention_module, encoded_question, question_attention,
                                  encoded_passage, passage_mask):
            ''' For a given question attention compute the passage attention '''
            attended_question = allenutil.weighted_sum(encoded_question.unsqueeze(0),
                                                       question_attention.unsqueeze(0))

            # Shape: (passage_length)
            passage_attention = find_attention_module(attended_question, encoded_passage.unsqueeze(0),
                                                      passage_mask.unsqueeze(0)).squeeze(0)

            return passage_attention


        def compute_date_scores(passage_passage_token2date_attention, passage_attention, num_passage_dates,
                                passage_tokenidx2dateidx):

            ''' Given a passage over passage token2date attention, and an additionalpassage attention for token
                importance, compute a distribution over normalized dates in the passage.

                Idea is to weigh the token distributions (for dates) according to the passage_attention, and then
                use these values as scores for the dates they refer to. Since each date can be referred to in
                multiple places, we use scatter_add_ to get the total_score. Softmax over these scores is the date dist.
            '''

            # Shape: (passage_length, ) -- indicating which tokens are dates
            passage_tokenidx2dateidx_mask = (passage_tokenidx2dateidx > -1)

            # Shape: (passage_length, passage_length) -- weighing each token2date distribution by the token attention
            passage_passage_tokendate_attention = passage_passage_token2date_attention * \
                                                    passage_attention.unsqueeze(1)

            # Shape: (passage_length, ) -- weighted average of distributions in above step
            # Attention value for each passage token to be a date associated to the query
            passage_datetoken_attention = passage_passage_tokendate_attention.sum(0)

            masked_passage_tokenidx2dateidx = passage_tokenidx2dateidx_mask.long() * passage_tokenidx2dateidx
            masked_passage_datetoken_attn = passage_tokenidx2dateidx_mask.float() * passage_datetoken_attention

            # Shape: (num_passage_dates, )
            # These will store the total attention value from each passage token for each normalized_date
            date_scores = passage_attention.new_zeros(num_passage_dates)

            date_scores.scatter_add_(0, masked_passage_tokenidx2dateidx, masked_passage_datetoken_attn)

            date_distribution = allenutil.masked_softmax(date_scores, mask=None)

            return date_distribution, date_scores

        qattn1 = ques_attn1._value * self.question_mask
        qattn2 = ques_attn2._value * self.question_mask

        # Shape: (passage_length)
        passage_attention_1 = get_passage_attention(self.parameters.find_attention,
                                                    self.encoded_question,
                                                    qattn1,
                                                    self.encoded_passage,
                                                    self.passage_mask)

        passage_attention_2 = get_passage_attention(self.parameters.find_attention,
                                                    self.encoded_question,
                                                    qattn2,
                                                    self.encoded_passage,
                                                    self.passage_mask)

        # Shape: (passage_length, passage_length) - for each token x in the row, weight given by it to each token y in
        # the column for y to be a date associated to x
        passage_passage_token2date_similarity = self.parameters.passage_to_date_attention(
                                                        self.encoded_passage.unsqueeze(0),
                                                        self.encoded_passage.unsqueeze(0)).squeeze(0)

        # Shape: (passage_length, passage_length) - above normalized across columns i.e. each row is normalized
        # distribution over tokens (likelihood of being a date token)
        passage_passage_token2date_attention = allenutil.masked_softmax(passage_passage_token2date_similarity,
                                                                        mask=self.passage_mask)

        date_scores_1, date_distribution_1 = compute_date_scores(passage_passage_token2date_attention,
                                                                 passage_attention_1,
                                                                 self.num_passage_dates,
                                                                 self.passage_tokenidx2dateidx)

        date_scores_2, date_distribution_2 = compute_date_scores(passage_passage_token2date_attention,
                                                                 passage_attention_2,
                                                                 self.num_passage_dates,
                                                                 self.passage_tokenidx2dateidx)


        # TODO(nitish): Not implemented after this
        # Use these date distributions to get a boolean distribution for greater than -- using placeholder right now
        bool = date_distribution_2.new_zeros(1).fill_(0.2)

        average_passage_distribution = bool * passage_attention_1 + (1 - bool) * passage_attention_2

        return PassageAttention(average_passage_distribution)

    # @predicate
    # def find_PassageAttention(self, question_attn: QuestionAttention) -> PassageAttention:
    #     # Shape: (1, D)
    #     attended_question = allenutil.weighted_sum(self.encoded_question.unsqueeze(0),
    #                                                question_attn._value.unsqueeze(0))
    #     # Shape: (passage_length)
    #     passage_attention = self.parameters.find_attention(attended_question,
    #                                                        self.encoded_passage.unsqueeze(0)).squeeze(0)
    #
    #     passage_attention = allenutil.masked_softmax(passage_attention, self.passage_mask)
    #
    #     return PassageAttention(passage_attention=passage_attention)


    # @predicate_with_side_args(['question_attention'])
    # def relocate_PassageAttention(self,
    #                               passage_attention: PassageAttention,
    #                               question_attention: Tensor) -> PassageAttention:
    #     # Shape: (1, D)
    #     attended_question = allenutil.weighted_sum(self.encoded_question.unsqueeze(0),
    #                                                question_attention.unsqueeze(0))
    #
    #     # Shape: (encoding_dim, 1)
    #     attended_passage = (passage_attention._value.unsqueeze(-1) * self.encoded_passage).sum(0)
    #     linear1 = self.parameters.relocate_linear1
    #     linear2 = self.parameters.relocate_linear2
    #     linear3 = self.parameters.relocate_linear3
    #     linear4 = self.parameters.relocate_linear4
    #     # Shape: (passage_length,)
    #     return_passage_attention = linear2(linear1(self.encoded_passage) * linear3(attended_passage) * linear4(attended_question)).squeeze()
    #     return PassageAttention(return_passage_attention)


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

        passage_span_start_log_probs = allenutil.replace_masked_values(passage_span_start_log_probs,
                                                                       self.passage_mask, -1e7)
        passage_span_end_log_probs = allenutil.replace_masked_values(passage_span_end_log_probs,
                                                                     self.passage_mask, -1e7)

        return PassageSpanAnswer(passage_span_start_log_probs=passage_span_start_log_probs,
                                 passage_span_end_log_probs=passage_span_end_log_probs,
                                 start_logits=passage_span_start_logits,
                                 end_logits=passage_span_end_logits)

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

        question_span_start_log_probs = allenutil.replace_masked_values(question_span_start_log_probs,
                                                                        self.question_mask, -1e7)
        question_span_end_log_probs = allenutil.replace_masked_values(question_span_end_log_probs,
                                                                      self.question_mask, -1e7)

        return QuestionSpanAnswer(question_span_start_log_probs=question_span_start_log_probs,
                                  question_span_end_log_probs=question_span_end_log_probs,
                                  start_logits=question_span_start_logits,
                                  end_logits=question_span_end_logits)


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
    dl = DropLanguage(None, None, None, None, None, None, None)

    # print(spanans.__class__.__name__)
