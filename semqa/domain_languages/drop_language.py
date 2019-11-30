import logging
from typing import List, Tuple, Any
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

import allennlp.nn.util as allenutil
from allennlp_semparse.domain_languages import (DomainLanguage, predicate, predicate_with_side_args)
from allennlp_semparse.common import ExecutionError

from semqa.domain_languages.drop_execution_parameters import ExecutorParameters
from semqa.domain_languages import domain_language_utils as dlutils
import utils.util as myutils

from semqa.profiler.profile import Profile, profile_func_decorator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
            # Penalize the underspecified date
            if self.year == -1:
                return False
            else:
                return True
            # return False  # comparison undefined
        # If both years are -1, we proceed.
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1, or both are -1.

        if self.month == -1 and other.month == -1:
            return False
        if self.month == -1:
            return False
        if other.month == -1:
            return True
        if self.month != other.month:
            return self.month > other.month
        # The months and years are equal and not -1
        if self.day == -1 and other.day == -1:
            return False
        if self.day == -1:
            return False
        if other.day == -1:
            return True
        return self.day > other.day

    def __ge__(self, other) -> bool:
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        return self > other or self == other

    def __str__(self):
        return f"{self.year}/{self.month}/{self.day}"

    def year_diff(self, other):
        """ Returns the difference in years between two dates.
            1. Either of the years is not defined, return -1
            ~ ~ ~ ~ 2. The difference in self - other is negative ~ ~ ~ ~
        """
        assert isinstance(other, Date), "Given object is not a date instance"
        other: Date = other
        year1, year2 = self.year, other.year
        if year1 == -1 or year2 == -1:
            return -1
        else:
            year_diff = year1 - year2
            return year_diff
            # if year_diff > 0:
            #     return year_diff
            # else:
            #     return 0


class PassageSpanAnswer:
    def __init__(
        self,
        passage_span_start_log_probs: Tensor,
        passage_span_end_log_probs: Tensor,
        start_logits,
        end_logits,
        loss=0.0,
        debug_value="",
    ) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (passage_span_start_log_probs, passage_span_end_log_probs)
        self.loss = loss
        self.debug_value = debug_value


class QuestionSpanAnswer:
    def __init__(
        self,
        question_span_start_log_probs: Tensor,
        question_span_end_log_probs: Tensor,
        start_logits,
        end_logits,
        loss=0.0,
        debug_value="",
    ) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (question_span_start_log_probs, question_span_end_log_probs)
        self.loss = loss
        self.debug_value = debug_value


class QuestionAttention:
    def __init__(self, question_attention, debug_value=""):
        self._value = question_attention
        self.debug_value = debug_value


class PassageAttention:
    def __init__(self, passage_attention, loss=0.0, debug_value=""):
        self._value = passage_attention
        self.loss = loss
        self.debug_value = debug_value


class PassageAttention_answer:
    def __init__(self, passage_attention, loss=0.0, debug_value=""):
        self._value = passage_attention
        self.loss = loss
        self.debug_value = debug_value


class YearDifference:
    def __init__(self, year_difference_dist, loss=0.0, debug_value=""):
        self._value = year_difference_dist
        self.loss = loss
        self.debug_value = debug_value


class PassageNumber:
    def __init__(self, passage_number_dist, loss=0.0, debug_value=""):
        self._value = passage_number_dist
        self.loss = loss
        self.debug_value = debug_value


class ComposedNumber:
    def __init__(self, composed_number_dist, loss=0.0, debug_value=""):
        self._value = composed_number_dist
        self.loss = loss
        self.debug_value = debug_value


class CountNumber:
    def __init__(self, count_number_dist, loss=0.0, debug_value=""):
        self._value = count_number_dist
        self.loss = loss
        self.debug_value = debug_value


# A ``NumberAnswer`` is a distribution over the possible answers from addition and subtraction.
class NumberAnswer(Tensor):
    pass


# A ``CountAnswer`` is a distribution over the possible counts in the passage.
class CountAnswer(Tensor):
    pass


def clamp_distribution(distribution):
    return torch.clamp(distribution, min=1e-20, max=1 - 1e-20)


def get_empty_language_object():
    droplanguage = DropLanguage(
        rawemb_question=None,
        embedded_question=None,
        encoded_question=None,
        rawemb_passage=None,
        embedded_passage=None,
        encoded_passage=None,
        modeled_passage=None,
        question_mask=None,
        passage_mask=None,
        passage_sentence_boundaries=None,
        passage_tokenidx2dateidx=None,
        passage_date_values=None,
        passage_tokenidx2numidx=None,
        passage_num_values=None,
        composed_numbers=None,
        passage_number_sortedtokenidxs=None,
        add_num_combination_indices=None,
        sub_num_combination_indices=None,
        year_differences=None,
        year_differences_mat=None,
        count_num_values=None,
        question_passage_attention=None,
        passage_question_attention=None,
        passage_token2date_alignment=None,
        passage_token2startdate_alignment=None,
        passage_token2enddate_alignment=None,
        passage_token2num_alignment=None,
        parameters=None,
        start_types=None,
    )
    return droplanguage


class DropLanguage(DomainLanguage):
    """
    DomainLanguage for the DROP dataset based on neural module networks. This language has a `learned execution model`,
    meaning that the predicates in this language have learned parameters.

    Parameters
    ----------
    parameters : ``DropNmnParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    implicit_numbers = [0, 100.0]

    # TODO(nitish): Defaulting all parameters to None since in the reader we create an
    def __init__(
        self,
        rawemb_question: Tensor,
        embedded_question: Tensor,
        encoded_question: Tensor,
        rawemb_passage: Tensor,
        embedded_passage: Tensor,
        encoded_passage: Tensor,
        question_mask: Tensor,
        passage_mask: Tensor,
        passage_sentence_boundaries: Tensor,
        passage_tokenidx2dateidx: torch.LongTensor,
        passage_date_values: List[Date],
        passage_tokenidx2numidx: torch.LongTensor,
        passage_num_values: List[float],
        composed_numbers: List[float],
        passage_number_sortedtokenidxs: List[int],
        add_num_combination_indices: Tensor,
        sub_num_combination_indices: Tensor,
        year_differences: List[int],
        year_differences_mat: np.array,
        count_num_values: List[int],
        question_passage_attention: Tensor,
        passage_question_attention: Tensor,
        passage_token2date_alignment: Tensor,
        passage_token2startdate_alignment: Tensor,
        passage_token2enddate_alignment: Tensor,
        passage_token2num_alignment: Tensor,
        parameters: ExecutorParameters,
        modeled_passage: Tensor = None,
        start_types=None,
        device_id: int = -1,
        max_samples=5,
        metadata={},
        debug=False,
    ) -> None:

        if start_types is None:
            start_types = {PassageSpanAnswer, YearDifference, PassageNumber, ComposedNumber, CountNumber}
            # QuestionSpanAnswer - could be one

        super().__init__(start_types=start_types)

        if embedded_question is None:
            return

        self.rawemb_question = rawemb_question
        self.embedded_question = embedded_question
        self.encoded_question = encoded_question
        self.question_mask = question_mask

        self.rawemb_passage = rawemb_passage
        self.embedded_passage = embedded_passage
        self.encoded_passage = encoded_passage
        self.modeled_passage = modeled_passage
        self.passage_mask = passage_mask
        self.passage_length = self.passage_mask.size()[0]
        self.passage_sentence_boundaries = passage_sentence_boundaries  # LongTensor of shape (numsent, 2)
        self.passage_sentboundary_mask = (self.passage_sentence_boundaries[:, 0] >= 0).long()  # Shape: (num_sents)
        self.passage_sentence_boundaries_masked = (self.passage_sentence_boundaries *
                                                   self.passage_sentboundary_mask.unsqueeze(1))
        self.passage_sentence_starts_masked = self.passage_sentence_boundaries[:, 0] * self.passage_sentboundary_mask
        self.passage_sentence_ends_masked = self.passage_sentence_boundaries[:, 1] * self.passage_sentboundary_mask

        self.passage_encoding_dim = self.modeled_passage.size()[-1]
        self.question_encoding_dim = self.encoded_question.size()[-1]

        # Shape: (passage_length, )
        self.passage_tokenidx2dateidx = passage_tokenidx2dateidx.long()
        passage_tokenidx2dateidx_mask = self.passage_tokenidx2dateidx > -1
        self.passage_datetokens_mask_long = passage_tokenidx2dateidx_mask.long()
        self.passage_datetokens_mask_float = passage_tokenidx2dateidx_mask.float()
        # List[Date] - number of unique dates in the passage
        self.passage_date_values: List[Date] = passage_date_values
        self.num_passage_dates = len(self.passage_date_values)

        # Shape: (passage_length, )
        self.passage_tokenidx2numidx = passage_tokenidx2numidx.long()
        passage_tokenidx2numidx_mask = self.passage_tokenidx2numidx > -1
        self.passage_numtokens_mask_long = passage_tokenidx2numidx_mask.long()
        self.passage_numtokens_mask_float = passage_tokenidx2numidx_mask.float()
        # List[float] - number of unique numbers in the passage (includes implicit numbers)
        self.passage_num_values: List[float] = passage_num_values
        self.composed_numbers: List[float] = composed_numbers
        # List[int] - number-token-idxs in an order so their values are sorted. Needed to max/min pattn
        self.passage_number_sortedtokenidxs = passage_number_sortedtokenidxs
        self.num_passage_nums = len(self.passage_num_values)
        self.num_composed_nums = len(self.composed_numbers)

        # List[int ] -- Shape: (num_implicit_numbers)
        implicit_num_indices = torch.LongTensor([self.passage_num_values.index(x)
                                                 for x in DropLanguage.implicit_numbers])
        self.implicit_num_indices = allenutil.move_to_device(implicit_num_indices, cuda_device=device_id)

        # Shape: (size_composed_numbers, max_num_combinations, 2) -- for each number in composed_nums (dim=0),
        #  indices for other passage_number combinations (dim=1) that lead to this number using the op.
        #  ComposedNum[i] = PassageNum(M[i,j,0]) OP PassageNum(M[i,j,1]) \forall j
        # Since all numbers won't have same num of combinations, these indices are padded w/ -1
        self.add_num_combination_indices = add_num_combination_indices.long()
        self.sub_num_combination_indices = sub_num_combination_indices.long()
        self.add_num_combination_mask = (self.add_num_combination_indices > -1).long()
        self.sub_num_combination_mask = (self.sub_num_combination_indices > -1).long()

        self.parameters = parameters
        self.max_samples = max_samples

        self.metadata = metadata
        self._debug = debug

        self.device_id = device_id

        # Shape: (question_length, passage_length)
        self.question_passage_attention = (
            question_passage_attention
        )  # initialization_returns["question_passage_attention"]
        # Shape: (passage_length, question_length)
        self.passage_question_attention = passage_question_attention
        # Shape: (passage_length, passage_length)
        self.passage_passage_token2date_alignment = passage_token2date_alignment
        self.passage_passage_token2startdate_alignment = passage_token2startdate_alignment
        self.passage_passage_token2enddate_alignment = passage_token2enddate_alignment
        self.passage_passage_token2num_alignment = passage_token2num_alignment
        initialization_returns = self.initialize()
        self.date_lt_mat = initialization_returns["date_lt_mat"]
        self.date_gt_mat = initialization_returns["date_gt_mat"]
        # These matrices are for passage numbers
        self.num_lt_mat = initialization_returns["num_lt_mat"]
        self.num_gt_mat = initialization_returns["num_gt_mat"]

        # List[int]
        self.year_differences = year_differences
        # Shape: (num_passage_dates, num_passage_dates, num_of_year_differences)
        self.year_differences_mat = allenutil.move_to_device(
            torch.FloatTensor(year_differences_mat), cuda_device=self.device_id
        )
        # List[int]
        self.count_num_values = count_num_values
        self.countvals = allenutil.move_to_device(torch.FloatTensor(range(0, 10)), cuda_device=self.device_id)

        """
        if self._debug:
            num_date_tokens = self.passage_datetokens_mask_float.sum()
            plen = self.passage_mask.sum()
            siml1norm = self.passage_passage_token2date_similarity.norm(p=1)/(num_date_tokens * plen)
            sim_avgval = self.passage_passage_token2date_similarity.sum() / (num_date_tokens * plen)
            if torch.isnan(sim_avgval):
                print("Date fault")
                print(f"{self.num_passage_dates} : {num_date_tokens}")
                print(self.passage_passage_token2date_similarity)
            print(f"Passage Token2Date sim, Avg L1 Norm: {siml1norm}. Avg Val: {sim_avgval}")
            num_numb_tokens = self.passage_numtokens_mask_float.sum()
            plen = self.passage_mask.sum()
            siml1norm = self.passage_passage_token2num_similarity.norm(p=1) / (num_numb_tokens * plen)
            sim_avgval = self.passage_passage_token2num_similarity.sum() / (num_numb_tokens * plen)
            if torch.isnan(sim_avgval):
                print("Num fault")
                print(self.passage_passage_token2num_similarity)
            print(f"Passage Token2Num sim, Avg L1 Norm: {siml1norm}. Avg Val: {sim_avgval}")
        """

    def initialize(self):
        date_gt_mat, date_lt_mat = self.compute_date_comparison_matrices(self.passage_date_values, self.device_id)
        num_gt_mat, num_lt_mat = self.compute_item_comparison_matrices(self.passage_num_values, self.device_id)

        return {
            "date_gt_mat": date_gt_mat,
            "date_lt_mat": date_lt_mat,
            "num_gt_mat": num_gt_mat,
            "num_lt_mat": num_lt_mat,
        }

    @staticmethod
    def compute_date_comparison_matrices(date_values: List[Date], device_id: int):
        date_gt_mat = [[0 for _ in range(len(date_values))] for _ in range(len(date_values))]
        date_lt_mat = [[0 for _ in range(len(date_values))] for _ in range(len(date_values))]
        # self.encoded_passage.new_zeros(self.num_passage_dates, self.num_passage_dates)
        for i in range(len(date_values)):
            for j in range(len(date_values)):
                date_gt_mat[i][j] = 1.0 if date_values[i] > date_values[j] else 0.0
                date_lt_mat[i][j] = 1.0 if date_values[i] < date_values[j] else 0.0
                # date_greater_than_mat[j][i] = 1.0 - date_greater_than_mat[i][j]
        date_gt_mat = allenutil.move_to_device(torch.FloatTensor(date_gt_mat), device_id)
        date_lt_mat = allenutil.move_to_device(torch.FloatTensor(date_lt_mat), device_id)

        return date_gt_mat, date_lt_mat

    @staticmethod
    def compute_item_comparison_matrices(values: List[Any], device_id: int):
        values_is_sorted = all(values[i] <= values[i+1] for i in range(0, len(values) - 1))
        if values_is_sorted:
            num_values = len(values)
            ones = np.ones((num_values, num_values), dtype=np.float32)
            upper_triu = np.triu(ones, k=1)  # The k=1 ensures main diagonal is zero for strict lt_mat
            lower_triu = np.tril(ones, k=-1)  # The k=1 ensures main diagonal is zero for strict gt_mat
            gt_mat = allenutil.move_to_device(torch.FloatTensor(lower_triu), device_id)
            lt_mat = allenutil.move_to_device(torch.FloatTensor(upper_triu), device_id)
        else:
            gt_mat = [[0 for _ in range(len(values))] for _ in range(len(values))]
            lt_mat = [[0 for _ in range(len(values))] for _ in range(len(values))]

            for i in range(len(values)):
                for j in range(len(values)):
                    gt_mat[i][j] = 1.0 if values[i] > values[j] else 0.0
                    lt_mat[i][j] = 1.0 if values[i] < values[j] else 0.0
                    # date_greater_than_mat[j][i] = 1.0 - date_greater_than_mat[i][j]
            gt_mat = allenutil.move_to_device(torch.FloatTensor(gt_mat), device_id)
            lt_mat = allenutil.move_to_device(torch.FloatTensor(lt_mat), device_id)

        return gt_mat, lt_mat

    # @predicate_with_side_args(['question_attention'])
    # def find_QuestionAttention(self, question_attention: Tensor) -> QuestionAttention:
    #
    #     debug_value = ""
    #     if self._debug:
    #         qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(question_attention, self.metadata["question_tokens"])
    #         debug_value = qattn_vis_most
    #
    #     return QuestionAttention(question_attention, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "passage_attention"])
    def find_PassageAttention(self, question_attention: Tensor, passage_attention: Tensor = None) -> PassageAttention:
        with Profile("find-pattn"):
            # The passage attention is only used as supervision for certain synthetic questions
            if passage_attention is not None:
                return PassageAttention(passage_attention, debug_value="Supervised-Pattn-Used")

            question_attention = question_attention * self.question_mask

            # Shape: (question_length, passage_length)
            question_passage_attention = self.question_passage_attention * question_attention.unsqueeze(1)

            passage_attention = question_passage_attention.sum(0)
            passage_attention = clamp_distribution(passage_attention)

            debug_value = ""
            if self._debug:
                qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(
                    question_attention, self.metadata["question_tokens"]
                )
                debug_value += f"Qattn: {qattn_vis_complete}"
                pattn_vis_complete, pattn_vis_most = dlutils.listTokensVis(
                    passage_attention, self.metadata["passage_tokens"]
                )
                most_attended_spans = dlutils.mostAttendedSpans(passage_attention, self.metadata["passage_tokens"])
                debug_value += f"\nPattn: {pattn_vis_complete}"
                debug_value += f"\nMostAttendedSpans: {most_attended_spans}"

        return PassageAttention(passage_attention, debug_value=debug_value)

    @predicate_with_side_args(["question_attention"])
    def filter_PassageAttention(
        self, passage_attention: PassageAttention, question_attention: Tensor
    ) -> PassageAttention:
        with Profile("filter-attn"):
            passage_attn: Tensor = passage_attention._value

            question_attention = question_attention * self.question_mask
            # Shape: (ques_encoding_dim)
            weighted_question_vector = torch.sum(self.encoded_question * question_attention.unsqueeze(1), 0)

            # Shape: (passage_length, encoded_dim)
            passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage

            # Shape: (1, num_sents, 2 * passage_dim)
            sentence_start_end_repr = self.parameters._endpoint_span_extractor(
                sequence_tensor=passage_repr.unsqueeze(0),
                span_indices=self.passage_sentence_boundaries_masked.unsqueeze(0))

            sentence_start_end_repr = (sentence_start_end_repr[:, :, 0:self.passage_encoding_dim] +
                                       sentence_start_end_repr[:, :, self.passage_encoding_dim:])

            # Shape: (1, 1, num_sents)
            sentence_logits_unsqueezed= self.parameters.filter_matrix_attention(
                weighted_question_vector.unsqueeze(0).unsqueeze(1), sentence_start_end_repr)
            # Shape: (num_sents)
            sentence_logits = sentence_logits_unsqueezed.squeeze(0).squeeze(0)
            sentence_filter_prob = torch.sigmoid(sentence_logits) * self.passage_sentboundary_mask.float()
            # Shape: (passage_length)
            range_vec = allenutil.get_range_vector(self.passage_length, device=self.device_id)
            range_vec_unsq = range_vec.unsqueeze(0)

            # Shape: (num_sents, passage_length)
            lower = range_vec_unsq >= self.passage_sentence_starts_masked.unsqueeze(1)
            upper = range_vec_unsq <= self.passage_sentence_ends_masked.unsqueeze(1)
            # (num_sents, passage_length)
            sentence_bool = (lower * upper).float() * self.passage_sentboundary_mask.unsqueeze(1).float()

            # Shape: (passage_length, )
            filter_attn = torch.sum(sentence_bool * sentence_filter_prob.unsqueeze(1), dim=0)
            original_filter_attn = filter_attn * passage_attn
            filtered_passage_attention = original_filter_attn / torch.sum(original_filter_attn)
            filtered_passage_attention = clamp_distribution(filtered_passage_attention)

            """
            # Shape: (1, 1, passage_length)
            passage_logits_unsqueezed = self.parameters.filter_matrix_attention(
                weighted_question_vector.unsqueeze(0).unsqueeze(1), passage_repr.unsqueeze(0)
            )

            passage_logits = passage_logits_unsqueezed.squeeze(0).squeeze(0)

            # filter_attn = allenutil.masked_softmax(passage_logits, mask=self.passage_mask, memory_efficient=True)
            filter_attn = torch.sigmoid(passage_logits * self.passage_mask)

            original_filter_attn = filter_attn * passage_attn

            filtered_passage_attention = original_filter_attn / torch.sum(original_filter_attn)
            filtered_passage_attention = clamp_distribution(filtered_passage_attention)
            """

            loss = passage_attention.loss

            debug_value = ""
            if self._debug:
                qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(
                    question_attention, self.metadata["question_tokens"]
                )
                debug_value += f"Qattn: {qattn_vis_complete}"

                f_attn_vis, _ = dlutils.listTokensVis(filter_attn, self.metadata["passage_tokens"])
                most_attended_spans = dlutils.mostAttendedSpans(filter_attn, self.metadata["passage_tokens"])
                debug_value += f"\nFilterAttn: {f_attn_vis}"
                debug_value += f"\nMostAttended: {most_attended_spans}"

                pattn_vis_complete, pattn_vis_most = dlutils.listTokensVis(
                    filtered_passage_attention, self.metadata["passage_tokens"]
                )
                most_attended_spans = dlutils.mostAttendedSpans(filtered_passage_attention, self.metadata["passage_tokens"])
                debug_value += f"\nPattn: {pattn_vis_complete}"
                debug_value += f"\nMostAttended: {most_attended_spans}"

        return PassageAttention(filtered_passage_attention, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention"])
    def relocate_PassageAttention(
        self, passage_attention: PassageAttention, question_attention: Tensor
    ) -> PassageAttention_answer:
        with Profile("relocate-attn"):
            passage_attn: Tensor = passage_attention._value
            passage_attn = passage_attn * self.passage_mask

            question_attention = question_attention * self.question_mask
            # Shape: (encoding_dim)
            weighted_question_vector = torch.sum(self.encoded_question * question_attention.unsqueeze(1), 0)

            # Shape: (passage_length, encoded_dim)
            passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage
            # Shape: (passage_length, encoded_dim)
            q_p_repr = weighted_question_vector.unsqueeze(0) + passage_repr

            # Shape: (passage_length, passage_length)
            passage_passage_relocate_similarity = self.parameters.relocate_matrix_attention(
                q_p_repr.unsqueeze(0), passage_repr.unsqueeze(0)
            ).squeeze(0)
            # Shape: (passage_length, passage_length)
            p_to_p_relocate_attention = allenutil.masked_softmax(
                passage_passage_relocate_similarity, mask=self.passage_mask.unsqueeze(0), dim=-1
            )
            p_to_p_relocate_attention = p_to_p_relocate_attention * self.passage_mask.unsqueeze(1)

            passage_length = passage_attn.size()[0]
            inwindow_mask, _ = dlutils.masking_blockdiagonal(
                passage_length=passage_length, window=15, device_id=self.device_id
            )
            inwindow_aux_loss = dlutils.aux_window_loss(
                ptop_attention=p_to_p_relocate_attention, passage_mask=self.passage_mask, inwindow_mask=inwindow_mask
            )

            # Shape: (passage_length, )
            relocate_attn = (p_to_p_relocate_attention * passage_attn.unsqueeze(1)).sum(0)
            relocate_attn = clamp_distribution(relocate_attn)

            loss = passage_attention.loss
            loss += 2.0 * inwindow_aux_loss

            debug_value = ""
            if self._debug:
                qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(
                    question_attention, self.metadata["question_tokens"]
                )
                debug_value += f"Qattn: {qattn_vis_complete}"
                r_attn_vis, _ = dlutils.listTokensVis(relocate_attn, self.metadata["passage_tokens"])
                most_attended_spans = dlutils.mostAttendedSpans(relocate_attn, self.metadata["passage_tokens"])
                debug_value += f"\nRelocateAttn: {r_attn_vis}"
                debug_value += f"\nMostAttended: {most_attended_spans}"

                # if self.metadata["original_question"] == "Which player scored the first touchdown of the game?" and \
                #         "In a matchup of fellow" in self.metadata["original_passage"]:
                #     import pdb
                #     pdb.set_trace()

        return PassageAttention_answer(relocate_attn, loss=loss, debug_value=debug_value)

    # New Date Distribtion
    def compute_date_scores(self, passage_attention: Tensor, date_type: str = None):
        """ Given a passage over passage token2date attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) dates in the passage.

            Using the token_attention from the passage_attention, find the expected input_token2date_token
            distribution - weigh each row in passage_passage_token2date_attention and sum
            Use this date_token attention as scores for the dates they refer to. Since each date can be referred
            to in multiple places, we use scatter_add_ to get the total_score.
            Softmax over these scores is the date dist.
        """
        with Profile("date-dist."):
            # passage_date_alignment_matrix = allenutil.masked_softmax(self.passage_passage_token2date_similarity,
            #                                                          mask=self.passage_datetokens_mask_float.unsqueeze(0),
            #                                                          memory_efficient=True)

            if date_type is None:
                passage_date_alignment_matrix = self.passage_passage_token2date_alignment
            elif date_type == "start":
                passage_date_alignment_matrix = self.passage_passage_token2startdate_alignment
            elif date_type == "end":
                passage_date_alignment_matrix = self.passage_passage_token2enddate_alignment
            else:
                raise NotImplementedError

            attn_weighted_date_aligment_matrix = passage_date_alignment_matrix * passage_attention.unsqueeze(1)
            # Shape: (passage_length, )
            passage_date_token_probs = attn_weighted_date_aligment_matrix.sum(0)

            """
            if self._debug:
                print('-------------------------')
                print(self.metadata['question_tokens'])
                passage_tokens = self.metadata['passage_tokens']
                attn, topattn = dlutils.listTokensVis(passage_attention, passage_tokens)
                print(f"PassageAttention: Top: {topattn}")
                print(attn)
                print()
    
                print("Only showing 10 date-tokens ...")
                _, date_indices = torch.topk(self.passage_tokenidx2dateidx, k=5, dim=0)
                date_indices = myutils.tocpuNPList(date_indices)
                for number_idx in date_indices:
                    token2datescores = passage_date_alignment_matrix[:, number_idx]
                    _, top_tokens = torch.topk(token2datescores, 5, dim=0)
                    top_tokens = myutils.tocpuNPList(top_tokens)
                    print(f"{passage_tokens[number_idx]}")
                    print([passage_tokens[x] for x in top_tokens])
                    print(f"Sum: {torch.sum(token2datescores)}")
                    compvis, _ = dlutils.listTokensVis(token2datescores, passage_tokens)
                    print(compvis)
    
                print("After passage attention; number-token-probs:")
                attn, _ = dlutils.listTokensVis(passage_date_token_probs, passage_tokens)
                print(attn)
                print()
                print("-----------------------------------")
            """

            masked_passage_tokenidx2dateidx = self.passage_datetokens_mask_long * self.passage_tokenidx2dateidx

            date_distribution = passage_attention.new_zeros(self.num_passage_dates)
            date_distribution.scatter_add_(0, masked_passage_tokenidx2dateidx, passage_date_token_probs)

            date_distribution = clamp_distribution(date_distribution)

            date_distribution_entropy = -1 * torch.sum(date_distribution * torch.log(date_distribution + 1e-40))

        return date_distribution, date_distribution, date_distribution_entropy

    # New Num Distribution by first computing a number-distribution for each passage-token
    def compute_num_distribution(self, passage_attention: Tensor):
        """ Given a passage over passage token2num attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) nums in the passage.
            See compute_date_distribution for details
        """

        with Profile("num-dist"):
            # Shape: (passage_length, passage_length) -- each row softmaxed over the number-tokens
            # passage_number_alignment_matrix = allenutil.masked_softmax(self.passage_passage_token2num_similarity,
            #                                                            mask=self.passage_numtokens_mask_float.unsqueeze(0),
            #                                                            memory_efficient=True)

            # Shape: (passage_length, passage_length) -- Each row is a masked-softmax over number-tokens
            passage_number_alignment_matrix = self.passage_passage_token2num_alignment

            # (passage_length, passage_length)
            attn_weighted_number_aligment_matrix = passage_number_alignment_matrix * passage_attention.unsqueeze(1)
            # Shape: (passage_length, )
            passage_number_token_probs = attn_weighted_number_aligment_matrix.sum(0)

            """
            if self._debug:
                print('-------------------------')
                print(self.metadata['question_tokens'])
                passage_tokens = self.metadata['passage_tokens']
                attn, topattn = dlutils.listTokensVis(passage_attention, passage_tokens)
                print(f"PassageAttention: Top: {topattn}")
                print(attn)
                print()
    
                print("Only showing 10 number-tokens ...")
                _, number_indices = torch.topk(self.passage_tokenidx2numidx, k=10, dim=0)
                number_indices = myutils.tocpuNPList(number_indices)
                for number_idx in number_indices:
                    token2numberscores = passage_number_alignment_matrix[:, number_idx]
                    _, top_tokens = torch.topk(token2numberscores, 5, dim=0)
                    top_tokens = myutils.tocpuNPList(top_tokens)
                    print(f"{passage_tokens[number_idx]}")
                    print([passage_tokens[x] for x in top_tokens])
                    print(f"Sum: {torch.sum(token2numberscores)}")
                    compvis, _ = dlutils.listTokensVis(token2numberscores, passage_tokens)
                    print(compvis)
    
                print("After passage attention; number-token-probs:")
                attn, _ = dlutils.listTokensVis(passage_number_token_probs, passage_tokens)
                print(attn)
                print()
                print("-----------------------------------")
            """

            masked_passage_tokenidx2numidx = self.passage_numtokens_mask_long * self.passage_tokenidx2numidx

            """ normalized method with method 2 """
            num_distribution = passage_attention.new_zeros(self.num_passage_nums)
            num_distribution.scatter_add_(0, masked_passage_tokenidx2numidx, passage_number_token_probs)

            num_distribution = clamp_distribution(num_distribution)

            num_distribution_entropy = -1 * torch.sum(num_distribution * torch.log(num_distribution + 1e-40))
        return num_distribution, num_distribution, num_distribution_entropy


    def compute_implicitnum_distribution(self, question_attention: Tensor):
        """ Given a question attention, compute a distribution over implicit numbers for this language.
            See compute_date_distribution for details
        """
        with Profile("implicit-num"):
            question_attention = question_attention * self.question_mask
            # Shape: (encoding_dim)
            weighted_question_vector = torch.sum(self.encoded_question * question_attention.unsqueeze(1), 0)

            # unsqueeze -- one for num-of-vecs and one for batch
            weighted_question_vector_ex = weighted_question_vector.unsqueeze(0).unsqueeze(0)

            implicit_num_embeddings_ex = self.parameters.implicit_num_embeddings.unsqueeze(0)

            # Shape: (1, 1, num_implicit_numbers)
            implicit_num_logits = self.parameters.implicitnum_bilinear_attention(weighted_question_vector_ex,
                                                                                 implicit_num_embeddings_ex)
            # Shape: (num_implicit_numbers)
            implicit_num_logits = implicit_num_logits.squeeze(0).squeeze(0)
            implicit_num_probs = torch.nn.functional.softmax(implicit_num_logits, dim=-1)

            num_distribution = implicit_num_probs.new_zeros(self.num_passage_nums)
            num_distribution.scatter_add_(0, self.implicit_num_indices, implicit_num_probs)

            num_distribution = clamp_distribution(num_distribution)
        return num_distribution


    def expected_number_addsub(self, num_dist_1: torch.FloatTensor, num_dist_2: torch.FloatTensor, operation: str):
        """Compute the expected number distribution for an addition/subtraction operation.

        add_num_combination_indices / sum_num_combination_indices: (size_composed_numbers, num_combinations, 2)
        Each combination is a tuple indexed into the passage numbers

        Expected distribution over composed numbers is computed by; for each composed number,
         1. extracting from num_dist_1, the probability of the first number in the combination
         2. extracting from num_dist_2, the probability of the second number in the combination
         3. Computing the marginalized joint probability by summing over the product  1. X 2.

        Parameters:
        -----------
        num_dist_1: (num_passage_numbers) Passage number distribution
        num_dist_2: (num_passage_numbers) Passage number distribution
        """
        with Profile("num-add-sub"):
            assert operation in ["add", "sub"]
            if operation == "add":
                num_combination_indices = self.add_num_combination_indices
                num_combination_mask = self.add_num_combination_mask
            elif operation == "sub":
                num_combination_indices = self.sub_num_combination_indices
                num_combination_mask = self.sub_num_combination_mask
            else:
                raise NotImplementedError

            # num_combination_indices: (size_composed_numbers, max_num_combinations, 2) for each number in composed numbers,
            # these are combinations (indices of passasge numbers) that combine (in order) using the operation and result
            # into this number. For example, num_combination_indices[i, :, :] is a combs=[NC, 2] array where
            # composed_number[i] = PassageNumber(combs[j, 0]) OP PassageNumber(combs[j, 1]) for all j.
            # These combinations are padded with -1
            masked_num_combination_indices = num_combination_indices * num_combination_mask

            # Making (B=1, seq_len=passage_numbers, dim=1) for batch index selection
            num_dist_1_uns = num_dist_1.unsqueeze(0).unsqueeze(2)
            num_dist_2_uns = num_dist_2.unsqueeze(0).unsqueeze(2)
            # B=1 unsqueezing
            masked_num_combination_indices_uns = masked_num_combination_indices.unsqueeze(0)

            # Indexing into num_dist_1 where indices are num_combination_indices[:, :, 0]
            selected_d_1 = allenutil.batched_index_select(
                target=num_dist_1_uns, indices=masked_num_combination_indices_uns[:, :, :, 0]
            )
            # Shape: (size_composed_numbers, max_num_combinations)
            selected_d_1 = selected_d_1.squeeze(0).squeeze(-1)
            selected_d_1 = selected_d_1 * num_combination_mask[:, :, 0].float()

            # Indexing into num_dist_2 where indices are num_combination_indices[:, :, 1]
            selected_d_2 = allenutil.batched_index_select(
                target=num_dist_2_uns, indices=masked_num_combination_indices_uns[:, :, :, 1]
            )
            # Shape: (size_composed_numbers, max_num_combinations)
            selected_d_2 = selected_d_2.squeeze(0).squeeze(-1)
            selected_d_2 = selected_d_2 * num_combination_mask[:, :, 1].float()

            # Shape: (number_support)
            expected_distribution = (selected_d_1 * selected_d_2).sum(dim=1)
            expected_distribution = clamp_distribution(expected_distribution)
        return expected_distribution

    def expected_date_year_difference(
        self, date_distribution_1: torch.FloatTensor, date_distribution_2: torch.FloatTensor
    ):
        """ Compute a distribution over possible year-differences by marginalizing over the year_differnces_mat.

            Parameters:
            -----------
            date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
            date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """
        with Profile("year-diff"):
            # Shape: (num_passage_dates, num_passage_dates)
            joint_dist = torch.matmul(date_distribution_1.unsqueeze(1), date_distribution_2.unsqueeze(0))

            # Shape: (number_year_differences, )
            year_differences_dist = torch.sum(self.year_differences_mat * joint_dist.unsqueeze(2), dim=(0, 1))

            # if torch.sum(year_differences_dist) > 1.0:
            # print("year dist")
            # print(f"{date_distribution_1} {date_distribution_1.sum()}")
            # print(f"{date_distribution_2} {date_distribution_2.sum()}")
            # print(f"{year_differences_dist} {year_differences_dist.sum()}")
            # print()

            year_differences_dist = clamp_distribution(year_differences_dist)

        return year_differences_dist

    def expected_date_comparison(self, date_distribution_1, date_distribution_2, comparison):
        """ Compute the boolean probability that date_1 > date_2 given distributions over passage_dates for each

        Parameters:
        -----------
        date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """
        with Profile("date-comp"):
            # Shape: (num_passage_dates, num_passage_dates)
            joint_dist = torch.matmul(date_distribution_1.unsqueeze(1), date_distribution_2.unsqueeze(0))

            if comparison == "greater":
                comparison_mat = self.date_gt_mat
            elif comparison == "lesser":
                comparison_mat = self.date_lt_mat
            else:
                comparison_mat = None
                raise NotImplementedError

            expected_bool = (comparison_mat * joint_dist).sum()
            expected_bool = clamp_distribution(expected_bool)
        return expected_bool

    def expected_num_comparison(self, distribution_1, distribution_2, comparison):
        """ Compute the boolean probability that date_1 > date_2 given distributions over passage_dates for each

        Parameters:
        -----------
        date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """
        with Profile("num-comp"):
            # Shape: (num_passage_nums, num_passage_nums)
            joint_dist = torch.matmul(distribution_1.unsqueeze(1), distribution_2.unsqueeze(0))

            if comparison == "greater":
                comparison_mat = self.num_gt_mat
            elif comparison == "lesser":
                comparison_mat = self.num_lt_mat
            else:
                comparison_mat = None
                raise NotImplementedError

            expected_bool = (comparison_mat * joint_dist).sum()
            expected_bool = clamp_distribution(expected_bool)
        return expected_bool

    def date_comparison(self, passage_attention_1, passage_attention_2, comparison: str, gold_date_groundings=None):

        date_distribution_1, _, d1_dist_entropy = self.compute_date_scores(passage_attention_1)
        date_distribution_2, _, d2_dist_entropy = self.compute_date_scores(passage_attention_2)

        bool1 = self.expected_date_comparison(date_distribution_1, date_distribution_2, comparison)
        bool2 = self.expected_date_comparison(date_distribution_2, date_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        if gold_date_groundings is not None:
            # These are one-hot vectors the size of passage_dates
            # These can be zeros as well, indicating the date grounding is unknown, in which case they shouldn't be
            # used for computing the auxiliary date-grounding loss
            gold_date_grounding_1, gold_date_grounding_2 = gold_date_groundings

            # Shape: (2, num_dates)
            gold_date_grounding_tensor = torch.cat(
                [gold_date_grounding_1.unsqueeze(0), gold_date_grounding_2.unsqueeze(0)], dim=0
            )

            # The groundings that contain a zero vector will yield max as zero, and hence mask = 0
            # For valid groundings, the max will be 1 (mask = 1) and will contain the correct idx from argmax
            date_grounding_mask, gold_date_idxs = torch.max(gold_date_grounding_tensor, dim=1)

            # Shape: [2, num_of_dates]
            stacked_date_distributions = torch.cat(
                [date_distribution_1.unsqueeze(0), date_distribution_2.unsqueeze(0)], dim=0
            )

            # Shape: (2) - Converting date_distribution to log-probabilties
            date_grounding_loss = F.nll_loss(
                input=(stacked_date_distributions + 1e-40).log(), target=gold_date_idxs, reduction="none"
            )
            date_grounding_loss = date_grounding_loss * date_grounding_mask
            date_grounding_loss = date_grounding_loss.mean()

        else:
            date_grounding_loss = 0.0

        # kl_div_neg_1_2 = -1 * F.kl_div(date_distribution_1, date_distribution_2, reduction='mean')
        # kl_div_neg_2_1 = -1 * F.kl_div(date_distribution_2, date_distribution_1, reduction='mean')

        aux_loss = date_grounding_loss
        # aux_loss += d1_dist_entropy + d2_dist_entropy

        return (date_distribution_1, date_distribution_2, bool1, bool2, average_passage_distribution, aux_loss)

    def num_comparison(self, passage_attention_1, passage_attention_2, comparison: str, gold_num_groundings=None):

        num_distribution_1, _, num1_entropy = self.compute_num_distribution(passage_attention_1)
        num_distribution_2, _, num2_entropy = self.compute_num_distribution(passage_attention_2)

        bool1 = self.expected_num_comparison(num_distribution_1, num_distribution_2, comparison)
        bool2 = self.expected_num_comparison(num_distribution_2, num_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        if gold_num_groundings is not None:
            # These are one-hot vectors the size of passage_dates
            # These can be zeros as well, indicating the date grounding is unknown, in which case they shouldn't be
            # used for computing the auxiliary date-grounding loss
            # TODO(nitish): Sometimes in validation we only get one from other questions, for example findNumber
            if len(gold_num_groundings) == 1:
                gold_num_groundings = [gold_num_groundings[0], gold_num_groundings[0]]
            gold_num_grounding_1, gold_num_grounding_2 = gold_num_groundings

            # Shape: (2, num_passage_nums)
            gold_num_grounding_tensor = torch.cat(
                [gold_num_grounding_1.unsqueeze(0), gold_num_grounding_2.unsqueeze(0)], dim=0
            )

            # The groundings that contain a zero vector will yield max as zero, and hence mask = 0
            # For valid groundings, the max will be 1 (mask = 1) and will contain the correct idx from argmax
            grounding_mask, gold_idxs = torch.max(gold_num_grounding_tensor, dim=1)

            # Shape: [2, num_of_dates]
            stacked_num_distributions = torch.cat(
                [num_distribution_1.unsqueeze(0), num_distribution_2.unsqueeze(0)], dim=0
            )
            # Shape: (2) - Converting date_distribution to log-probabilties
            num_grounding_loss = F.nll_loss(
                input=(stacked_num_distributions + 1e-40).log(), target=gold_idxs, reduction="none"
            )
            num_grounding_loss = num_grounding_loss * grounding_mask
            num_grounding_loss = num_grounding_loss.mean()

        else:
            num_grounding_loss = 0.0

        aux_loss = num_grounding_loss
        # aux_loss += num1_entropy + num2_entropy

        return (num_distribution_1, num_distribution_2, bool1, bool2, average_passage_distribution, aux_loss)

    # @predicate
    @predicate_with_side_args(["event_date_groundings"])
    def compare_date_lesser_than(
        self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention, event_date_groundings=None
    ) -> PassageAttention_answer:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            date_distribution_1,
            date_distribution_2,
            prob_date1_lesser,
            prob_date2_lesser,
            average_passage_distribution,
            aux_loss,
        ) = self.date_comparison(passage_attention_1, passage_attention_2, "lesser", event_date_groundings)

        average_passage_distribution = clamp_distribution(average_passage_distribution)
        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        if self._debug:
            if event_date_groundings:
                gold_date_1 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[0]), 3)
                gold_date_2 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[1]), 3)
            else:
                gold_date_1, gold_date_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1, self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2, self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            d1_lt_d2 = myutils.round_all(myutils.tocpuNPList(prob_date1_lesser), 3)
            d2_lt_d1 = myutils.round_all(myutils.tocpuNPList(prob_date2_lesser), 3)

            debug_value += (
                f"Pattn1: {pattn_vis_most_1}\n Date1: {date1}"
                + f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}"
                + f"\nP(D1 < D2): {d1_lt_d2}  P(D2 < D1): {d2_lt_d1}"
            )
            if gold_date_1:
                debug_value += f"\nGoldDates  Date1: {gold_date_1}  Date2: {gold_date_2}"

        return PassageAttention_answer(average_passage_distribution, loss=loss, debug_value=debug_value)

    # @predicate
    @predicate_with_side_args(["event_date_groundings"])
    def compare_date_greater_than(
        self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention, event_date_groundings=None
    ) -> PassageAttention_answer:
        """ In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2 """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            date_distribution_1,
            date_distribution_2,
            prob_date1_greater,
            prob_date2_greater,
            average_passage_distribution,
            aux_loss,
        ) = self.date_comparison(passage_attention_1, passage_attention_2, "greater", event_date_groundings)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        if self._debug:
            if event_date_groundings:
                gold_date_1 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[0]), 3)
                gold_date_2 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[1]), 3)
            else:
                gold_date_1, gold_date_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1, self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2, self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            d1_gt_d2 = myutils.round_all(myutils.tocpuNPList(prob_date1_greater), 3)
            d2_gt_d1 = myutils.round_all(myutils.tocpuNPList(prob_date2_greater), 3)

            debug_value += (
                f"Pattn1: {pattn_vis_most_1}\n Date1: {date1}"
                + f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}"
                + f"\nP(D1 > D2): {d1_gt_d2}  P(D2 > D1): {d2_gt_d1}"
            )

            if gold_date_1:
                debug_value += f"\nGoldDates  Date1: {gold_date_1}  Date2: {gold_date_2}"

        return PassageAttention_answer(average_passage_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["event_num_groundings"])
    def compare_num_lesser_than(
        self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention, event_num_groundings=None
    ) -> PassageAttention_answer:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            num_distribution_1,
            num_distribution_2,
            prob_num1_lesser,
            prob_num2_lesser,
            average_passage_distribution,
            aux_loss,
        ) = self.num_comparison(passage_attention_1, passage_attention_2, "lesser", event_num_groundings)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        if self._debug:
            if event_num_groundings:
                gold_num_1 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[0]), 3)
                gold_num_2 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[1]), 3)
            else:
                gold_num_1, gold_num_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1, self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2, self.metadata["passage_tokens"])

            num1 = myutils.round_all(myutils.tocpuNPList(num_distribution_1), 3)
            num2 = myutils.round_all(myutils.tocpuNPList(num_distribution_2), 3)
            d1_lt_d2 = myutils.round_all(myutils.tocpuNPList(prob_num1_lesser), 3)
            d2_lt_d1 = myutils.round_all(myutils.tocpuNPList(prob_num2_lesser), 3)

            debug_value += (
                f"Pattn1: {pattn_vis_most_1}\n Num1: {num1}"
                + f"\nPattn2: {pattn_vis_most_2}\n Num2: {num2}"
                + f"\nP(N1 < N2): {d1_lt_d2}  P(N2 < N1): {d2_lt_d1}"
            )
            if gold_num_1:
                debug_value += f"\nGoldNums Num1: {gold_num_1}  Num2: {gold_num_2}"

        return PassageAttention_answer(average_passage_distribution, loss=loss, debug_value=debug_value)

    # @predicate
    @predicate_with_side_args(["event_num_groundings"])
    def compare_num_greater_than(
        self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention, event_num_groundings=None
    ) -> PassageAttention_answer:
        """ In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2
        """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            num_distribution_1,
            num_distribution_2,
            prob_num1_greater,
            prob_num2_greater,
            average_passage_distribution,
            aux_loss,
        ) = self.num_comparison(passage_attention_1, passage_attention_2, "greater", event_num_groundings)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        if self._debug:
            if event_num_groundings:
                gold_num_1 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[0]), 3)
                gold_num_2 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[1]), 3)
            else:
                gold_num_1, gold_num_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1, self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2, self.metadata["passage_tokens"])

            num1 = myutils.round_all(myutils.tocpuNPList(num_distribution_1), 3)
            num2 = myutils.round_all(myutils.tocpuNPList(num_distribution_2), 3)
            d1_gt_d2 = myutils.round_all(myutils.tocpuNPList(prob_num1_greater), 3)
            d2_gt_d1 = myutils.round_all(myutils.tocpuNPList(prob_num2_greater), 3)

            debug_value += (
                f"Pattn1: {pattn_vis_most_1}\n num1: {num1}"
                + f"\nPattn2: {pattn_vis_most_2}\n num2: {num2}"
                + f"\nP(N1 > N2): {d1_gt_d2}  P(N2 > N1): {d2_gt_d1}"
            )

            if gold_num_1:
                debug_value += f"\nGoldNums  num1: {gold_num_1}  num2: {gold_num_2}"

        return PassageAttention_answer(average_passage_distribution, loss=loss, debug_value=debug_value)

    @predicate
    def year_difference(self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention) -> YearDifference:
        """ Given two passage spans, ground them to dates, and then return the difference between their years """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        date_distribution_1, _, d1_dist_entropy = self.compute_date_scores(passage_attention_1)
        date_distribution_2, _, d2_dist_entropy = self.compute_date_scores(passage_attention_2)

        # Shape: (number_of_year_differences, )
        year_difference_dist = self.expected_date_year_difference(date_distribution_1, date_distribution_2)

        loss = 0.0
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss
        # If we want to use an auxiliary entropy loss
        # loss += d1_dist_entropy + d2_dist_entropy

        debug_value = ""
        if self._debug:
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1, self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2, self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            year_diff_dist = myutils.round_all(myutils.tocpuNPList(year_difference_dist), 3)

            debug_value += (
                f"YearDiffDist: {year_diff_dist}\n"
                + f"\nPattn1: {pattn_vis_most_1}\n Date1: {date1}"
                + f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}"
            )

        return YearDifference(year_difference_dist=year_difference_dist, loss=loss, debug_value=debug_value)

    @predicate
    def year_difference_single_event(self, passage_attn: PassageAttention) -> YearDifference:
        """ Given a single passage span, find its start and end dates, then return the difference in years """

        passage_attention = passage_attn._value * self.passage_mask

        # DATE_1 is end since the difference is computed as DATE_1 - DATE_2
        date_distribution_1, _, d1_dist_entropy = self.compute_date_scores(passage_attention, date_type="end")
        date_distribution_2, _, d2_dist_entropy = self.compute_date_scores(passage_attention, date_type="start")

        # Shape: (number_of_year_differences, )
        year_difference_dist = self.expected_date_year_difference(date_distribution_1, date_distribution_2)

        loss = 0.0
        loss += passage_attn.loss
        # If we want to use an auxiliary entropy loss
        # loss += d1_dist_entropy + d2_dist_entropy

        debug_value = ""
        if self._debug:
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention, self.metadata["passage_tokens"])
            most_attended_spans = dlutils.mostAttendedSpans(passage_attention, self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            year_diff_dist = myutils.round_all(myutils.tocpuNPList(year_difference_dist), 3)

            debug_value += (
                f"YearDiffDist: {year_diff_dist}\n"
                + f"\nPattn1: {pattn_vis_most_1}"
                + f"\nMostAttendedSpans: {most_attended_spans}"
                + f"\n Date1: {date1}\n Date2: {date2}"
            )

        return YearDifference(year_difference_dist=year_difference_dist, loss=loss, debug_value=debug_value)

    @predicate
    def extractPassageSpanAnswer(self) -> PassageSpanAnswer:
        with Profile("pass-span-ans"):
            # Shape: (passage_length, encoded_dim)
            passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage

            # Shape: (passage_length, 2)
            passage_ans_startend_logits = self.parameters.oneshot_psa_startend_predictor(passage_repr)
            # Shape: (passage_length,)
            span_start_logits = passage_ans_startend_logits[:, 0]
            span_end_logits = passage_ans_startend_logits[:, 1]

            span_start_logits = allenutil.replace_masked_values(span_start_logits, self.passage_mask, -1e32)
            span_end_logits = allenutil.replace_masked_values(span_end_logits, self.passage_mask, -1e32)

            span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, self.passage_mask)
            span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, self.passage_mask)

            span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs, self.passage_mask, -1e32)
            span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs, self.passage_mask, -1e32)

            loss = 0.0

            debug_value = ""
            if self._debug:
                debug_value += f"OneShotPassageSpanAnswer extraction: nothing to visualize"

        return PassageSpanAnswer(
            passage_span_start_log_probs=span_start_log_probs,
            passage_span_end_log_probs=span_end_log_probs,
            start_logits=span_start_logits,
            end_logits=span_end_logits,
            loss=loss,
            debug_value=debug_value,
        )

    @predicate
    def find_passageSpanAnswer(self, passage_attention: PassageAttention_answer) -> PassageSpanAnswer:
        with Profile("find-span-ans"):
            passage_attn = passage_attention._value
            passage_mask = self.passage_mask

            # Shape: (passage_length)
            # passage_attn = passage_attn * self.passage_mask
            passage_attn = passage_attn * passage_mask

            scaled_attentions = [passage_attn * sf for sf in self.parameters.passage_attention_scalingvals]
            # Shape: (passage_length, num_scaling_factors)
            scaled_passage_attentions = torch.stack(scaled_attentions, dim=1)

            # Shape: (passage_lengths, hidden_dim)
            passage_span_hidden_reprs = self.parameters.passage_attention_to_span(
                scaled_passage_attentions.unsqueeze(0), passage_mask.unsqueeze(0)
            ).squeeze(0)

            # Shape: (passage_lengths, 2)
            passage_span_logits = self.parameters.passage_startend_predictor(passage_span_hidden_reprs)

            # Shape: (passage_length)
            span_start_logits = passage_span_logits[:, 0]
            span_end_logits = passage_span_logits[:, 1]

            span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask, -1e32)
            span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask, -1e32)

            span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, passage_mask)
            span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, passage_mask)

            span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs, passage_mask, -1e32)
            span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs, passage_mask, -1e32)

            loss = passage_attention.loss

            debug_value = ""
            if self._debug:
                _, pattn_vis_most = dlutils.listTokensVis(passage_attn, self.metadata["passage_tokens"])
                most_attended_spans = dlutils.mostAttendedSpans(passage_attn, self.metadata["passage_tokens"])
                # debug_value += f"Pattn: {pattn_vis_most}\n"
                debug_value += f"MostAttendedSpans: {most_attended_spans}"

        return PassageSpanAnswer(
            passage_span_start_log_probs=span_start_log_probs,
            passage_span_end_log_probs=span_end_log_probs,
            start_logits=span_start_logits,
            end_logits=span_end_logits,
            loss=loss,
            debug_value=debug_value,
        )

    @predicate
    def passageAttn2Count(self, passage_attention: PassageAttention) -> CountNumber:
        with Profile("count-mod"):
            passage_attn = passage_attention._value

            # Shape: (passage_length)
            passage_attn = passage_attn * self.passage_mask

            scaled_attentions = [passage_attn * sf for sf in self.parameters.passage_attention_scalingvals]
            # Shape: (passage_length, num_scaling_factors)
            scaled_passage_attentions = torch.stack(scaled_attentions, dim=1)

            # Shape: (passage_length, hidden_dim)
            count_hidden_repr = self.parameters.passage_attention_to_count(
                scaled_passage_attentions.unsqueeze(0), self.passage_mask.unsqueeze(0)
            ).squeeze(0)

            # Shape: (passage_length, 1)
            passage_token_logits = self.parameters.passage_count_hidden2logits(count_hidden_repr)
            # Shape: (passage_length)
            passage_token_logits = passage_token_logits.squeeze(1)

            passage_token_sigmoids = torch.sigmoid(passage_token_logits)
            passage_token_sigmoids = passage_token_sigmoids * self.passage_mask

            count_mean = torch.sum(passage_token_sigmoids)
            variance = 0.5

            loss = 0
            loss += passage_attention.loss
            if count_mean > 10.0:
                extra_loss = F.mse_loss(count_mean, allenutil.move_to_device(torch.tensor(9.0), cuda_device=self.device_id))
                loss += extra_loss
                logger.info(f"CountMean: {count_mean} ExtraLoss: {extra_loss}")
                count_mean = allenutil.move_to_device(torch.tensor(9.0), cuda_device=self.device_id)

            # Shape: (num_count_values, )
            l2_by_vsquared = torch.pow(self.countvals - count_mean, 2) / (2 * variance * variance)
            exp_val = torch.exp(-1 * l2_by_vsquared) + 1e-30
            count_distribution = exp_val / (torch.sum(exp_val))

            count_distribution = clamp_distribution(count_distribution)

            debug_value = ""
            if self._debug:
                countdist = myutils.round_all(myutils.tocpuNPList(count_distribution), 3)
                psigms, pattn_vis_most = dlutils.listTokensVis(passage_token_sigmoids, self.metadata["passage_tokens"])
                debug_value += f"CountDist: {countdist}"
                debug_value += f"CountMean: {count_mean}"
                debug_value += f"\nPSigms: {psigms}"

        return CountNumber(count_number_dist=count_distribution, loss=loss, debug_value=debug_value)

    def number_add_sub_module(self, number_1: PassageNumber, number_2: PassageNumber, add_sub: str):
        assert add_sub in ["add", "sub"]
        numberdist_1 = number_1._value
        numberdist_2 = number_2._value

        # Shape: (size_composed_numbers, )
        composed_number_dist = self.expected_number_addsub(
            num_dist_1=numberdist_1, num_dist_2=numberdist_2, operation=add_sub
        )

        loss = 0.0
        loss += number_1.loss + number_2.loss

        debug_value = ""
        if self._debug:
            composed_number_dist_list = myutils.round_all(myutils.tocpuNPList(composed_number_dist), 3)
            topknumdiff = dlutils.topProbMassElems(
                attention=composed_number_dist, support=self.composed_numbers, k=5
            )
            topknum1 = dlutils.topProbMassElems(attention=numberdist_1, support=self.passage_num_values, k=5)
            topknum2 = dlutils.topProbMassElems(attention=numberdist_2, support=self.passage_num_values, k=5)

            debug_value += (
                f"ComposedNumberDist: {composed_number_dist_list}\n"
                + f"Top-num-diff: {topknumdiff}\n"
                + f"\n Top-Num1: {topknum1}"
                + f"\n Top-Num2: {topknum2}"
            )

        return composed_number_dist, loss, debug_value

    @predicate
    def passagenumber_difference(
        self, passage_number_1: PassageNumber, passage_number_2: PassageNumber
    ) -> ComposedNumber:
        """ Find the expected difference between two number distributions. """

        number_difference_dist, loss, debug_value = self.number_add_sub_module(
            number_1=passage_number_1, number_2=passage_number_2, add_sub="sub"
        )

        return ComposedNumber(composed_number_dist=number_difference_dist, loss=loss, debug_value=debug_value)

    @predicate
    def passagenumber_addition(
        self, passage_number_1: PassageNumber, passage_number_2: PassageNumber
    ) -> ComposedNumber:
        """ Find the expected sum of two number distributions. """
        number_addition_dist, loss, debug_value = self.number_add_sub_module(
            number_1=passage_number_1, number_2=passage_number_2, add_sub="add"
        )

        return ComposedNumber(composed_number_dist=number_addition_dist, loss=loss, debug_value=debug_value)

    """
    @predicate
    def find_questionSpanAnswer(self, passage_attention: PassageAttention_answer) -> QuestionSpanAnswer:
        passage_attn = passage_attention._value
        # passage_attn = passage_attention

        # Shape: (passage_length)
        passage_attn = (passage_attn * self.passage_mask)


        # Shape: (question_length)
        question_attn = torch.sum(self.passage_question_attention * passage_attn.unsqueeze(1), dim=0)
        question_attn = question_attn * self.question_mask

        scaled_attentions = [question_attn * sf for sf in self.parameters.passage_attention_scalingvals]
        # Shape: (question_length, num_scaling_factors)
        scaled_question_attentions = torch.stack(scaled_attentions, dim=1)

        # Shape: (question_length, hidden_dim)
        ques_span_hidden_reprs = self.parameters.question_attention_to_span(scaled_question_attentions.unsqueeze(0),
                                                                            self.question_mask.unsqueeze(0)).squeeze(0)

        # Shape: (passage_lengths, 2)
        question_span_logits = self.parameters.question_startend_predictor(ques_span_hidden_reprs)

        # Shape: (passage_length)
        span_start_logits = question_span_logits[:, 0]
        span_end_logits = question_span_logits[:, 1]

        span_start_logits = allenutil.replace_masked_values(span_start_logits, self.question_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, self.question_mask, -1e32)

        span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, self.question_mask)
        span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, self.question_mask)

        span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs,
                                                               self.question_mask, -1e32)
        span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs,
                                                             self.question_mask, -1e32)

        loss = passage_attention.loss

        debug_value = ""
        if self._debug:
            _, pattn_vis_most = dlutils.listTokensVis(question_attn, self.metadata["question_tokens"])
            debug_value += f"Qattn: {pattn_vis_most}"

        return QuestionSpanAnswer(question_span_start_log_probs=span_start_log_probs,
                                  question_span_end_log_probs=span_end_log_probs,
                                  start_logits=span_start_logits,
                                  end_logits=span_end_logits,
                                  loss=loss,
                                  debug_value=debug_value)
    """

    def max_number_distribution(self, num_dist: torch.FloatTensor):
        cum_dist = num_dist.cumsum(0)
        cum_dist_n = cum_dist ** self.max_samples
        maximum_distribution = cum_dist_n - torch.cat([cum_dist_n.new_zeros(1), cum_dist_n[:-1]])
        maximum_distribution = clamp_distribution(maximum_distribution)
        return maximum_distribution

    def min_number_distribution(self, num_dist: torch.FloatTensor):
        cumulative_distribution_function = num_dist.cumsum(0)
        # P(x>=i) = 1 - (P(x<=i) - P(x=i))
        inverse_cumdist = 1 - cumulative_distribution_function + num_dist
        inverse_cumdist_n = inverse_cumdist ** self.max_samples
        inverse_cumdist_n_shifted = torch.cat([inverse_cumdist_n[1:], inverse_cumdist_n.new_zeros(1)])
        minimum_distribution = inverse_cumdist_n - inverse_cumdist_n_shifted
        minimum_distribution = clamp_distribution(minimum_distribution)
        return minimum_distribution

    @predicate_with_side_args(["event_num_groundings"])
    def find_PassageNumber(self, passage_attention: PassageAttention, event_num_groundings=None) -> PassageNumber:
        # This comes as a list of groundings; even though for this action there's just one
        # This can be completely-zero indicating masked. In that case, don't compute loss
        if event_num_groundings is not None:
            number_grounding_supervision = event_num_groundings[0]
        else:
            number_grounding_supervision = None

        passage_attn = passage_attention._value
        # Shape: (passage_length)
        passage_attn = passage_attn * self.passage_mask

        number_distribution, _, num_dist_entropy = self.compute_num_distribution(passage_attention=passage_attn)

        loss = 0.0
        grounding_loss = 0.0
        if number_grounding_supervision is not None:
            grounding_mask = (torch.sum(number_grounding_supervision) > 0).float()
            log_probs = torch.log(number_distribution + 1e-40) * number_grounding_supervision
            log_likelihood = torch.sum(log_probs)  # Want all grounded numbers to be high, hence prod of probs
            grounding_loss = -1 * grounding_mask * log_likelihood

        loss += grounding_loss
        loss += passage_attention.loss

        debug_value = ""
        if self._debug:
            number_dist = myutils.round_all(myutils.tocpuNPList(number_distribution), 3)
            topk_numdist = dlutils.topProbMassElems(attention=number_distribution, support=self.passage_num_values, k=5)
            if number_grounding_supervision is not None:
                num_grounding_sup = myutils.round_all(myutils.tocpuNPList(number_grounding_supervision), 3)
            else:
                num_grounding_sup = None
            # _, pattn_vis_most = dlutils.listTokensVis(passage_attn, self.metadata["passage_tokens"])
            top_spans = dlutils.mostAttendedSpans(passage_attn, self.metadata["passage_tokens"])
            debug_value += f"PassageNumber: {number_dist}"
            debug_value += f"\ntopk-num-dist: {topk_numdist}"
            debug_value += f"\nGoldNum: {num_grounding_sup}"
            debug_value += f"\ninput-pattn-top-spans: {top_spans}"
        return PassageNumber(passage_number_dist=number_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention"])
    def ground_ImplicitNumber(self, question_attention: Tensor) -> PassageNumber:
        number_distribution = self.compute_implicitnum_distribution(question_attention=question_attention)
        loss = 0.0
        debug_value = ""
        if self._debug:
            number_dist = myutils.round_all(myutils.tocpuNPList(number_distribution), 3)
            topk_numdist = dlutils.topProbMassElems(attention=number_distribution, support=self.passage_num_values, k=5)
            debug_value += f"PassageNumber: {number_dist}"
            debug_value += f"\ntopk-num-dist: {topk_numdist}"
            qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(
                question_attention, self.metadata["question_tokens"]
            )
            debug_value += f"input-qattn: {qattn_vis_complete}"
        return PassageNumber(passage_number_dist=number_distribution, loss=loss, debug_value=debug_value)


    @predicate_with_side_args(["event_num_groundings"])
    def minNumPattn(self, passage_attention: PassageAttention, event_num_groundings=None) -> PassageAttention:
        minmax_num_pattn, loss, debug_value = self.minmaxNumPattn_module(
            passage_attention=passage_attention, min_max_op="min", event_num_groundings=event_num_groundings
        )

        return PassageAttention(passage_attention=minmax_num_pattn, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["event_num_groundings"])
    def maxNumPattn(self, passage_attention: PassageAttention, event_num_groundings=None) -> PassageAttention:
        minmax_num_pattn, loss, debug_value = self.minmaxNumPattn_module(
            passage_attention=passage_attention, min_max_op="max", event_num_groundings=event_num_groundings
        )

        return PassageAttention(passage_attention=minmax_num_pattn, loss=loss, debug_value=debug_value)

    def minmaxNumPattn_module(self, passage_attention: PassageAttention, min_max_op: str, event_num_groundings=None):
        assert min_max_op in ["min", "max"]
        if event_num_groundings is not None:
            num_grounding_supervision = event_num_groundings[0]
        else:
            num_grounding_supervision = None

        pattn = passage_attention._value * self.passage_mask
        # Computing for debugging and aux loss purposes
        inputpattn_num_dist, _, _ = self.compute_num_distribution(pattn)
        minmax_num_pattn, input_numtoken_probs, minmax_numtoken_probs = self.pattn_for_minmaxNum(pattn=pattn,
                                                                                                 max_min=min_max_op)

        loss = 0.0
        if num_grounding_supervision is not None:
            grounding_mask = (torch.sum(num_grounding_supervision) > 0).float()
            if grounding_mask > 0:
                # Number distribution for input pattn
                log_probs = torch.log(inputpattn_num_dist + 1e-40) * num_grounding_supervision
                log_likelihood = torch.sum(log_probs)  # Want all grounded numbers to be high, hence prod of probs
                grounding_loss = -1 * grounding_mask * log_likelihood
                loss += grounding_loss
        loss += passage_attention.loss

        debug_value = ""
        if self._debug:
            minmax_num_dist, _, _ = self.compute_num_distribution(minmax_num_pattn)
            input_token_probs = myutils.round_all(myutils.tocpuNPList(input_numtoken_probs), 3)
            minmax_token_probs = myutils.round_all(myutils.tocpuNPList(minmax_numtoken_probs), 3)

            input_topknums = dlutils.topProbMassElems(attention=inputpattn_num_dist, support=self.passage_num_values)
            topknums = dlutils.topProbMassElems(attention=minmax_num_dist, support=self.passage_num_values)
            if num_grounding_supervision is not None:
                num_grounding_sup = myutils.round_all(myutils.tocpuNPList(num_grounding_supervision), 3)
            else:
                num_grounding_sup = None
            min_pattn_comp, _ = dlutils.listTokensVis(minmax_num_pattn, self.metadata["passage_tokens"])
            topspans = dlutils.mostAttendedSpans(minmax_num_pattn, self.metadata["passage_tokens"])
            debug_value += f"{min_max_op}-pattn-module"
            debug_value += f"\noutput-{min_max_op}-num-pattn-most-attended-spans: {topspans}"
            debug_value += f"\ninput-numtoken-probs: {input_token_probs}"
            debug_value += f"\ninput-num-dist: {input_topknums}"
            debug_value += f"\nminmax-numtoken-probs: {minmax_token_probs}"
            debug_value += f"\ntopk-input-num-dist: {topknums}"
            debug_value += f"\nGoldNum: {num_grounding_sup}"
        return minmax_num_pattn, loss, debug_value

    def pattn_for_minmaxNum(
        self, pattn: torch.FloatTensor, max_min: str
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """ Re-distribute the passage-attention based on a max number-token distribution
            The idea here is to find a `max-number-token` passage attention, i.e. given a distribution over
            passage-tokens, and for each token a distribution over numbers, find the passage-attention that contains,
            for each token, the probability that the number-associated with that token is the max.

            The output passage-distribution can be used compute an expected-number-distribution, which is the
            expected value under the distribution over max-events. For eg, "How many yards was the longest TD"
            Given the passage-attention for "TD", return the computed attention for P("longest TD").

            This is a little tricky to compute; the high level steps are:
            1. Compute an expected token-number-distribution. (this is a masked passage-length attention)
            2. Arrange the token-number-probs in an ascending order for the max
            3. Compute the max-distribution and re-arrange the numbers in order
            4. Re-map the token-number probs into a masked vector.
            5. ....

            If we know the number-grounding-supervision for the input passage-attention events, we can also compute
            an auxiliary loss here.
        """
        with Profile("pattn-minmax"):
            # Shape: (passage_length, passage_length) -- each row is a number-token-distribution
            pattn_times_numbertokenprobs = self.passage_passage_token2num_alignment * pattn.unsqueeze(1)

            # Shape: (passage_length, num_of_number_tokens) -- These are now in sorted order
            pattn_weighted_numbertoken_probs = pattn_times_numbertokenprobs[:, self.passage_number_sortedtokenidxs]

            # Shape: (num_of_number_tokens, ) -- the probability of the number tokens in sorted order
            only_expected_numbertoken_probs = pattn_weighted_numbertoken_probs.sum(0)
            if max_min == "max":
                only_numbertoken_minmaxprobs_sorted = self.max_number_distribution(only_expected_numbertoken_probs)
            elif max_min == "min":
                only_numbertoken_minmaxprobs_sorted = self.min_number_distribution(only_expected_numbertoken_probs)
            else:
                raise NotImplementedError

            # For each (token i, number j), using pattn_weighted_numbertoken_probs[i, j] as the weight,
            # Total weight for numbertoken as pattn_weighted_numbertoken_probs.sum(0), redistribute the minmax-number-prob
            # to all tokens
            # Shape: (1, num_of_number_tokens)
            total_weight_to_numbertoken = pattn_weighted_numbertoken_probs.sum(0, keepdim=True)
            # Shape: (passage_length, num_of_number_tokens) - each entry here is (pattn * numberprob * number-max-prob)
            maxprob_times_pattn_numbertokenprob = (
                pattn_weighted_numbertoken_probs * only_numbertoken_minmaxprobs_sorted.unsqueeze(0)
            )
            # Divide each entry above by \sum_tokens pattn * numberprob --
            # This is the new-distributed weight of number-max-prob on the i-th token, j-th number.
            # Now marginalize over numbers, to get the new-passage-attention
            new_pattn = (maxprob_times_pattn_numbertokenprob / total_weight_to_numbertoken).sum(1)

            new_pattn = clamp_distribution(new_pattn)

        return new_pattn, only_expected_numbertoken_probs, only_numbertoken_minmaxprobs_sorted


if __name__ == "__main__":
    dl = get_empty_language_object()
    print("All possible productions")
    print(dl.all_possible_productions())

    print("\nAllGet nonterminal productions")
    print(dl.get_nonterminal_productions())

    # print(spanans.__class__.__name__)
