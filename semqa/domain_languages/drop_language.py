import logging
from typing import List, Tuple, Any, Union, Optional, Dict
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


class Output:
    def __init__(self, output_type: str, values: Union[List[float], torch.Tensor], label: str):
        """ This class is used to represent some output produced by a module for visualization.

        Parameters:
        -----------
        output_type: `str`
            Is the type of the output produced. E.g. passage attention, question attention, number dist, etc.
            One of `passage`, `question`, `count`, `numbers`, `dates`, `year_diffs`, `composed_numbers`, `count`
        values: `List[float]`
            Attention value over the relevant support
        label: `str`
            The label for the output which helps distinguish between multiple outputs a module produces
        """
        assert output_type in ['passage', 'question', 'numbers', 'dates', 'year_diffs', 'composed_numbers',
                               'count'], "Incorrect output type: {}".format(output_type)
        self.output_type = output_type
        self.values = values
        self.label = label

    def to_json(self):
        json_dict = {
            "output_type": self.output_type,
            "values": self.values,
            "label": self.label
        }
        return json_dict


def output_from_dict(json_dict: Dict):
    return Output(output_type=json_dict["output_type"], values = json_dict["values"], label = json_dict["label"])


def aux_symbol_loss(gold_entidxs: Union[None, List[int]], attention_distribution: torch.Tensor):
    """ Compute NLL loss between a multi-label one-zero gold entidxs list and a probability distribution.

     gold_entidxs: `Union[None, List[int]]`
        Contains indices for elements that should have high probability
    attention_distribution: `torch.FloatTensor`
        1-d tensor containing a 'softmax-ed' probability distribution


    -ve sum of log-probs for gold elements
    Loss = -1.0 * \sum_i [(gold_entidxs[i]==1) * log(attention_distribution[i])]
    """
    loss = 0.0
    if gold_entidxs:  # This can be None or empty; this should take care of both
        gold_entidxs: List[int] = gold_entidxs  # This is a list of gold-entidxs that need to be predicted
        relevant_logprobs = [torch.log(attention_distribution[x] + 1e-40) for x in gold_entidxs]
        loss = -1.0 * sum(relevant_logprobs)
    return loss


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
            passage_attn,
            bio_logprobs=None,
            loss=0.0,
            debug_value="",
    ) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.passage_span_start_log_probs = passage_span_start_log_probs
        self.passage_span_end_log_probs = passage_span_end_log_probs
        self._value = (passage_span_start_log_probs, passage_span_end_log_probs)
        self.passage_attn = passage_attn
        self.bio_logprobs = bio_logprobs
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


class QuestionNumber:
    def __init__(self, ques_number_dist, loss=0.0, debug_value=""):
        self._value = ques_number_dist
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
            encoded_passage: Tensor,
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
            parameters: ExecutorParameters,
            modeled_passage: Tensor = None,
            start_types=None,
            device_id: int = -1,
            max_samples=10,
            metadata={},
            debug=False,
    ) -> None:

        if start_types is None:
            start_types = {PassageSpanAnswer, YearDifference, PassageNumber, ComposedNumber, CountNumber}
            # QuestionSpanAnswer - could be one

        super().__init__(start_types=start_types)

        if encoded_passage is None:
            return

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

        # Shape: (passage_length, )
        self.passage_tokenidx2dateidx = passage_tokenidx2dateidx.long()
        passage_tokenidx2dateidx_mask = self.passage_tokenidx2dateidx > -1
        self.passage_datetokens_mask_bool = self.passage_tokenidx2dateidx > -1
        self.passage_datetokens_mask_long = passage_tokenidx2dateidx_mask.long()
        self.passage_datetokens_mask_float = passage_tokenidx2dateidx_mask.float()
        # List[Date] - number of unique dates in the passage
        self.passage_date_values: List[Date] = passage_date_values
        self.num_passage_dates = len(self.passage_date_values)

        # Shape: (passage_length, )
        self.passage_tokenidx2numidx = passage_tokenidx2numidx.long()
        passage_tokenidx2numidx_mask = self.passage_tokenidx2numidx > -1
        self.passage_numtokens_mask_bool = self.passage_tokenidx2numidx > -1
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

        self.inwindow_mask, self.outwindow_mask = dlutils.masking_blockdiagonal(
            passage_length=self.passage_length, window=15, device_id=self.device_id
        )

        # This is a list of list, where each list element corresponds to one program execution. Each element stores
        # details related to the module execution as added by the modules below.
        # drop_parse_base._get_denotations appends a new list to this before executing a program to start a new program
        # Each module below can then append info into self.modules_debug_info[-1] -- to add to the latest program, i.e.
        # the one getting executed.
        self.modules_debug_info = []

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
        values_is_sorted = all(values[i] <= values[i + 1] for i in range(0, len(values) - 1))
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

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def select_passage(self, question_attention: Tensor, weighted_question_vector: Tensor) -> PassageAttention:
        with Profile("find-pattn"):
            weighted_question_vector_ex = weighted_question_vector.unsqueeze(0)
            passage_matrix = self.encoded_passage.unsqueeze(0)
            # (passage_length, )  s_i = sim(q, p_i)
            passage_similarity = self.parameters.qvec_to_passage_attention(weighted_question_vector_ex,
                                                                           passage_matrix).squeeze(0)
            # p_attn = softmax(s_i)
            passage_attention = allenutil.masked_softmax(passage_similarity, mask=self.passage_mask.bool(),
                                                         memory_efficient=True)
            passage_attention = clamp_distribution(passage_attention)

            # if self._debug:
            qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
            pattn_output = Output(output_type="passage", values=passage_attention, label="passage_attn")
            # debug_info_dict is a dictionary from {module_name: List[Output]}
            debug_info_dict = {"select_passage": [qattn_output, pattn_output]}
            self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(passage_attention, debug_value="")

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def filter_passage(
            self, passage_attention: PassageAttention, question_attention: Tensor,
            weighted_question_vector: Tensor
    ) -> PassageAttention:
        with Profile("filter-attn"):
            passage_attn: Tensor = passage_attention._value

            # Shape: (passage_length, encoded_dim)
            passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage

            # Shape: (1, num_sents, 2 * passage_dim)
            sentence_start_end_repr = self.parameters._endpoint_span_extractor(
                sequence_tensor=passage_repr.unsqueeze(0),
                span_indices=self.passage_sentence_boundaries_masked.unsqueeze(0))

            sentence_start_end_repr = (sentence_start_end_repr[:, :, 0:self.passage_encoding_dim] +
                                       sentence_start_end_repr[:, :, self.passage_encoding_dim:])

            # Shape: (1, 1, num_sents)
            sentence_logits_unsqueezed = self.parameters.filter_matrix_attention(
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
                most_attended_spans = dlutils.mostAttendedSpans(filtered_passage_attention,
                                                                self.metadata["passage_tokens"])
                debug_value += f"\nPattn: {pattn_vis_complete}"
                debug_value += f"\nMostAttended: {most_attended_spans}"

                qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
                pattn_output = Output(output_type="passage", values=filtered_passage_attention, label="filtered_pattn")
                debug_info_dict = {"filter_passage": [qattn_output, pattn_output]}
                # debug_info_dict = {"filter": {"passage": filtered_passage_attention,
                #                               "passage_input": passage_attn,
                #                               "question": question_attention}}
                self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(filtered_passage_attention, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def project_passage(
            self, passage_attention: PassageAttention, question_attention: Tensor, weighted_question_vector: Tensor
    ) -> PassageAttention:
        with Profile("relocate-attn"):
            passage_attn: Tensor = passage_attention._value
            passage_attn = passage_attn * self.passage_mask

            # Shape: (passage_length, encoded_dim)
            passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage
            question_repr = weighted_question_vector.unsqueeze(0).expand(*passage_repr.size())
            # Shape: (passage_length, encoded_dim + encoded_dim)
            # q_p_repr = weighted_question_vector.unsqueeze(0) + passage_repr
            q_p_repr = torch.cat([question_repr, passage_repr], dim=1)

            # Shape: (passage_length, passage_length)
            passage_passage_relocate_similarity = self.parameters.relocate_matrix_attention(
                q_p_repr.unsqueeze(0), passage_repr.unsqueeze(0)
            ).squeeze(0)

            square_mask_bool = (self.passage_mask.unsqueeze(0) * self.passage_mask.unsqueeze(1)).bool()

            # Shape: (passage_length, passage_length)
            p_to_p_relocate_attention = allenutil.masked_softmax(
                passage_passage_relocate_similarity, mask=square_mask_bool, dim=-1,
                memory_efficient=True
            )
            p_to_p_relocate_attention = p_to_p_relocate_attention * self.passage_mask.unsqueeze(1)

            # Shape: (passage_length, )
            relocate_attn = (p_to_p_relocate_attention * passage_attn.unsqueeze(1)).sum(0)
            relocate_attn = clamp_distribution(relocate_attn)

            inwindow_aux_loss = dlutils.aux_window_loss(ptop_attention=p_to_p_relocate_attention,
                                                        passage_mask=self.passage_mask,
                                                        inwindow_mask=self.inwindow_mask)

            loss = passage_attention.loss
            loss += inwindow_aux_loss

            debug_value = ""
            qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
            pattn_output = Output(output_type="passage", values=relocate_attn, label="project_pattn")
            debug_info_dict = {"project_passage": [qattn_output, pattn_output]}
            self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(relocate_attn, loss=loss, debug_value=debug_value)

    # New Date Distribtion
    def compute_date_scores(self, passage_attention: Tensor, weighted_question_vector: Tensor, date_type: str = None):
        """ Given a passage over passage token2date attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) dates in the passage.

            Using the token_attention from the passage_attention, find the expected input_token2date_token
            distribution - weigh each row in passage_passage_token2date_attention and sum
            Use this date_token attention as scores for the dates they refer to. Since each date can be referred
            to in multiple places, we use scatter_add_ to get the total_score.
            Softmax over these scores is the date dist.
        """
        if date_type is None:
            qp_to_passagedate_attention = self.parameters.qp_to_numdate_attention
        elif date_type == "start":
            qp_to_passagedate_attention = self.parameters.qp_to_startdate_attention
        elif date_type == "end":
            qp_to_passagedate_attention = self.parameters.qp_to_enddate_attention
        else:
            raise NotImplementedError

        # Shape: (passage_length, encoding_dim)
        passage_repr = self.encoded_passage
        question_repr = weighted_question_vector.unsqueeze(0).expand(*self.encoded_passage.size())

        q_p_repr = torch.cat([passage_repr, question_repr], dim=1)

        passage_similarity = qp_to_passagedate_attention(q_p_repr.unsqueeze(0),
                                                         self.encoded_passage.unsqueeze(0)).squeeze(0)
        passage_similarity = passage_similarity * self.passage_mask.unsqueeze(0) * self.passage_mask.unsqueeze(1)
        # Masking non-date tokens in dim=1
        passage_similarity = passage_similarity * self.passage_datetokens_mask_float.unsqueeze(0)

        p2p_date_alignment = allenutil.masked_softmax(passage_similarity,
                                                      self.passage_datetokens_mask_bool.unsqueeze(0),
                                                      dim=-1, memory_efficient=True)

        passage_date_token_probs = torch.sum(p2p_date_alignment * passage_attention.unsqueeze(1), dim=0)

        date_win_loss = dlutils.aux_window_loss(ptop_attention=p2p_date_alignment,
                                                passage_mask=self.passage_mask,
                                                inwindow_mask=self.inwindow_mask,
                                                outwindow_mask=self.outwindow_mask,
                                                p_tokensymbol_mask_float=self.passage_datetokens_mask_float)

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

        return date_distribution, passage_date_token_probs, date_win_loss

    # New Num Distribution by first computing a number-distribution for each passage-token
    def compute_num_distribution(self, passage_attention: Tensor, weighted_question_vector: Tensor, pdb=False):
        """ Given a passage over passage token2num attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) nums in the passage.
            See compute_date_distribution for details
        """

        if pdb:
            import pdb
            pdb.set_trace()

        # Shape: (passage_length, encoding_dim)
        passage_repr = self.encoded_passage
        question_repr = weighted_question_vector.unsqueeze(0).expand(*self.encoded_passage.size())

        q_p_repr = torch.cat([passage_repr, question_repr], dim=1)

        passage_similarity = self.parameters.qp_to_numdate_attention(q_p_repr.unsqueeze(0),
                                                                     self.encoded_passage.unsqueeze(0)).squeeze(0)
        passage_similarity = passage_similarity * self.passage_mask.unsqueeze(0) * self.passage_mask.unsqueeze(1)
        passage_similarity = passage_similarity * self.passage_numtokens_mask_float.unsqueeze(0)

        p2p_num_alignment = allenutil.masked_softmax(passage_similarity,
                                                     self.passage_numtokens_mask_bool.unsqueeze(0),
                                                     dim=-1, memory_efficient=True)

        passage_number_token_probs = torch.sum(p2p_num_alignment * passage_attention.unsqueeze(1), dim=0)

        num_win_loss = dlutils.aux_window_loss(ptop_attention=p2p_num_alignment,
                                               passage_mask=self.passage_mask,
                                               inwindow_mask=self.inwindow_mask,
                                               outwindow_mask=self.outwindow_mask,
                                               p_tokensymbol_mask_float=self.passage_numtokens_mask_float)

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
        # num_distribution_entropy = -1 * torch.sum(num_distribution * torch.log(num_distribution + 1e-40))

        return num_distribution, passage_number_token_probs, p2p_num_alignment, num_win_loss


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
        # with Profile("num-add-sub"):
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

    def date_comparison(self, passage_attention_1, passage_attention_2, weighted_question_vector: Tensor,
                        comparison: str, date1_entidxs=None, date2_entidxs=None):

        date_distribution_1, passage_datetoken_prob_1, date_winloss1 = self.compute_date_scores(
            passage_attention_1, weighted_question_vector)
        date_distribution_2, passage_datetoken_prob_2, date_winloss2 = self.compute_date_scores(
            passage_attention_2, weighted_question_vector)

        bool1 = self.expected_date_comparison(date_distribution_1, date_distribution_2, comparison)
        bool2 = self.expected_date_comparison(date_distribution_2, date_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        loss1 = aux_symbol_loss(date1_entidxs, date_distribution_1)
        loss2 = aux_symbol_loss(date2_entidxs, date_distribution_2)
        date_grounding_loss = loss1 + loss2

        aux_loss = date_grounding_loss + date_winloss1 + date_winloss2

        return (date_distribution_1, date_distribution_2, bool1, bool2,
                passage_datetoken_prob_1, passage_datetoken_prob_2, average_passage_distribution, aux_loss)

    def num_comparison(self, passage_attention_1, passage_attention_2, weighted_question_vector: Tensor,
                       comparison: str, num1_entidxs=None, num2_entidxs=None):

        num_distribution_1, passage_numtoken_prob_1, _, num_winloss1 = self.compute_num_distribution(
            passage_attention_1, weighted_question_vector)
        num_distribution_2, passage_numtoken_prob_2, _, num_winloss2 = self.compute_num_distribution(
            passage_attention_2, weighted_question_vector)

        bool1 = self.expected_num_comparison(num_distribution_1, num_distribution_2, comparison)
        bool2 = self.expected_num_comparison(num_distribution_2, num_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        loss1 = aux_symbol_loss(num1_entidxs, num_distribution_1)
        loss2 = aux_symbol_loss(num2_entidxs, num_distribution_2)
        num_grounding_loss = loss1 + loss2

        aux_loss = num_grounding_loss + num_winloss1 + num_winloss2

        return (num_distribution_1, num_distribution_2, bool1, bool2,
                passage_numtoken_prob_1, passage_numtoken_prob_2,
                average_passage_distribution, aux_loss)

    # @predicate
    @predicate_with_side_args(["question_attention", "weighted_question_vector", "date1_entidxs", "date2_entidxs"])
    def compare_date_lt(
            self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
            date1_entidxs=None, date2_entidxs=None
    ) -> PassageAttention:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            date_distribution_1,
            date_distribution_2,
            prob_date1_lesser,
            prob_date2_lesser,
            passage_datetoken_prob_1,
            passage_datetoken_prob_2,
            average_passage_distribution,
            aux_loss,
        ) = self.date_comparison(passage_attention_1, passage_attention_2, weighted_question_vector,
                                 "lesser", date1_entidxs, date2_entidxs)

        average_passage_distribution = clamp_distribution(average_passage_distribution)
        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        # if self._debug:
        date1_token = myutils.round_all(myutils.tocpuNPList(passage_datetoken_prob_1), 3)
        date2_token = myutils.round_all(myutils.tocpuNPList(passage_datetoken_prob_2), 3)
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        date1_output = Output(output_type="dates", values=date_distribution_1, label="date_1")
        date2_output = Output(output_type="dates", values=date_distribution_2, label="date_2")
        pattn_output = Output(output_type="passage", values=average_passage_distribution, label="avg_pattn")
        debug_info_dict = {"compare_date_lt": [qattn_output, date1_output, date2_output, pattn_output]}

        self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(average_passage_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector", "date1_entidxs", "date2_entidxs"])
    def compare_date_gt(
            self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
            date1_entidxs=None, date2_entidxs=None
    ) -> PassageAttention:
        """ In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2 """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            date_distribution_1,
            date_distribution_2,
            prob_date1_greater,
            prob_date2_greater,
            passage_datetoken_prob_1,
            passage_datetoken_prob_2,
            average_passage_distribution,
            aux_loss,
        ) = self.date_comparison(passage_attention_1, passage_attention_2, weighted_question_vector,
                                 "greater", date1_entidxs, date2_entidxs)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        date1_output = Output(output_type="dates", values=date_distribution_1, label="date_1")
        date2_output = Output(output_type="dates", values=date_distribution_2, label="date_2")
        pattn_output = Output(output_type="passage", values=average_passage_distribution, label="avg_pattn")
        debug_info_dict = {"compare_date_gt": [qattn_output, date1_output, date2_output, pattn_output]}
        self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(average_passage_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector", "num1_entidxs", "num2_entidxs"])
    def compare_num_lt(
            self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
            num1_entidxs=None, num2_entidxs=None
    ) -> PassageAttention:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            num_distribution_1,
            num_distribution_2,
            prob_num1_lesser,
            prob_num2_lesser,
            passage_numtoken_prob_1, passage_numtoken_prob_2,
            average_passage_distribution,
            aux_loss,
        ) = self.num_comparison(passage_attention_1, passage_attention_2, weighted_question_vector,
                                "lesser", num1_entidxs, num2_entidxs)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""

        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        num1_output = Output(output_type="numbers", values=num_distribution_1, label="num_1")
        num2_output = Output(output_type="numbers", values=num_distribution_2, label="num_2")
        pattn_output = Output(output_type="passage", values=average_passage_distribution, label="avg_pattn")
        debug_info_dict = {"compare_num_lt": [qattn_output, num1_output, num2_output, pattn_output]}
        self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(average_passage_distribution, loss=loss, debug_value=debug_value)

    # @predicate
    @predicate_with_side_args(["question_attention", "weighted_question_vector", "num1_entidxs", "num2_entidxs"])
    def compare_num_gt(
            self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
            num1_entidxs=None, num2_entidxs=None
    ) -> PassageAttention:
        """ In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2
        """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (
            num_distribution_1,
            num_distribution_2,
            prob_num1_greater,
            prob_num2_greater,
            passage_numtoken_prob_1, passage_numtoken_prob_2,
            average_passage_distribution,
            aux_loss,
        ) = self.num_comparison(passage_attention_1, passage_attention_2, weighted_question_vector,
                                "greater", num1_entidxs, num2_entidxs)
        average_passage_distribution = clamp_distribution(average_passage_distribution)

        loss = 0.0
        loss += aux_loss
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss

        debug_value = ""
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        num1_output = Output(output_type="numbers", values=num_distribution_1, label="num_1")
        num2_output = Output(output_type="numbers", values=num_distribution_2, label="num_2")
        pattn_output = Output(output_type="passage", values=average_passage_distribution, label="avg_pattn")
        debug_info_dict = {"compare_num_gt": [qattn_output, num1_output, num2_output, pattn_output]}
        self.modules_debug_info[-1].append(debug_info_dict)

        return PassageAttention(average_passage_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def year_difference_two_events(
            self, passage_attn_1: PassageAttention, passage_attn_2: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor
    ) -> YearDifference:
        """ Given two passage spans, ground them to dates, and then return the difference between their years """

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        date_distribution_1, passage_datetoken_prob_1, date_winloss1 = self.compute_date_scores(
            passage_attention_1, weighted_question_vector)
        date_distribution_2, passage_datetoken_prob_2, date_winloss2 = self.compute_date_scores(
            passage_attention_2, weighted_question_vector)

        # Shape: (number_of_year_differences, )
        year_difference_dist = self.expected_date_year_difference(date_distribution_1, date_distribution_2)

        loss = 0.0
        loss += passage_attn_1.loss
        loss += passage_attn_2.loss
        loss += date_winloss1 + date_winloss2

        debug_value = ""
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        date1_output = Output(output_type="dates", values=date_distribution_1, label="date_1")
        date2_output = Output(output_type="dates", values=date_distribution_2, label="date_2")
        year_diff_output = Output(output_type="year_diffs", values=year_difference_dist, label="year_diff")
        debug_info_dict = {"year_difference_two_events": [qattn_output, date1_output, date2_output, year_diff_output]}
        self.modules_debug_info[-1].append(debug_info_dict)

        return YearDifference(year_difference_dist=year_difference_dist, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def year_difference_single_event(
            self, passage_attn: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor
    ) -> YearDifference:
        """ Given a single passage span, find its start and end dates, then return the difference in years """

        passage_attention = passage_attn._value * self.passage_mask

        # DATE_1 is end since the difference is computed as DATE_1 - DATE_2
        date_distribution_1, passage_datetoken_prob_1, date_winloss1 = self.compute_date_scores(
            passage_attention, weighted_question_vector, date_type="end")
        date_distribution_2, passage_datetoken_prob_2, date_winloss2 = self.compute_date_scores(
            passage_attention, weighted_question_vector, date_type="start")

        # Shape: (number_of_year_differences, )
        year_difference_dist = self.expected_date_year_difference(date_distribution_1, date_distribution_2)

        loss = 0.0
        loss += passage_attn.loss
        loss += date_winloss1 + date_winloss2

        debug_value = ""
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        date1_output = Output(output_type="dates", values=date_distribution_1, label="date_1")
        date2_output = Output(output_type="dates", values=date_distribution_2, label="date_2")
        year_diff_output = Output(output_type="year_diffs", values=year_difference_dist, label="year_diff")
        debug_info_dict = {"year_difference_single_event": [qattn_output, date1_output, date2_output, year_diff_output]}

        self.modules_debug_info[-1].append(debug_info_dict)

        return YearDifference(year_difference_dist=year_difference_dist, loss=loss, debug_value=debug_value)

    # @predicate
    # def extract_passagespan_answer(self) -> PassageSpanAnswer:
    #     with Profile("pass-span-ans"):
    #         # Shape: (passage_length, encoded_dim)
    #         passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage
    #
    #         # Shape: (passage_length, 2)
    #         passage_ans_startend_logits = self.parameters.oneshot_psa_startend_predictor(passage_repr)
    #         # Shape: (passage_length,)
    #         span_start_logits = passage_ans_startend_logits[:, 0]
    #         span_end_logits = passage_ans_startend_logits[:, 1]
    #
    #         passage_mask_bool: torch.BoolTensor = self.passage_mask.bool()
    #         span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask_bool, -1e32)
    #         span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask_bool, -1e32)
    #
    #         span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, passage_mask_bool)
    #         span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, passage_mask_bool)
    #
    #         span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs, passage_mask_bool, -1e32)
    #         span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs, passage_mask_bool, -1e32)
    #
    #         loss = 0.0
    #
    #         debug_value = ""
    #         if self._debug:
    #             debug_info_dict = {"extract_passagespan_answer": []}
    #             self.modules_debug_info[-1].append(debug_info_dict)
    #             debug_value += f"OneShotPassageSpanAnswer extraction: nothing to visualize"
    #
    #     return PassageSpanAnswer(
    #         passage_span_start_log_probs=span_start_log_probs,
    #         passage_span_end_log_probs=span_end_log_probs,
    #         start_logits=span_start_logits,
    #         end_logits=span_end_logits,
    #         loss=loss,
    #         debug_value=debug_value,
    #     )

    @predicate
    def select_passagespan_answer(self, passage_attention: PassageAttention) -> PassageSpanAnswer:
        with Profile("find-span-ans"):
            passage_attn = passage_attention._value
            passage_mask_bool: torch.BoolTensor = self.passage_mask.bool()

            # Shape: (passage_length)
            # passage_attn = passage_attn * self.passage_mask
            passage_attn = passage_attn * self.passage_mask

            scaled_attentions = [passage_attn * sf for sf in self.parameters.passage_attention_scalingvals]
            # Shape: (passage_length, num_scaling_factors)
            scaled_passage_attentions = torch.stack(scaled_attentions, dim=1)

            # Shape: (passage_lengths, hidden_dim)
            passage_span_hidden_reprs = self.parameters.passage_attention_to_span(
                scaled_passage_attentions.unsqueeze(0), self.passage_mask.unsqueeze(0)
            ).squeeze(0)

            span_start_log_probs, span_end_log_probs, span_start_logits, span_end_logits = None, None, None, None
            passage_bio_logprobs = None
            if self.parameters.passage_startend_predictor is not None:
                # Shape: (passage_lengths, 2)
                passage_span_logits = self.parameters.passage_startend_predictor(passage_span_hidden_reprs)

                # Shape: (passage_length)
                span_start_logits = passage_span_logits[:, 0]
                span_end_logits = passage_span_logits[:, 1]

                span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask_bool, -1e32)
                span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask_bool, -1e32)

                span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, passage_mask_bool)
                span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, passage_mask_bool)

                span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs, passage_mask_bool, -1e32)
                span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs, passage_mask_bool, -1e32)

            elif self.parameters.passage_bio_predictor is not None:
                # Shape: (passage_length, num_bio_tags)
                passage_span_logits = self.parameters.passage_bio_predictor(passage_span_hidden_reprs)
                passage_bio_logprobs = allenutil.masked_log_softmax(passage_span_logits, dim=-1, mask=None)

            loss = passage_attention.loss

            #if self._debug:
            debug_info_dict = {"select_passagespan_answer": []}
            self.modules_debug_info[-1].append(debug_info_dict)

        return PassageSpanAnswer(
            passage_span_start_log_probs=span_start_log_probs,
            passage_span_end_log_probs=span_end_log_probs,
            start_logits=span_start_logits,
            end_logits=span_end_logits,
            passage_attn=passage_attn,
            bio_logprobs=passage_bio_logprobs,
            loss=loss,
            debug_value="",
        )

    @predicate
    def aggregate_count(self, passage_attention: PassageAttention) -> CountNumber:
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
            variance = 0.2

            loss = 0
            loss += passage_attention.loss
            if count_mean > 10.0:
                extra_loss = F.mse_loss(count_mean,
                                        allenutil.move_to_device(torch.tensor(9.0), cuda_device=self.device_id))
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

                count_output = Output(output_type="count", values=count_distribution, label="count_dist")
                debug_info_dict = {"aggregate_count": [count_output]}
                # debug_info_dict = {"count": {"count": count_distribution,
                #                              "passage_input": passage_attn}}
                self.modules_debug_info[-1].append(debug_info_dict)

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

        if self._debug:
            num_diff_output = Output(output_type="composed_numbers", values=number_difference_dist, label="num_diff")
            debug_info_dict = {"passagenumber_difference": [num_diff_output]}
            # debug_info_dict = {"number-difference": {"difference_value": number_difference_dist}}
            self.modules_debug_info[-1].append(debug_info_dict)

        return ComposedNumber(composed_number_dist=number_difference_dist, loss=loss, debug_value=debug_value)

    @predicate
    def passagenumber_addition(
            self, passage_number_1: PassageNumber, passage_number_2: PassageNumber
    ) -> ComposedNumber:
        """ Find the expected sum of two number distributions. """
        number_addition_dist, loss, debug_value = self.number_add_sub_module(
            number_1=passage_number_1, number_2=passage_number_2, add_sub="add"
        )

        if self._debug:
            num_add_output = Output(output_type="composed_numbers", values=number_addition_dist, label="num_add")
            debug_info_dict = {"passagenumber_addition": [num_add_output]}
            # debug_info_dict = {"number-addition": {"addition_value": number_addition_dist}}
            self.modules_debug_info[-1].append(debug_info_dict)

        return ComposedNumber(composed_number_dist=number_addition_dist, loss=loss, debug_value=debug_value)

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

    @predicate_with_side_args(["question_attention", "weighted_question_vector", "num_entidxs"])
    def select_num(
            self, passage_attention: PassageAttention,
            question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
            num_entidxs=None
    ) -> PassageNumber:
        # This comes as a list of groundings; even though for this action there's just one
        # This can be completely-zero indicating masked. In that case, don't compute loss
        passage_attn = passage_attention._value
        # Shape: (passage_length)
        passage_attn = passage_attn * self.passage_mask

        # if self.metadata["question_id"] == "715cc0d0-fdad-4db3-b83e-7067b0b65d69":
        #     import pdb
        #     pdb.set_trace()
        #     number_distribution, passage_numtoken_probs, _, num_win_loss = self.compute_num_distribution(
        #         passage_attention=passage_attn, weighted_question_vector=weighted_question_vector, pdb=True)

        number_distribution, passage_numtoken_probs, _, num_win_loss = self.compute_num_distribution(
            passage_attention=passage_attn, weighted_question_vector=weighted_question_vector)

        grounding_loss = aux_symbol_loss(num_entidxs, number_distribution)

        loss = 0.0
        loss += num_win_loss
        loss += grounding_loss
        loss += passage_attention.loss

        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        num_output = Output(output_type="numbers", values=number_distribution, label="number_dist")
        debug_info_dict = {"select_num": [qattn_output, num_output]}
        self.modules_debug_info[-1].append(debug_info_dict)
        return PassageNumber(passage_number_dist=number_distribution, loss=loss, debug_value="")

    def compute_implicitnum_distribution(self, weighted_question_vector: Tensor):
        """ Given a question attention, compute a distribution over implicit numbers for this language.
            See compute_date_distribution for details
        """
        with Profile("implicit-num"):
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

    @predicate_with_side_args(["question_attention", "weighted_question_vector"])
    def select_implicit_num(self, question_attention: Tensor, weighted_question_vector: Tensor) -> PassageNumber:
        number_distribution = self.compute_implicitnum_distribution(weighted_question_vector=weighted_question_vector)
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
            num_output = Output(output_type="numbers", values=number_distribution, label="number_dist")
            debug_info_dict = {"select_implicit_num": [num_output]}
            # debug_info_dict = {"find-implicit-num": {"number": number_distribution,
            #                                          "question": question_attention}}
            self.modules_debug_info[-1].append(debug_info_dict)
        return PassageNumber(passage_number_dist=number_distribution, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector", "num_entidxs"])
    def select_min_num(
        self, passage_attention: PassageAttention,
        question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
        num_entidxs=None
    ) -> PassageAttention:
        (minmax_num_pattn, inputpattn_num_dist, inputpattn_numtoken_probs, minmax_numtoken_probs,
         loss, debug_value) = self.minmaxNumPattn_module(
            passage_attention=passage_attention, weighted_question_vector=weighted_question_vector,
            min_max_op="min", num_entidxs_supervision=num_entidxs)
        # if self._debug:
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        num_input = Output(output_type="passage", values=inputpattn_numtoken_probs, label="number_input")
        minmax_out = Output(output_type="passage", values=minmax_numtoken_probs, label="min_number")
        pattn_output = Output(output_type="passage", values=minmax_num_pattn, label="passage_attn")
        debug_info_dict = {"select_min_num": [qattn_output, num_input, minmax_out, pattn_output]}
        self.modules_debug_info[-1].append(debug_info_dict)
        return PassageAttention(passage_attention=minmax_num_pattn, loss=loss, debug_value=debug_value)

    @predicate_with_side_args(["question_attention", "weighted_question_vector", "num_entidxs"])
    def select_max_num(
        self, passage_attention: PassageAttention,
        question_attention: torch.Tensor, weighted_question_vector: torch.Tensor,
        num_entidxs=None
    ) -> PassageAttention:
        (minmax_num_pattn, inputpattn_num_dist, inputpattn_numtoken_probs, minmax_numtoken_probs,
         loss, debug_value) = self.minmaxNumPattn_module(
            passage_attention=passage_attention, weighted_question_vector=weighted_question_vector,
            min_max_op="max", num_entidxs_supervision=num_entidxs)
        # if self._debug:
        qattn_output = Output(output_type="question", values=question_attention, label="ques_attn")
        num_input = Output(output_type="passage", values=inputpattn_numtoken_probs, label="number_input")
        minmax_out = Output(output_type="passage", values=minmax_numtoken_probs, label="max_number")
        pattn_output = Output(output_type="passage", values=minmax_num_pattn, label="passage_attn")
        debug_info_dict = {"select_max_num": [qattn_output, num_input, minmax_out, pattn_output]}
        self.modules_debug_info[-1].append(debug_info_dict)
        return PassageAttention(passage_attention=minmax_num_pattn, loss=loss, debug_value=debug_value)

    def minmaxNumPattn_module(self, passage_attention: PassageAttention, weighted_question_vector: torch.Tensor,
                              min_max_op: str, num_entidxs_supervision=None):
        assert min_max_op in ["min", "max"]

        pattn = passage_attention._value * self.passage_mask
        # Computing for debugging and aux loss purposes
        inputpattn_num_dist, inputpnum_token_prob, p2p_token2num_alignment, num_winloss = self.compute_num_distribution(
            pattn, weighted_question_vector)
        (minmax_num_pattn, input_numtoken_probs_only, minmax_numtoken_probs_only,
         minmaxnum_token_prob) = self.pattn_for_minmaxNum(pattn=pattn,
                                                          p2p_token2num_alignment=p2p_token2num_alignment,
                                                          max_min=min_max_op)

        loss = aux_symbol_loss(num_entidxs_supervision, inputpattn_num_dist)
        loss += num_winloss

        loss += passage_attention.loss

        debug_value = ""
        inputpattn_numtoken_probs = myutils.round_all(myutils.tocpuNPList(inputpnum_token_prob), 3)
        minmax_numtoken_probs = myutils.round_all(myutils.tocpuNPList(minmaxnum_token_prob), 3)

        return (minmax_num_pattn, inputpattn_num_dist, inputpattn_numtoken_probs, minmax_numtoken_probs,
                loss, debug_value)

    def pattn_for_minmaxNum(self, pattn: torch.FloatTensor, p2p_token2num_alignment: torch.Tensor, max_min: str):
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
            pattn_times_numbertokenprobs = p2p_token2num_alignment * pattn.unsqueeze(1)

            # Shape: (passage_length, num_of_number_tokens) -- For each token, a number-token distribution, where
            # 1. The underlying numbers are sorted in increasing order
            # 2. The probabiltiy is weighed by the token-prob in the input pattn
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

            # These are the tokens ids for numbers s.t. numbers are sorted in increasing order
            number_tokenidxs_sorted = allenutil.move_to_device(
                torch.LongTensor(self.passage_number_sortedtokenidxs), cuda_device=self.device_id)
            # Scattering min-max number distribution back to tokens
            minmax_number_token_distribution = pattn.new_zeros(self.passage_length)
            minmax_number_token_distribution.scatter_add_(0, number_tokenidxs_sorted,
                                                          only_numbertoken_minmaxprobs_sorted)

        return (new_pattn, only_expected_numbertoken_probs, only_numbertoken_minmaxprobs_sorted,
                minmax_number_token_distribution)

    """
    @predicate_with_side_args(["question_attention"])
    def get_question_number(self, question_attention: QuestionAttention) -> QuestionNumber:
        pass

    @predicate
    def filter_num_eq(self, passage_attention: PassageAttention, question_number: QuestionNumber) -> PassageAttention:
        pass

    @predicate
    def filter_num_lt(self, passage_attention: PassageAttention, question_number: QuestionNumber) -> PassageAttention:
        pass

    @predicate
    def filter_num_gt(self, passage_attention: PassageAttention, question_number: QuestionNumber) -> PassageAttention:
        pass

    @predicate
    def filter_num_lt_eq(self, passage_attention: PassageAttention,
                         question_number: QuestionNumber) -> PassageAttention:
        pass

    @predicate
    def filter_num_gt_eq(self, passage_attention: PassageAttention,
                         question_number: QuestionNumber) -> PassageAttention:
        pass
    """


def get_empty_language_object() -> DropLanguage:
    droplanguage = DropLanguage(
        encoded_passage=None,
        modeled_passage=None,
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
        parameters=None,
        start_types=None,
    )
    return droplanguage

if __name__ == "__main__":
    dl = get_empty_language_object()
    print("All possible productions")
    print("\n".join(dl.all_possible_productions()))

    print("\nAllGet nonterminal productions")
    print("\n".join(list(dl.get_nonterminal_productions())))

    print("Functions in DROP Language")
    print(list(dl._functions.keys()))



    # print(spanans.__class__.__name__)
