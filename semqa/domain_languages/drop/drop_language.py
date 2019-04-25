from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch import Tensor
from torch.nn import LSTM
import torch.nn.functional as F

import allennlp.nn.util as allenutil
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args, ExecutionError)

from semqa.domain_languages.drop.execution_parameters import ExecutorParameters
from semqa.domain_languages import domain_language_utils as dlutils
import utils.util as myutils


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



# class DateDistribution:
#     def __init__(self,
#                  year_distribution: Tensor,
#                  month_distribution: Tensor,
#                  day_distribution: Tensor) -> None:
#         self.year_distribution = year_distribution
#         self.month_distribution = month_distribution
#         self.day_distribution = day_distribution
#
#
# class DateDelta:
#     def __init__(self,
#                  year_delta: Tensor,
#                  month_delta: Tensor,
#                  day_delta: Tensor) -> None:
#         self.year_delta = year_delta
#         self.month_delta = month_delta
#         self.day_delta = day_delta


class PassageSpanAnswer():
    def __init__(self,
                 passage_span_start_log_probs: Tensor,
                 passage_span_end_log_probs: Tensor,
                 start_logits,
                 end_logits,
                 loss=0.0,
                 debug_value="") -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (passage_span_start_log_probs, passage_span_end_log_probs)
        self.loss = loss
        self.debug_value = debug_value


class QuestionSpanAnswer():
    def __init__(self,
                 question_span_start_log_probs: Tensor,
                 question_span_end_log_probs: Tensor,
                 start_logits,
                 end_logits,
                 loss=0.0,
                 debug_value="") -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (question_span_start_log_probs, question_span_end_log_probs)
        self.loss = loss
        self.debug_value = debug_value


class QuestionAttention():
    def __init__(self, question_attention, debug_value=""):
        self._value = question_attention
        self.debug_value = debug_value


class PassageAttention():
    def __init__(self, passage_attention, debug_value=""):
        self._value = passage_attention
        self.debug_value = debug_value


class PassageAttention_answer():
    def __init__(self, passage_attention, loss=0.0, debug_value=""):
        self._value = passage_attention
        self.loss = loss
        self.debug_value = debug_value


class YearDifference():
    def __init__(self, year_difference_dist, loss=0.0, debug_value=""):
        self._value = year_difference_dist
        self.loss = loss
        self.debug_value = debug_value


class PassageNumber():
    def __init__(self, passage_number_dist, loss=0.0, debug_value=""):
        self._value = passage_number_dist
        self.loss = loss
        self.debug_value = debug_value


# A ``NumberAnswer`` is a distribution over the possible answers from addition and subtraction.
class NumberAnswer(Tensor):
    pass


# A ``CountAnswer`` is a distribution over the possible counts in the passage.
class CountAnswer(Tensor):
    pass


def get_empty_language_object():
    droplanguage = DropLanguage(rawemb_question=None,
                                embedded_question=None,
                                encoded_question=None,
                                rawemb_passage=None,
                                embedded_passage=None,
                                encoded_passage=None,
                                question_mask=None,
                                passage_mask=None,
                                passage_tokenidx2dateidx=None,
                                passage_date_values=None,
                                passage_tokenidx2numidx=None,
                                passage_num_values=None,
                                year_differences=None,
                                year_differences_mat=None,
                                question_passage_attention=None,
                                passage_question_attention=None,
                                passage_token2date_similarity=None,
                                passage_token2num_similarity=None,
                                parameters=None,
                                start_types=None)
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

    #TODO(nitish): Defaulting all parameters to None since in the reader we create an
    def __init__(self,
                 rawemb_question: Tensor,
                 embedded_question: Tensor,
                 encoded_question: Tensor,
                 rawemb_passage: Tensor,
                 embedded_passage: Tensor,
                 encoded_passage: Tensor,
                 # modeled_passage: Tensor,
                 # passage_token2datetoken_sim: Tensor,
                 question_mask: Tensor,
                 passage_mask: Tensor,
                 passage_tokenidx2dateidx: torch.LongTensor,
                 passage_date_values: List[Date],
                 passage_tokenidx2numidx: torch.LongTensor,
                 passage_num_values: List[float],
                 year_differences: List[int],
                 year_differences_mat: np.array,
                 question_passage_attention: Tensor,
                 passage_question_attention: Tensor,
                 passage_token2date_similarity: Tensor,
                 passage_token2num_similarity: Tensor,
                 parameters: ExecutorParameters,
                 start_types=None,
                 device_id: int = -1,
                 max_samples=5,
                 metadata={},
                 debug=False) -> None:

        if start_types is None:
            start_types = {PassageSpanAnswer, YearDifference, PassageNumber}   # , QuestionSpanAnswer}

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
        # self.modeled_passage = modeled_passage
        self.passage_mask = passage_mask

        # Shape: (passage_length, )
        self.passage_tokenidx2dateidx = passage_tokenidx2dateidx.long()
        passage_tokenidx2dateidx_mask = (self.passage_tokenidx2dateidx > -1)
        self.passage_datetokens_mask_long = passage_tokenidx2dateidx_mask.long()
        self.passage_datetokens_mask_float = passage_tokenidx2dateidx_mask.float()
        # List[Date] - number of unique dates in the passage
        self.passage_date_values: List[Date] = passage_date_values
        self.num_passage_dates = len(self.passage_date_values)

        # Shape: (passage_length, )
        self.passage_tokenidx2numidx = passage_tokenidx2numidx.long()
        passage_tokenidx2numidx_mask = (self.passage_tokenidx2numidx > -1)
        self.passage_numtokens_mask_long = passage_tokenidx2numidx_mask.long()
        self.passage_numtokens_mask_float = passage_tokenidx2numidx_mask.float()
        # List[float] - number of unique numbers in the passage
        self.passage_num_values: List[float] = passage_num_values
        self.num_passage_nums = len(self.passage_num_values)

        self.parameters = parameters
        self.max_samples = max_samples

        self.metadata = metadata
        self._debug = debug

        self.device_id = device_id

        # Shape: (question_length, passage_length)
        self.question_passage_attention = question_passage_attention  # initialization_returns["question_passage_attention"]
        # Shape: (passage_length, question_length)
        self.passage_question_attention = passage_question_attention
        # Shape: (passage_length, passage_length)
        self.passage_passage_token2date_similarity = passage_token2date_similarity
        self.passage_passage_token2num_similarity = passage_token2num_similarity
        initialization_returns = self.initialize()
        self.date_lt_mat = initialization_returns["date_lt_mat"]
        self.date_gt_mat = initialization_returns["date_gt_mat"]
        self.num_lt_mat = initialization_returns["num_lt_mat"]
        self.num_gt_mat = initialization_returns["num_gt_mat"]

        # Shape: (num_passage_dates, num_passage_dates, num_of_year_differences)
        self.year_differences_mat = allenutil.move_to_device(torch.FloatTensor(year_differences_mat),
                                                             cuda_device=self.device_id)

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

    def initialize(self):
        date_gt_mat, date_lt_mat = self.compute_date_comparison_matrices(self.passage_date_values, self.device_id)
        num_gt_mat, num_lt_mat = self.compute_item_comparison_matrices(self.passage_num_values, self.device_id)

        # question_passage_attention = None # self.compute_question_passage_similarity()
        # passage_passage_token2date_similarity = None # self.compute_passage_token2date_similarity()

        return {"date_gt_mat": date_gt_mat,
                "date_lt_mat": date_lt_mat,
                "num_gt_mat": num_gt_mat,
                "num_lt_mat": num_lt_mat}
                #"question_passage_attention": question_passage_attention,
                #"p2p_t2date_sim_mat": passage_passage_token2date_similarity}


    def compute_question_passage_similarity(self):
        # Shape: (1, question_length, passage_length)
        question_passage_similarity = self.parameters.dotprod_matrix_attn(self.rawemb_question.unsqueeze(0),
                                                                          self.rawemb_passage.unsqueeze(0))
        question_passage_similarity = self.parameters._dropout(question_passage_similarity)

        # Shape: (question_length, passage_length)
        question_passage_attention = allenutil.masked_softmax(question_passage_similarity,
                                                              self.passage_mask.unsqueeze(0),
                                                              memory_efficient=True).squeeze(0)

        return question_passage_attention

    def compute_passage_token2date_similarity(self):
        # Shape: (passage_length, passage_length) - for each token x in the row, weight given by it to each token y in
        # the column for y to be a date associated to x
        passage_passage_token2date_similarity = self.parameters._dropout(self.parameters.passage_to_date_attention(
            self.passage.unsqueeze(0),
            self.passage.unsqueeze(0))).squeeze(0)

        passage_passage_token2date_similarity = passage_passage_token2date_similarity * self.passage_mask.unsqueeze(0)
        passage_passage_token2date_similarity = passage_passage_token2date_similarity * self.passage_mask.unsqueeze(1)

        passage_passage_token2date_similarity = (passage_passage_token2date_similarity *
                                                 self.passage_datetokens_mask_float.unsqueeze(0))

        return passage_passage_token2date_similarity


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

    @predicate_with_side_args(['question_attention'])
    def find_QuestionAttention(self, question_attention: Tensor) -> QuestionAttention:

        debug_value = ""
        if self._debug:
            qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(question_attention, self.metadata["question_tokens"])
            debug_value = qattn_vis_most

        return QuestionAttention(question_attention, debug_value=debug_value)


    @predicate_with_side_args(['question_attention'])
    def find_PassageAttention(self, question_attention: Tensor) -> PassageAttention:
        question_attention = question_attention * self.question_mask

        # Shape: (question_length, passage_length)
        question_passage_attention = self.question_passage_attention * question_attention.unsqueeze(1)

        passage_attention = question_passage_attention.sum(0)

        debug_value = ""
        if self._debug:
            qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(question_attention, self.metadata["question_tokens"])
            debug_value += f"Qattn: {qattn_vis_complete} \t "
            pattn_vis_complete, pattn_vis_most = dlutils.listTokensVis(passage_attention, self.metadata["passage_tokens"])
            debug_value += f"Pattn: {pattn_vis_most}"

        return PassageAttention(passage_attention, debug_value=debug_value)


    def compute_date_scores(self, passage_attention: Tensor):
        ''' Given a passage over passage token2date attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) dates in the passage.

            Using the token_attention from the passage_attention, find the expected input_token2date_token
            distribution - weigh each row in passage_passage_token2date_attention and sum
            Use this date_token attention as scores for the dates they refer to. Since each date can be referred
            to in multiple places, we use scatter_add_ to get the total_score.
            Softmax over these scores is the date dist.
        '''

        # (passage_length, passage_length)
        passage_passage_tokendate_attention = self.passage_passage_token2date_similarity * passage_attention.unsqueeze(1)

        # maxidx = torch.argmax(passage_attention)
        # print(passage_attention[maxidx])
        # print(self.passage_passage_token2date_similarity[maxidx].sum())
        # print(self.passage_passage_token2date_similarity[maxidx+1].sum())
        # print(self.passage_passage_token2date_similarity[maxidx + 2].sum())
        # print(passage_passage_tokendate_attention[maxidx])
        # print()

        # Shape: (passage_length, ) -- weighted average of distributions in above step
        # Attention value for each passage token to be a date associated to the query
        passage_datetoken_attention = passage_passage_tokendate_attention.sum(0)

        masked_passage_tokenidx2dateidx = self.passage_tokenidx2dateidx * self.passage_datetokens_mask_long

        ''' We first softmax scores over tokens then aggregate probabilities for same dates
            Another way could be to first aggregate scores, then compute softmx -- in that method if the scores are
            large, longer dates will get an unfair advantage; hence we keep it this way.
            For eg. [800, 800, 800] where first two point to same date, current method will get a dist of [0.66, 0.33],
            where as first-score-agg. will get [1.0, 0.0].
            Potential downside, if the scores are [500, 500, 1200] our method will get [small, ~1.0],
            where as first-score-agg. might get something more uniform. 
            I think the bottomline is to control the scaling of scores.
        '''
        masked_passage_datetoken_probs = allenutil.masked_softmax(passage_datetoken_attention,
                                                                  mask=self.passage_datetokens_mask_long,
                                                                  memory_efficient=True)
        masked_passage_datetoken_probs = masked_passage_datetoken_probs * self.passage_datetokens_mask_float

        ''' normalized method with method 2 '''
        date_distribution = passage_attention.new_zeros(self.num_passage_dates)
        date_distribution.scatter_add_(0, masked_passage_tokenidx2dateidx, masked_passage_datetoken_probs)

        date_distribution = torch.clamp(date_distribution, min=1e-20, max=1 - 1e-20)

        return date_distribution, date_distribution

    def compute_num_distribution(self, passage_attention: Tensor):
        ''' Given a passage over passage token2num attention (normalized), and an additional passage attention
            for token importance, compute a distribution over (unique) nums in the passage.
            See compute_date_distribution for details
        '''

        # (passage_length, passage_length)
        passage_passage_tokennum_attention = self.passage_passage_token2num_similarity * passage_attention.unsqueeze(1)
        # Shape: (passage_length, ) -- weighted average of distributions in above step
        # Attention value for each passage token to be a date associated to the query
        passage_numtoken_attention = passage_passage_tokennum_attention.sum(0)

        # maxidx = torch.argmax(passage_attention)
        # print(passage_attention[maxidx])
        # print(self.passage_passage_token2date_similarity[maxidx].sum())
        # print(self.passage_passage_token2date_similarity[maxidx+1].sum())
        # print(self.passage_passage_token2date_similarity[maxidx + 2].sum())
        # print(passage_passage_tokendate_attention[maxidx])
        # print()

        masked_passage_tokenidx2numidx = self.passage_numtokens_mask_long * self.passage_tokenidx2numidx
        masked_passage_numtoken_scores = self.passage_numtokens_mask_float * passage_numtoken_attention

        masked_passage_numtoken_probs = allenutil.masked_softmax(masked_passage_numtoken_scores,
                                                                 mask=self.passage_numtokens_mask_float,
                                                                 memory_efficient=True)
        masked_passage_numtoken_probs = masked_passage_numtoken_probs * self.passage_numtokens_mask_float
        # print(masked_passage_datetoken_scores)
        # print(masked_passage_datetoken_probs)
        # print()
        ''' normalized method with method 2 '''
        num_distribution = passage_attention.new_zeros(self.num_passage_nums)
        num_distribution.scatter_add_(0, masked_passage_tokenidx2numidx, masked_passage_numtoken_probs)
        num_scores = num_distribution

        num_distribution = torch.clamp(num_distribution, min=1e-20, max=1 - 1e-20)

        return num_distribution, num_distribution


    def expected_date_year_difference(self,
                                      date_distribution_1: torch.FloatTensor,
                                      date_distribution_2: torch.FloatTensor):
        """ Compute a distribution over possible year-differences by marginalizing over the year_differnces_mat.

            Parameters:
            -----------
            date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
            date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """

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

        year_differences_dist = torch.clamp(year_differences_dist, min=1e-20, max=1 - 1e-20)

        return year_differences_dist


    def expected_date_comparison(self, date_distribution_1, date_distribution_2, comparison):
        """ Compute the boolean probability that date_1 > date_2 given distributions over passage_dates for each

        Parameters:
        -----------
        date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """
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

        return expected_bool

    def expected_num_comparison(self, distribution_1, distribution_2, comparison):
        """ Compute the boolean probability that date_1 > date_2 given distributions over passage_dates for each

        Parameters:
        -----------
        date_distribution_1: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        date_distribution_2: ``torch.FloatTensor`` Shape: (self.num_passage_dates, )
        """
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
        return expected_bool


    def date_comparison(self, passage_attention_1, passage_attention_2, comparison: str,
                        gold_date_groundings=None):

        date_distribution_1, _ = self.compute_date_scores(passage_attention_1)
        date_distribution_2, _ = self.compute_date_scores(passage_attention_2)

        bool1 = self.expected_date_comparison(date_distribution_1, date_distribution_2, comparison)
        bool2 = self.expected_date_comparison(date_distribution_2, date_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        if gold_date_groundings is not None:
            # These are one-hot vectors the size of passage_dates
            # These can be zeros as well, indicating the date grounding is unknown, in which case they shouldn't be
            # used for computing the auxiliary date-grounding loss
            gold_date_grounding_1, gold_date_grounding_2 = gold_date_groundings

            # Shape: (2, num_dates)
            gold_date_grounding_tensor = torch.cat([gold_date_grounding_1.unsqueeze(0),
                                                    gold_date_grounding_2.unsqueeze(0)], dim=0)

            # The groundings that contain a zero vector will yield max as zero, and hence mask = 0
            # For valid groundings, the max will be 1 (mask = 1) and will contain the correct idx from argmax
            date_grounding_mask, gold_date_idxs = torch.max(gold_date_grounding_tensor, dim=1)

            # Shape: [2, num_of_dates]
            stacked_date_distributions = torch.cat([date_distribution_1.unsqueeze(0),
                                                    date_distribution_2.unsqueeze(0)], dim=0)

            # Shape: (2) - Converting date_distribution to log-probabilties
            date_grounding_loss = F.nll_loss(input=(stacked_date_distributions + 1e-40).log(),
                                             target=gold_date_idxs,
                                             reduction='none')
            date_grounding_loss = date_grounding_loss * date_grounding_mask
            date_grounding_loss = date_grounding_loss.mean()

        else:
            date_grounding_loss = 0.0

        kl_div_neg_1_2 = -1 * F.kl_div(date_distribution_1, date_distribution_2, reduction='mean')
        kl_div_neg_2_1 = -1 * F.kl_div(date_distribution_2, date_distribution_1, reduction='mean')

        dist1_entropy = -1 * torch.sum(date_distribution_1 * torch.log(date_distribution_1 + 1e-40))
        dist2_entropy = -1 * torch.sum(date_distribution_2 * torch.log(date_distribution_2 + 1e-40))

        aux_loss = date_grounding_loss

        return (date_distribution_1, date_distribution_2, bool1, bool2, average_passage_distribution, aux_loss)


    def num_comparison(self, passage_attention_1, passage_attention_2, comparison: str,
                        gold_num_groundings=None):

        num_distribution_1, _ = self.compute_num_distribution(passage_attention_1)
        num_distribution_2, _ = self.compute_num_distribution(passage_attention_2)

        bool1 = self.expected_num_comparison(num_distribution_1, num_distribution_2, comparison)
        bool2 = self.expected_num_comparison(num_distribution_2, num_distribution_1, comparison)

        average_passage_distribution = bool1 * passage_attention_1 + bool2 * passage_attention_2

        if gold_num_groundings is not None:
            # These are one-hot vectors the size of passage_dates
            # These can be zeros as well, indicating the date grounding is unknown, in which case they shouldn't be
            # used for computing the auxiliary date-grounding loss
            gold_num_grounding_1, gold_num_grounding_2 = gold_num_groundings

            # Shape: (2, num_passage_nums)
            gold_num_grounding_tensor = torch.cat([gold_num_grounding_1.unsqueeze(0),
                                                   gold_num_grounding_2.unsqueeze(0)], dim=0)

            # The groundings that contain a zero vector will yield max as zero, and hence mask = 0
            # For valid groundings, the max will be 1 (mask = 1) and will contain the correct idx from argmax
            grounding_mask, gold_idxs = torch.max(gold_num_grounding_tensor, dim=1)

            # Shape: [2, num_of_dates]
            stacked_num_distributions = torch.cat([num_distribution_1.unsqueeze(0),
                                                   num_distribution_2.unsqueeze(0)], dim=0)
            # Shape: (2) - Converting date_distribution to log-probabilties
            num_grounding_loss = F.nll_loss(input=(stacked_num_distributions + 1e-40).log(),
                                            target=gold_idxs,
                                            reduction='none')
            num_grounding_loss = num_grounding_loss * grounding_mask
            num_grounding_loss = num_grounding_loss.mean()

        else:
            num_grounding_loss = 0.0

        aux_loss = num_grounding_loss

        return (num_distribution_1, num_distribution_2, bool1, bool2, average_passage_distribution, aux_loss)


    # @predicate
    @predicate_with_side_args(['event_date_groundings'])
    def compare_date_lesser_than(self,
                                 passage_attn_1: PassageAttention,
                                 passage_attn_2: PassageAttention,
                                 event_date_groundings = None) -> PassageAttention_answer:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (date_distribution_1,
         date_distribution_2,
         prob_date1_lesser, prob_date2_lesser,
         average_passage_distribution, aux_loss) = self.date_comparison(passage_attention_1, passage_attention_2,
                                                                        "lesser", event_date_groundings)

        debug_value = ""
        if self._debug:
            if event_date_groundings:
                gold_date_1 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[0]), 3)
                gold_date_2 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[1]), 3)
            else:
                gold_date_1, gold_date_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1,
                                                        self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2,
                                                        self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            d1_lt_d2 = myutils.round_all(myutils.tocpuNPList(prob_date1_lesser), 3)
            d2_lt_d1 = myutils.round_all(myutils.tocpuNPList(prob_date2_lesser), 3)

            debug_value += f"Pattn1: {pattn_vis_most_1}\n Date1: {date1}" + \
                           f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}" + \
                           f"\nP(D1 < D2): {d1_lt_d2}  P(D2 < D1): {d2_lt_d1}"
            if gold_date_1:
                debug_value += f"\nGoldDates  Date1: {gold_date_1}  Date2: {gold_date_2}"

        return PassageAttention_answer(average_passage_distribution, loss=aux_loss, debug_value=debug_value)

    # @predicate
    @predicate_with_side_args(['event_date_groundings'])
    def compare_date_greater_than(self,
                                  passage_attn_1: PassageAttention,
                                  passage_attn_2: PassageAttention,
                                  event_date_groundings=None) -> PassageAttention_answer:
        ''' In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2 '''

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (date_distribution_1,
         date_distribution_2,
         prob_date1_greater, prob_date2_greater,
         average_passage_distribution,
         aux_loss) = self.date_comparison(passage_attention_1, passage_attention_2,
                                          "greater", event_date_groundings)

        debug_value = ""
        if self._debug:
            if event_date_groundings:
                gold_date_1 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[0]), 3)
                gold_date_2 = myutils.round_all(myutils.tocpuNPList(event_date_groundings[1]), 3)
            else:
                gold_date_1, gold_date_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1,
                                                        self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2,
                                                        self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            d1_gt_d2 = myutils.round_all(myutils.tocpuNPList(prob_date1_greater), 3)
            d2_gt_d1 = myutils.round_all(myutils.tocpuNPList(prob_date2_greater), 3)

            debug_value += f"Pattn1: {pattn_vis_most_1}\n Date1: {date1}" + \
                           f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}" + \
                           f"\nP(D1 > D2): {d1_gt_d2}  P(D2 > D1): {d2_gt_d1}"

            if gold_date_1:
                debug_value += f"\nGoldDates  Date1: {gold_date_1}  Date2: {gold_date_2}"

        return PassageAttention_answer(average_passage_distribution, loss=aux_loss, debug_value=debug_value)

    @predicate_with_side_args(['event_num_groundings'])
    def compare_num_lesser_than(self,
                                passage_attn_1: PassageAttention,
                                passage_attn_2: PassageAttention,
                                event_num_groundings=None) -> PassageAttention_answer:

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (num_distribution_1,
         num_distribution_2,
         prob_num1_lesser, prob_num2_lesser,
         average_passage_distribution, aux_loss) = self.num_comparison(passage_attention_1, passage_attention_2,
                                                                       "lesser", event_num_groundings)

        debug_value = ""
        if self._debug:
            if event_num_groundings:
                gold_num_1 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[0]), 3)
                gold_num_2 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[1]), 3)
            else:
                gold_num_1, gold_num_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1,
                                                        self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2,
                                                        self.metadata["passage_tokens"])

            num1 = myutils.round_all(myutils.tocpuNPList(num_distribution_1), 3)
            num2 = myutils.round_all(myutils.tocpuNPList(num_distribution_2), 3)
            d1_lt_d2 = myutils.round_all(myutils.tocpuNPList(prob_num1_lesser), 3)
            d2_lt_d1 = myutils.round_all(myutils.tocpuNPList(prob_num2_lesser), 3)

            debug_value += f"Pattn1: {pattn_vis_most_1}\n Num1: {num1}" + \
                           f"\nPattn2: {pattn_vis_most_2}\n Num2: {num2}" + \
                           f"\nP(N1 < N2): {d1_lt_d2}  P(N2 < N1): {d2_lt_d1}"
            if gold_num_1:
                debug_value += f"\nGoldNums Num1: {gold_num_1}  Num2: {gold_num_2}"

        return PassageAttention_answer(average_passage_distribution, loss=aux_loss, debug_value=debug_value)

    # @predicate
    @predicate_with_side_args(['event_num_groundings'])
    def compare_num_greater_than(self,
                                 passage_attn_1: PassageAttention,
                                 passage_attn_2: PassageAttention,
                                 event_num_groundings=None) -> PassageAttention_answer:
        ''' In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2
        '''

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        (num_distribution_1,
         num_distribution_2,
         prob_num1_greater, prob_num2_greater,
         average_passage_distribution, aux_loss) = self.num_comparison(passage_attention_1, passage_attention_2,
                                                                       "greater", event_num_groundings)

        debug_value = ""
        if self._debug:
            if event_num_groundings:
                gold_num_1 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[0]), 3)
                gold_num_2 = myutils.round_all(myutils.tocpuNPList(event_num_groundings[1]), 3)
            else:
                gold_num_1, gold_num_2 = None, None
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1,
                                                        self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2,
                                                        self.metadata["passage_tokens"])

            num1 = myutils.round_all(myutils.tocpuNPList(num_distribution_1), 3)
            num2 = myutils.round_all(myutils.tocpuNPList(num_distribution_2), 3)
            d1_gt_d2 = myutils.round_all(myutils.tocpuNPList(prob_num1_greater), 3)
            d2_gt_d1 = myutils.round_all(myutils.tocpuNPList(prob_num2_greater), 3)

            debug_value += f"Pattn1: {pattn_vis_most_1}\n num1: {num1}" + \
                           f"\nPattn2: {pattn_vis_most_2}\n num2: {num2}" + \
                           f"\nP(N1 > N2): {d1_gt_d2}  P(N2 > N1): {d2_gt_d1}"

            if gold_num_1:
                debug_value += f"\nGoldNums  num1: {gold_num_1}  num2: {gold_num_2}"

        return PassageAttention_answer(average_passage_distribution, loss=aux_loss, debug_value=debug_value)


    @predicate
    def year_difference(self,
                        passage_attn_1: PassageAttention,
                        passage_attn_2: PassageAttention) -> YearDifference:
        ''' Given two passage spans, ground them to dates, and then return the difference between their years '''

        passage_attention_1 = passage_attn_1._value * self.passage_mask
        passage_attention_2 = passage_attn_2._value * self.passage_mask

        date_distribution_1, _ = self.compute_date_scores(passage_attention_1)
        date_distribution_2, _ = self.compute_date_scores(passage_attention_2)

        # Shape: (number_of_year_differences, )
        year_difference_dist = self.expected_date_year_difference(date_distribution_1, date_distribution_2)

        debug_value = ""
        if self._debug:
            _, pattn_vis_most_1 = dlutils.listTokensVis(passage_attention_1,
                                                        self.metadata["passage_tokens"])
            _, pattn_vis_most_2 = dlutils.listTokensVis(passage_attention_2,
                                                        self.metadata["passage_tokens"])

            date1 = myutils.round_all(myutils.tocpuNPList(date_distribution_1), 3)
            date2 = myutils.round_all(myutils.tocpuNPList(date_distribution_2), 3)
            year_diff_dist = myutils.round_all(myutils.tocpuNPList(year_difference_dist), 3)

            debug_value += f"YearDiffDist: {year_diff_dist}\n" + \
                           f"\nPattn1: {pattn_vis_most_1}\n Date1: {date1}" + \
                           f"\nPattn2: {pattn_vis_most_2}\n Date2: {date2}"

        return YearDifference(year_difference_dist=year_difference_dist, loss=0.0, debug_value=debug_value)


    @predicate
    def find_passageSpanAnswer(self, passage_attention: PassageAttention_answer) -> PassageSpanAnswer:
        passage_attn = passage_attention._value
        # passage_attn = passage_attention

        # Shape: (passage_length)
        passage_attn = (passage_attn * self.passage_mask)

        scaled_attentions = [passage_attn * sf for sf in self.parameters.passage_attention_scalingvals]
        # Shape: (passage_length, num_scaling_factors)
        scaled_passage_attentions = torch.stack(scaled_attentions, dim=1)

        # Shape: (passage_lengths, hidden_dim)
        passage_span_hidden_reprs = self.parameters.passage_attention_to_span(scaled_passage_attentions.unsqueeze(0),
                                                                              self.passage_mask.unsqueeze(0)).squeeze(0)

        # Shape: (passage_lengths, 2)
        passage_span_logits = self.parameters.passage_startend_predictor(passage_span_hidden_reprs)

        # Shape: (passage_length)
        span_start_logits = passage_span_logits[:, 0]
        span_end_logits = passage_span_logits[:, 1]

        span_start_logits = allenutil.replace_masked_values(span_start_logits, self.passage_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, self.passage_mask, -1e32)

        span_start_log_probs = allenutil.masked_log_softmax(span_start_logits, self.passage_mask)
        span_end_log_probs = allenutil.masked_log_softmax(span_end_logits, self.passage_mask)

        span_start_log_probs = allenutil.replace_masked_values(span_start_log_probs,
                                                               self.passage_mask, -1e32)
        span_end_log_probs = allenutil.replace_masked_values(span_end_log_probs,
                                                             self.passage_mask, -1e32)

        loss = passage_attention.loss

        debug_value = ""
        if self._debug:
            _, pattn_vis_most = dlutils.listTokensVis(passage_attn, self.metadata["passage_tokens"])
            debug_value += f"Pattn: {pattn_vis_most}"

        return PassageSpanAnswer(passage_span_start_log_probs=span_start_log_probs,
                     passage_span_end_log_probs=span_end_log_probs,
                     start_logits=span_start_logits,
                     end_logits=span_end_logits,
                     loss=loss,
                     debug_value=debug_value)

    '''
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
    '''


    @predicate
    def max_PassageNumber(self, number_distribution: PassageNumber) -> PassageNumber:
        num_dist = number_distribution._value

        cum_dist = num_dist.cumsum(0)
        cum_dist_n = cum_dist ** self.max_samples
        maximum_distribution = cum_dist_n - torch.cat([cum_dist_n.new_zeros(1), cum_dist_n[:-1]])

        loss = number_distribution.loss

        maximum_distribution = torch.clamp(maximum_distribution, min=1e-10, max=1 - 1e-10)

        return PassageNumber(passage_number_dist=maximum_distribution, loss=loss, debug_value="")

    @predicate
    def min_PassageNumber(self, number_distribution: PassageNumber) -> PassageNumber:
        num_dist = number_distribution._value

        cumulative_distribution_function = num_dist.cumsum(0)
        # P(x>=i) = 1 - (P(x<=i) - P(x=i))
        inverse_cumdist = 1 - cumulative_distribution_function + num_dist

        inverse_cumdist_n = inverse_cumdist ** self.max_samples

        inverse_cumdist_n_shifted = torch.cat([inverse_cumdist_n[1:], inverse_cumdist_n.new_zeros(1)])
        minimum_distribution = inverse_cumdist_n - inverse_cumdist_n_shifted

        loss = number_distribution.loss

        minimum_distribution = torch.clamp(minimum_distribution, min=1e-10, max=1 - 1e-10)

        return PassageNumber(passage_number_dist=minimum_distribution, loss=loss, debug_value="")


    @predicate
    def find_PassageNumber(self, passage_attention: PassageAttention) -> PassageNumber:
        passage_attn = passage_attention._value
        # Shape: (passage_length)
        passage_attn = (passage_attn * self.passage_mask)

        number_distribution, _ = self.compute_num_distribution(passage_attention=passage_attn)

        loss = 0.0

        debug_value = ""
        if self._debug:
            number_dist = myutils.round_all(myutils.tocpuNPList(number_distribution), 3)
            _, pattn_vis_most = dlutils.listTokensVis(passage_attn, self.metadata["passage_tokens"])
            debug_value += f"PassageNumber: {number_dist}"
            debug_value += f"\nPattn: {pattn_vis_most}"

        return PassageNumber(passage_number_dist=number_distribution,
                             loss=loss,
                             debug_value=debug_value)


if __name__=='__main__':
    dl = get_empty_language_object()
    print(dl.all_possible_productions())
    print(dl.get_nonterminal_productions())



    # print(spanans.__class__.__name__)
