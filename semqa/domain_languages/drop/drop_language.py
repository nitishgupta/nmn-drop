from typing import Dict, List, Tuple

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
                 end_logits) -> None:
        """ Tuple of start_log_probs and end_log_probs tensor """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._value = (question_span_start_log_probs, question_span_end_log_probs)
        self.debug_value = ""


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
                                # modeled_passage=None,
                                # passage_token2datetoken_sim=None,
                                question_mask=None,
                                passage_mask=None,
                                passage_tokenidx2dateidx=None,
                                passage_date_values=None,
                                question_passage_attention=None,
                                passage_token2date_similarity=None,
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
                 question_passage_attention: Tensor,
                 passage_token2date_similarity: Tensor,
                 parameters: ExecutorParameters,
                 start_types,
                 device_id: int = -1,
                 max_samples=10,
                 question_to_use: str = 'encoded',
                 passage_to_use: str = 'encoded',
                 metadata={},
                 debug=False) -> None:

        if start_types is None:
            start_types = {PassageSpanAnswer, QuestionSpanAnswer}
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

        self.parameters = parameters
        self.max_samples = max_samples

        self.metadata = metadata
        self._debug = debug

        if passage_to_use == 'embedded':
            self.passage = self.embedded_passage
        elif passage_to_use == 'encoded':
            self.passage = self.encoded_passage
        elif passage_to_use == 'modeled':
            raise NotImplementedError
            # self.passage = self.modeled_passage
        else:
            raise NotImplementedError

        if question_to_use == 'embedded':
            self.question = self.embedded_question
        elif question_to_use == 'encoded':
            self.question = self.encoded_question

        self.device_id = device_id

        initialization_returns = self.initialize()
        self.question_passage_attention = question_passage_attention # initialization_returns["question_passage_attention"]
        self.date_lt_mat = initialization_returns["date_lt_mat"]
        self.date_gt_mat = initialization_returns["date_gt_mat"]
        # Shape: (passage_length, passage_length)
        self.passage_passage_token2date_similarity = passage_token2date_similarity # initialization_returns["p2p_t2date_sim_mat"]

        if self._debug:
            num_date_tokens = self.passage_datetokens_mask_float.sum()
            plen = self.passage_mask.sum()
            siml1norm = self.passage_passage_token2date_similarity.norm(p=1)/(num_date_tokens * plen)
            sim_avgval = self.passage_passage_token2date_similarity.sum() / (num_date_tokens * plen)
            print(f"Passage Token2Date sim, Avg L1 Norm: {siml1norm}. Avg Val: {sim_avgval}")

    def initialize(self):
        date_gt_mat, date_lt_mat = self.compute_date_comparison_matrices(self.passage_date_values, self.device_id)

        question_passage_attention = None # self.compute_question_passage_similarity()

        passage_passage_token2date_similarity = None # self.compute_passage_token2date_similarity()

        return {"question_passage_attention": question_passage_attention,
                "date_gt_mat": date_gt_mat,
                "date_lt_mat": date_lt_mat,
                "p2p_t2date_sim_mat": passage_passage_token2date_similarity}


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

        # Shape: (passage_length, ) -- indicating which tokens are dates
        passage_tokenidx2dateidx_mask = (self.passage_tokenidx2dateidx > -1)
        passage_datetokens_mask_fl = passage_tokenidx2dateidx_mask.float()

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

        masked_passage_tokenidx2dateidx = passage_tokenidx2dateidx_mask.long() * self.passage_tokenidx2dateidx
        masked_passage_datetoken_scores = passage_tokenidx2dateidx_mask.float() * passage_datetoken_attention

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
                                                                  mask=passage_tokenidx2dateidx_mask,
                                                                  memory_efficient=True)

        # print(masked_passage_datetoken_scores)
        # print(masked_passage_datetoken_probs)
        # print()

        ''' normalized method with method 2 '''
        date_distribution = passage_attention.new_zeros(self.num_passage_dates)
        date_distribution.scatter_add_(0, masked_passage_tokenidx2dateidx, masked_passage_datetoken_probs)
        date_scores = date_distribution

        # print(passage_datetoken_attention)
        # print(masked_passage_datetoken_scores)
        # print(date_distribution)
        # print()

        return date_distribution, date_scores


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
        ''' In short; outputs PA_1 if D1 > D2 i.e. is PA_1 occurred after PA_2
        '''

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

    @predicate
    def find_passageSpanAnswer(self, passage_attention: PassageAttention_answer) -> PassageSpanAnswer:
        passage_attn = passage_attention._value
        # passage_attn = passage_attention

        # Shape: (passage_length)
        passage_attn = (passage_attn * self.passage_mask)

        scaled_attentions = [passage_attn * sf for sf in self.parameters.passage_attention_scalingvals]
        # Shape: (passage_lengths, num_scaling_factors)
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
    '''


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
    dl = get_empty_language_object()
    print(dl.get_nonterminal_productions())

    # print(spanans.__class__.__name__)
