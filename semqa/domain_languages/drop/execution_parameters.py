from typing import Dict, Optional, List, Any

import copy

import torch
import torch.nn
from allennlp.common import Registrable
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder

from allennlp.modules.attention import Attention, DotProductAttention
from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.modules.matrix_attention import (MatrixAttention, BilinearMatrixAttention,
                                               DotProductMatrixAttention, LinearMatrixAttention)

class ExecutorParameters(torch.nn.Module, Registrable):
    """
        Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 question_encoding_dim: int,
                 passage_encoding_dim: int,
                 passage_attention_to_span: Seq2SeqEncoder,
                 question_attention_to_span: Seq2SeqEncoder,
                 passage_attention_to_count: Seq2SeqEncoder,
                 passage_count_predictor=None,
                 passage_count_hidden2logits=None,
                 dropout: float = 0.0):
        super().__init__()

        self.num_counts = 10

        self.passage_attention_scalingvals = [1, 2, 5, 10]

        self.passage_attention_to_span = passage_attention_to_span
        self.passage_startend_predictor = torch.nn.Linear(self.passage_attention_to_span.get_output_dim(), 2)

        self.question_attention_to_span = question_attention_to_span
        self.question_startend_predictor = torch.nn.Linear(self.question_attention_to_span.get_output_dim(), 2)

        self.passage_attention_to_count = passage_attention_to_count
        # self.passage_count_predictor = torch.nn.Linear(self.passage_attention_to_count.get_output_dim(),
        #                                                self.num_counts)
        self.passage_count_predictor = passage_count_predictor
        # Linear from self.passage_attention_to_count.output_dim --> 1
        self.passage_count_hidden2logits = passage_count_hidden2logits

        self.dotprod_matrix_attn = DotProductMatrixAttention()

        self.filter_matrix_attention = LinearMatrixAttention(tensor_1_dim=question_encoding_dim,
                                                             tensor_2_dim=passage_encoding_dim,
                                                             combination="x,y,x*y")

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_date_attention: MatrixAttention = BilinearMatrixAttention(matrix_1_dim=passage_encoding_dim,
                                                                                  matrix_2_dim=passage_encoding_dim)
        # self.passage_to_date_attention: MatrixAttention = LinearMatrixAttention(tensor_1_dim=passage_encoding_dim,
        #                                                                         tensor_2_dim=passage_encoding_dim,
        #                                                                         combination='x,y,x*y')

        # This computes a passage_to_passage attention, hopefully, for each token, putting a weight on date tokens
        # that are related to it.
        self.passage_to_num_attention: MatrixAttention = BilinearMatrixAttention(matrix_1_dim=passage_encoding_dim,
                                                                                 matrix_2_dim=passage_encoding_dim)
        # self.passage_to_num_attention: MatrixAttention = LinearMatrixAttention(tensor_1_dim=passage_encoding_dim,
        #                                                                        tensor_2_dim=passage_encoding_dim,
        #                                                                        combination='x,y,x*y')



        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x


# Deprecated
"""
@predicate_with_side_args(['question_attention', 'passage_attention'])
def find_PassageAttention(self, question_attention: Tensor,
                          passage_attention: Tensor = None) -> PassageAttention:

    # The passage attention is only used as supervision for certain synthetic questions
    if passage_attention is not None:
        return PassageAttention(passage_attention, debug_value="Supervised-Pattn-Used")

    question_attention = question_attention * self.question_mask
    # Shape: (encoding_size)
    weighted_question_vector = torch.sum(self.encoded_question * question_attention.unsqueeze(1), 0)

    # Shape: (passage_length, encoded_dim)
    passage_repr = self.modeled_passage if self.modeled_passage is not None else self.encoded_passage

    # Shape: (1, 1, 10)
    passage_logits_unsqueezed = self.parameters.q2p_matrix_attention(
                                                            weighted_question_vector.unsqueeze(0).unsqueeze(1),
                                                            passage_repr.unsqueeze(0))

    passage_logits = passage_logits_unsqueezed.squeeze(0).squeeze(0)

    passage_attention = allenutil.masked_softmax(passage_logits, mask=self.passage_mask, memory_efficient=True)

    passage_attention = passage_attention * self.passage_mask

    debug_value = ""
    if self._debug:
        qattn_vis_complete, qattn_vis_most = dlutils.listTokensVis(question_attention, self.metadata["question_tokens"])
        debug_value += f"Qattn: {qattn_vis_complete}"
        pattn_vis_complete, pattn_vis_most = dlutils.listTokensVis(passage_attention, self.metadata["passage_tokens"])
        debug_value += f"\nPattn: {pattn_vis_complete}"

    return PassageAttention(passage_attention, debug_value=debug_value)
"""

"""
# self.q2p_matrix_attention = LinearMatrixAttention(tensor_1_dim=question_encoding_dim,
#                                                   tensor_2_dim=passage_encoding_dim,
#                                                   combination="x,y,x*y")

self.q2p_matrix_attention = BilinearMatrixAttention(matrix_1_dim=question_encoding_dim,
                                                    matrix_2_dim=passage_encoding_dim)
"""

"""
# def compute_date_scores(self, passage_attention: Tensor):
    #     ''' Given a passage over passage token2date attention (normalized), and an additional passage attention
    #         for token importance, compute a distribution over (unique) dates in the passage.
    #
    #         Using the token_attention from the passage_attention, find the expected input_token2date_token
    #         distribution - weigh each row in passage_passage_token2date_attention and sum
    #         Use this date_token attention as scores for the dates they refer to. Since each date can be referred
    #         to in multiple places, we use scatter_add_ to get the total_score.
    #         Softmax over these scores is the date dist.
    #     '''
    #
    #     # (passage_length, passage_length)
    #     passage_passage_tokendate_attention = self.passage_passage_token2date_similarity * passage_attention.unsqueeze(1)
    #
    #     # maxidx = torch.argmax(passage_attention)
    #     # print(passage_attention[maxidx])
    #     # print(self.passage_passage_token2date_similarity[maxidx].sum())
    #     # print(self.passage_passage_token2date_similarity[maxidx+1].sum())
    #     # print(self.passage_passage_token2date_similarity[maxidx + 2].sum())
    #     # print(passage_passage_tokendate_attention[maxidx])
    #     # print()
    #
    #     # Shape: (passage_length, ) -- weighted average of distributions in above step
    #     # Attention value for each passage token to be a date associated to the query
    #     passage_datetoken_attention = passage_passage_tokendate_attention.sum(0)
    #
    #     masked_passage_tokenidx2dateidx = self.passage_tokenidx2dateidx * self.passage_datetokens_mask_long
    #
    #     ''' We first softmax scores over tokens then aggregate probabilities for same dates
    #         Another way could be to first aggregate scores, then compute softmx -- in that method if the scores are
    #         large, longer dates will get an unfair advantage; hence we keep it this way.
    #         For eg. [800, 800, 800] where first two point to same date, current method will get a dist of [0.66, 0.33],
    #         where as first-score-agg. will get [1.0, 0.0].
    #         Potential downside, if the scores are [500, 500, 1200] our method will get [small, ~1.0],
    #         where as first-score-agg. might get something more uniform.
    #         I think the bottomline is to control the scaling of scores.
    #     '''
    #     masked_passage_datetoken_probs = allenutil.masked_softmax(passage_datetoken_attention,
    #                                                               mask=self.passage_datetokens_mask_long,
    #                                                               memory_efficient=True)
    #     masked_passage_datetoken_probs = masked_passage_datetoken_probs * self.passage_datetokens_mask_float
    #
    #     ''' normalized method with method 2 '''
    #     date_distribution = passage_attention.new_zeros(self.num_passage_dates)
    #     date_distribution.scatter_add_(0, masked_passage_tokenidx2dateidx, masked_passage_datetoken_probs)
    #
    #     date_distribution = torch.clamp(date_distribution, min=1e-20, max=1 - 1e-20)
    #
    #     date_distribution_entropy = -1 * torch.sum(date_distribution * torch.log(date_distribution + 1e-40))
    #
    #     return date_distribution, date_distribution, date_distribution_entropy
    
    
    # def compute_num_distribution(self, passage_attention: Tensor):
    #     ''' Given a passage over passage token2num attention (normalized), and an additional passage attention
    #         for token importance, compute a distribution over (unique) nums in the passage.
    #         See compute_date_distribution for details
    #     '''
    #
    #     # print('-------------------------')
    #     # print(self.metadata['question_tokens'])
    #     # passage_tokens = self.metadata['passage_tokens']
    #     # print(f"NUmber of passage tokens : {len(passage_tokens)}")
    #     # attn, _= dlutils.listTokensVis(passage_attention, passage_tokens)
    #     # print(attn)
    #     # print()
    #
    #     # (passage_length, passage_length)
    #     passage_passage_tokennum_attention = self.passage_passage_token2num_similarity * passage_attention.unsqueeze(1)
    #
    #     # # tokkindices = (K, num_tokens)
    #     # _, number_indices = torch.topk(self.passage_tokenidx2numidx, k=10, dim=0)
    #     # number_indices = myutils.tocpuNPList(number_indices)
    #     # for number_idx in number_indices:
    #     #     token2numberscores = self.passage_passage_token2num_similarity[:, number_idx]
    #     #     _, top_tokens = torch.topk(token2numberscores, 5, dim=0)
    #     #     top_tokens = myutils.tocpuNPList(top_tokens)
    #     #     print(f"{self.metadata['passage_tokens'][number_idx]}")
    #     #     print([passage_tokens[x] for x in top_tokens])
    #     #     print(f"Sum: {torch.sum(self.passage_passage_token2num_similarity[:, number_idx])}")
    #     #     compvis, _ = dlutils.listTokensVis(self.passage_passage_token2num_similarity[:, number_idx],
    #     #                                        self.metadata['passage_tokens'])
    #     #     print(compvis)
    #
    #     # Shape: (passage_length, ) -- weighted average of distributions in above step
    #     # Attention value for each passage token to be a date associated to the query
    #     passage_numtoken_attention = passage_passage_tokennum_attention.sum(0)
    #
    #     # compvis, _  = dlutils.listTokensVis(self.passage_passage_token2num_similarity.sum(0), self.metadata['passage_tokens'])
    #     # print("Sum of Similarity Scores")
    #     # print(compvis)
    #     # compvis, _ = dlutils.listTokensVis(passage_numtoken_attention, self.metadata['passage_tokens'])
    #     # print("Weighted Sum")
    #     # print(compvis)
    #     # print('-------------------------')
    #
    #     masked_passage_tokenidx2numidx = self.passage_numtokens_mask_long * self.passage_tokenidx2numidx
    #     masked_passage_numtoken_scores = self.passage_numtokens_mask_float * passage_numtoken_attention
    #
    #     masked_passage_numtoken_probs = allenutil.masked_softmax(masked_passage_numtoken_scores,
    #                                                              mask=self.passage_numtokens_mask_float,
    #                                                              memory_efficient=True)
    #     masked_passage_numtoken_probs = masked_passage_numtoken_probs * self.passage_numtokens_mask_float
    #     # print(masked_passage_datetoken_scores)
    #     # print(masked_passage_datetoken_probs)
    #     # print()
    #     ''' normalized method with method 2 '''
    #     num_distribution = passage_attention.new_zeros(self.num_passage_nums)
    #     num_distribution.scatter_add_(0, masked_passage_tokenidx2numidx, masked_passage_numtoken_probs)
    #     num_scores = num_distribution
    #
    #     num_distribution = torch.clamp(num_distribution, min=1e-20, max=1 - 1e-20)
    #
    #     num_distribution_entropy = -1 * torch.sum(num_distribution * torch.log(num_distribution + 1e-40))
    #
    #     return num_distribution, num_distribution, num_distribution_entropy
"""