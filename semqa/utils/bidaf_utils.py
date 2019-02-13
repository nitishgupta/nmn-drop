import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.nn import util
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def forward_bidaf(bidaf_model: BidirectionalAttentionFlow,
                  embedded_question: torch.FloatTensor,
                  encoded_passage: torch.FloatTensor,
                  question_lstm_mask: torch.FloatTensor,
                  passage_lstm_mask: torch.FloatTensor):

    """
    Runs bidaf model on already embedded question and passage. This function can be used to run on sub-questions,
    or sub-questions concatenated.

    Parameters
    ----------
    question : ``torch.FloatTensor``
        This should be a (B, Q_T, D) sized tensor, say a slice of the output of the highway layer
    passage : ``torch.FloatTensor``
        This should be a (B, C_T, D) sized tensor, say a slice of the output of the highway layer
    question_lstm_mask: ``torch.FloatTensor``
        Question mask of shape (B, Q_T). Name is lstm mask to stay consistent with bidaf's code
    passage_lstm_mask: ``torch.FloatTensor``
        Passage mask of shape (B, C_T). Name is lstm mask to stay consistent with bidaf's code

    Returns:
    --------

    """
    batch_size = embedded_question.size(0)
    passage_length = encoded_passage.size(1)
    encoded_question = bidaf_model._dropout(bidaf_model._phrase_layer(embedded_question, question_lstm_mask))
    # encoded_passage = bidaf_model._dropout(bidaf_model._phrase_layer(embedded_passage, passage_lstm_mask))
    encoding_dim = encoded_question.size(-1)

    # Shape: (batch_size, passage_length, question_length)
    passage_question_similarity = bidaf_model._matrix_attention(encoded_passage, encoded_question)
    # Shape: (batch_size, passage_length, question_length)
    passage_question_attention = util.masked_softmax(passage_question_similarity, question_lstm_mask)
    # Shape: (batch_size, passage_length, encoding_dim)
    passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

    # We replace masked values with something really negative here, so they don't affect the
    # max below.
    masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                   question_lstm_mask.unsqueeze(1),
                                                   -1e7)
    # Shape: (batch_size, passage_length)
    question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
    # Shape: (batch_size, passage_length)
    question_passage_attention = util.masked_softmax(question_passage_similarity, passage_lstm_mask)
    # Shape: (batch_size, encoding_dim)
    question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
    # Shape: (batch_size, passage_length, encoding_dim)
    tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                passage_length,
                                                                                encoding_dim)

    # Shape: (batch_size, passage_length, encoding_dim * 4)
    final_merged_passage = torch.cat([encoded_passage,
                                      passage_question_vectors,
                                      encoded_passage * passage_question_vectors,
                                      encoded_passage * tiled_question_passage_vector],
                                     dim=-1)

    modeled_passage = bidaf_model._dropout(bidaf_model._modeling_layer(final_merged_passage, passage_lstm_mask))

    output_dict = {
        "encoded_question": encoded_question,
        "encoded_passage": encoded_passage,
        "passage_vector": question_passage_vector,
        "final_merged_passage": final_merged_passage,
        "modeled_passage": modeled_passage,
        "passage_question_attention": passage_question_attention
    }

    return output_dict




