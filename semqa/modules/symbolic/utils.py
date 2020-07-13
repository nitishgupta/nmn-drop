from allennlp.nn import util as allenutil


def compute_token_symbol_alignments(
        modeled_passage, passage_mask, passageidx2symbolidx, passage_to_symbol_attention_params
):
    """Compute the passage_token-to-passage_date alignment matrix.

    Args:
    -----
        modeled_passage: (batch_size, passage_length, hidden_dim)
            Contextual passage repr.
        passage_mask: (batch_size, passage_length)
            Passage mask
        passageidx2dateidx: (batch_size, passage_length)
            For date-tokens, the index of the date-entity it belongs to, o/w masked with value = -1
        passage_to_date_attention_params: Some matrix-attention parameterization for computing the alignment matrix

    Returns:
    --------
        pasage_passage_token2symbol_aligment: (batch_size, passage_length, passage_length)
            Alignment matrix from passage_token (dim=1) to passage_date (dim=2)
            Should be masked in dim=2 for tokens that are not date-tokens
    """
    # ### Passage Token - Date Alignment
    # Shape: (batch_size, passage_length, passage_length)
    passage_passage_token2symbol_similarity = passage_to_symbol_attention_params(modeled_passage, modeled_passage)
    passage_passage_token2symbol_similarity = passage_passage_token2symbol_similarity * passage_mask.unsqueeze(1)
    passage_passage_token2symbol_similarity = passage_passage_token2symbol_similarity * passage_mask.unsqueeze(2)

    # Shape: (batch_size, passage_length) -- masking for number tokens in the passage
    passage_tokenidx2symbolidx_mask = (passageidx2symbolidx > -1).float()
    # Shape: (batch_size, passage_length, passage_length)
    passage_passage_token2symbol_similarity = (
        passage_passage_token2symbol_similarity * passage_tokenidx2symbolidx_mask.unsqueeze(1)
    )
    # Shape: (batch_size, passage_length, passage_length)
    pasage_passage_token2symbol_aligment = allenutil.masked_softmax(
        passage_passage_token2symbol_similarity,
        mask=passage_tokenidx2symbolidx_mask.unsqueeze(1).bool(),
        memory_efficient=True,
    )
    return pasage_passage_token2symbol_aligment