from typing import Dict, Any

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields.text_field import TextFieldTensors
from semqa.modules.qp_encodings.qp_encoding import QPEncoding
from allennlp.common import Registrable


@QPEncoding.register("bert_independent_qp_encoding")
class BertIndependentQPEncoding(QPEncoding, Registrable):
    def __init__(self, text_field_embedder: TextFieldEmbedder, pad_token_id: int, sep_token_id: int):
        super().__init__(text_field_embedder=text_field_embedder)
        self._pad_token_id = pad_token_id
        self._sep_token_id = sep_token_id

    def get_representation(self,
                           question: TextFieldTensors = None,
                           passage: TextFieldTensors = None,
                           question_passage: TextFieldTensors = None,
                           max_ques_len: int = None):
        """
        Parameters:
        -----------
        question:
            Question wordpiece indices from "pretrained_transformer" token_indexer (w/ [CLS] [SEP])
        passage:
            Passage wordpiece indices from "pretrained_transformer" token_indexer (w/ [CLS] [SEP])

        Returns: All returns have [CLS] and [SEP] removed
        --------
        question_token_idxs / passage_token_idxs::
            Question / Passage wordpiece indices after removing [CLS] and [SEP]
        question_mask / passage_mask:
            Question / Passage mask
        encoded_question / encoded_passage: `(batch_size, seq_length, BERT_dim)`
            Contextual BERT representations
        pooled_encoding: `(batch_size, BERT_dim)`
            Pooled Question representation used to prime program-decoder. [CLS] embedding from Question bert-encoding
        """

        def get_token_ids_and_mask(text_field_tensors: TextFieldTensors):
            """Get token_idxs and mask for a BERT TextField. """
            # Removing [CLS] and last [SEP], there might still be [SEP] for shorter texts
            token_idxs = text_field_tensors["tokens"]["token_ids"][:, 1:-1]
            # mask for [SEP] and [PAD] tokens
            mask = (token_idxs != self._pad_token_id) * (token_idxs != self._sep_token_id)
            # Mask [SEP] and [PAD] within the question
            token_idxs = token_idxs * mask
            mask = mask.float()
            return token_idxs, mask

        question_bert_out = self._text_field_embedder(question)
        passage_bert_out = self._text_field_embedder(passage)

        bert_pooled_out = question_bert_out[:, 0, :]  # CLS embedding

        # Remove [CLS] and last [SEP], and mask internal [SEP] and [PAD]
        question_token_idxs, question_mask = get_token_ids_and_mask(question)
        passage_token_idxs, passage_mask = get_token_ids_and_mask(passage)

        # Skip [CLS] and last [SEP]
        encoded_question = question_bert_out[:, 1:-1, :] * question_mask.unsqueeze(-1)
        encoded_passage = passage_bert_out[:, 1:-1, :] * passage_mask.unsqueeze(-1)

        return {"question_token_idxs": question_token_idxs,
                "passage_token_idxs": passage_token_idxs,
                "question_mask": question_mask,
                "passage_mask": passage_mask,
                "encoded_question": encoded_question,
                "encoded_passage": encoded_passage,
                "pooled_encoding": bert_pooled_out}
