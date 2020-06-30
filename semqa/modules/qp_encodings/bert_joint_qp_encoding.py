from typing import Dict, Any

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields.text_field import TextFieldTensors
from semqa.modules.qp_encodings.qp_encoding import QPEncoding
from allennlp.common import Registrable


@QPEncoding.register("bert_joint_qp_encoding")
class BertJointQPEncoding(QPEncoding, Registrable):
    def __init__(self, text_field_embedder: TextFieldEmbedder, pad_token_id: int, sep_token_id: int):
        super().__init__(text_field_embedder=text_field_embedder)
        self._pad_token_id = pad_token_id
        self._sep_token_id = sep_token_id

    def get_representation(self,
                           question_passage: TextFieldTensors = None,
                           max_ques_len: int = None,
                           question: TextFieldTensors = None,
                           passage: TextFieldTensors = None):
        """BERT encoding for question and passage on concatenated input `[CLS] question .. [PAD] .. [SEP] passage [SEP]`

        Parameters:
        -----------
        question_passage:
            Joint question & passage wordpiece indices from "pretrained_transformer" token_indexer (w/ [CLS] [SEP])
        max_ques_len: `int`
            Number of word-pieces to which all questions have been padded

        Returns: All returns have [CLS] and [SEP] removed
        --------
        question_token_idxs / passage_token_idxs:
            Question / Passage wordpiece indices after removing [CLS] and [SEP]
        question_mask / passage_mask:
            Question / Passage mask
        encoded_question / encoded_passage: `(batch_size, seq_length, BERT_dim)`
            Contextual BERT representations
        pooled_encoding: `(batch_size, BERT_dim)`
            Pooled Question representation used to prime program-decoder. [CLS] embedding from bert-encoding
        """

        question_passage_tokens = question_passage["tokens"]["token_ids"]
        pad_mask = (question_passage_tokens != self._pad_token_id).long()
        question_passage["tokens"]["mask"] = pad_mask
        question_passage["tokens"]["type_ids"][:, max_ques_len + 1:] = 1  # type-id = 1 starting from [SEP] after Q

        # Shape: (batch_size, seqlen, bert_dim); (batch_size, bert_dim)
        # if not self.scaling_bert:
        bert_out = self._text_field_embedder(question_passage)
        bert_pooled_out = bert_out[:, 0, :]  # CLS embedding

        question_token_idxs = question_passage_tokens[:, 1:max_ques_len + 1]
        question_mask = (pad_mask[:, 1:max_ques_len + 1]).float()
        passage_token_idxs = question_passage_tokens[:, 1 + max_ques_len + 1:-1]  # Skip [CLS] Q [SEP] and last [SEP]

        passage_mask = (passage_token_idxs != self._pad_token_id) * \
                       (passage_token_idxs != self._sep_token_id)  # mask [SEP] token in passage
        passage_token_idxs = passage_token_idxs * passage_mask
        passage_mask = passage_mask.float()

        # Skip [CLS]; then the next max_ques_len tokens are question tokens
        encoded_question = bert_out[:, 1:max_ques_len + 1, :] * question_mask.unsqueeze(-1)
        encoded_passage = bert_out[:, 1 + max_ques_len + 1:-1, :] * passage_mask.unsqueeze(-1)

        return {"question_token_idxs": question_token_idxs,
                "passage_token_idxs": passage_token_idxs,
                "question_mask": question_mask,
                "passage_mask": passage_mask,
                "encoded_question": encoded_question,
                "encoded_passage": encoded_passage,
                "pooled_encoding": bert_pooled_out}
