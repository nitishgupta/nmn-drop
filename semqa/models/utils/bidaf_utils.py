import sys
import logging
from typing import List, Dict, Any, Tuple

import torch

from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as allenutil
import allennlp.common.util as alcommon_util
from allennlp.models.archival import load_archive
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow



from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction


class PretrainedBidafModelUtils(torch.nn.Module, Registrable):
    def __init__(self,
                 vocab: Vocabulary,
                 bidaf_model_path: str,
                 bidaf_wordemb_file: str,
                 dropout:float=0.0,
                 fine_tune_bidaf:bool=False) -> None:
        super(PretrainedBidafModelUtils, self).__init__()
        if bidaf_model_path is None:
            logger.info(f"NOT loading pretrained bidaf model. bidaf_model_path - not given")
            raise NotImplementedError

        logger.info(f"Loading BIDAF model: {bidaf_model_path}")
        bidaf_model_archive = load_archive(bidaf_model_path)
        self._bidaf_model: BidirectionalAttentionFlow = bidaf_model_archive.model

        # Needs to be a tuple
        # untuneable_parameter_prefixes = ('_text_field_embedder', '_highway_layer')

        # for n, p in self.bidaf_model.named_parameters(recurse=True):
        #     n: str = n
        #     if n.startswith(untuneable_parameter_prefixes):
        #         p.requires_grad = False

        if not fine_tune_bidaf:
            for p in self._bidaf_model.parameters():
                p.requires_grad = False

        logger.info(f"Bidaf model successfully loaded! Bidaf fine-tuning is set to {fine_tune_bidaf}")
        logger.info(f"Extending bidaf model's embedders based on the extended_vocab")
        logger.info(f"Preatrained word embedding file being used: {bidaf_wordemb_file}")

        # Extending token embedding matrix for Bidaf based on current vocab
        for key, _ in self._bidaf_model._text_field_embedder._token_embedders.items():
            token_embedder = getattr(self._bidaf_model._text_field_embedder, 'token_embedder_{}'.format(key))
            if isinstance(token_embedder, Embedding):
                token_embedder.extend_vocab(extended_vocab=vocab, pretrained_file=bidaf_wordemb_file)
        logger.info(f"Embedder for bidaf extended. New size: {token_embedder.weight.size()}")


        self.bidaf_encoder_bidirectional = self._bidaf_model._phrase_layer.is_bidirectional()
        self._bidaf_encoded_dim = self._bidaf_model._phrase_layer.get_output_dim()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x


    def embed_ques_passages(self, question, contexts):
        # Shape: (B, question_length, D)
        embedded_question = self._bidaf_model._highway_layer(self._bidaf_model._text_field_embedder(question))
        # Shape: (B, question_length)
        question_mask = allenutil.get_text_field_mask(question).float()

        (tokenindexer, indexed_tensor) = next(iter(contexts.items()))
        batch_size, num_contexts = indexed_tensor.size()[0], indexed_tensor.size()[1]
        # Making a separate batched token_indexer_dict for each context -- [{token_inderxer: (C, T, *)}]
        contexts_indices_list: List[Dict[str, torch.LongTensor]] = [{} for _ in range(batch_size)]
        for token_indexer_name, token_indices_tensor in contexts.items():
            # print(f"{token_indexer_name}  : {token_indices_tensor.size()}")
            for i in range(batch_size):
                # For a tensor shape (B, C, *), this will slice from dim-0 a tensor of shape (C, *)
                contexts_indices_list[i][token_indexer_name] = token_indices_tensor[i, ...]

        # Each tensor of shape (num_contexts, context_len, D)
        embedded_contexts_list = []
        contexts_mask_list = []
        # Shape: (num_contexts, context_length, D)
        for i in range(batch_size):
            embedded_contexts_i = self._bidaf_model._highway_layer(
                self._bidaf_model._text_field_embedder(contexts_indices_list[i]))
            embedded_contexts_list.append(embedded_contexts_i)
            contexts_mask_i = allenutil.get_text_field_mask(contexts_indices_list[i]).float()
            contexts_mask_list.append(contexts_mask_i)

        embedded_passages = torch.cat([x.unsqueeze(0) for x in embedded_contexts_list], dim=0)
        passages_mask = torch.cat([x.unsqueeze(0) for x in contexts_mask_list], dim=0)

        return embedded_question, embedded_passages, question_mask, passages_mask


    def encode_question(self, embedded_question, question_lstm_mask):
        encoded_question = self._dropout(self._bidaf_model._phrase_layer(embedded_question, question_lstm_mask))
        return encoded_question


    def encode_context(self, embedded_passage, passage_lstm_mask):
        encoded_passage = self._dropout(self._bidaf_model._phrase_layer(embedded_passage, passage_lstm_mask))
        return encoded_passage


    def forward_bidaf(self,
                      encoded_question: torch.FloatTensor,
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
        batch_size = encoded_question.size(0)
        passage_length = encoded_passage.size(1)
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._bidaf_model._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = allenutil.masked_softmax(passage_question_similarity, question_lstm_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = allenutil.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = allenutil.replace_masked_values(passage_question_similarity,
                                                            question_lstm_mask.unsqueeze(1),
                                                            -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = allenutil.masked_softmax(question_passage_similarity, passage_lstm_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = allenutil.weighted_sum(encoded_passage, question_passage_attention)
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

        modeled_passage = self._dropout(self._bidaf_model._modeling_layer(final_merged_passage,
                                                                          passage_lstm_mask))

        output_dict = {
            "encoded_question": encoded_question,
            "encoded_passage": encoded_passage,
            "passage_vector": question_passage_vector,
            "final_merged_passage": final_merged_passage,
            "modeled_passage": modeled_passage,
            "passage_question_attention": passage_question_attention
        }

        return output_dict


    def bidaf_reprs(self, question, contexts):
        # Shape: (B, ques_len, D), (B, num_contexts, context_len, D)
        (embedded_question_tensor, embedded_passages_tensor,
         question_mask_tensor, passages_mask_tensor) = self.embed_ques_passages(question, contexts)

        batch_size = embedded_question_tensor.size()[0]
        num_contexts = embedded_passages_tensor.size()[1]

        embedded_questions = []
        questions_mask = []
        embedded_contexts = []
        contexts_mask = []

        for i in range(0, batch_size):
            embedded_questions.append(embedded_question_tensor[i])
            embedded_contexts.append(embedded_passages_tensor[i])
            questions_mask.append(question_mask_tensor[i])
            contexts_mask.append(passages_mask_tensor[i])

        # Shape: (B, ques_len, D)
        encoded_ques_tensor = self.encode_question(embedded_question=embedded_question_tensor,
                                                   question_lstm_mask=question_mask_tensor)

        # Shape: (B, D)
        ques_encoded_final_state = allenutil.get_final_encoder_states(encoded_ques_tensor,
                                                                      question_mask_tensor,
                                                                      self.bidaf_encoder_bidirectional)

        # List of tensors: (question_len, D)
        encoded_questions = []
        # List of tensors: (num_contexts, context_len, D)
        encoded_contexts = []
        for i in range(0, batch_size):
            # Shape: (1, ques_len, D)
            # encoded_ques = self.encode_question(embedded_question=embedded_questions[i].unsqueeze(0),
            #                                     question_lstm_mask=questions_mask[i].unsqueeze(0))
            encoded_questions.append(encoded_ques_tensor[i])
            # Shape: (num_contexts, context_len, D)
            encoded_context = self.encode_context(embedded_passage=embedded_contexts[i],
                                                  passage_lstm_mask=contexts_mask[i])
            encoded_contexts.append(encoded_context)

        modeled_contexts = []
        for i in range(0, batch_size):
            # Shape: (question_len, D)
            encoded_ques = encoded_questions[i]
            ques_mask = questions_mask[i]
            encoded_ques_ex = encoded_ques.unsqueeze(0).expand(num_contexts, *encoded_ques.size())
            ques_mask_ex = ques_mask.unsqueeze(0).expand(num_contexts, *ques_mask.size())

            output_dict = self.forward_bidaf(encoded_question=encoded_ques_ex,
                                             encoded_passage=encoded_contexts[i],
                                             question_lstm_mask=ques_mask_ex,
                                             passage_lstm_mask=contexts_mask[i])

            # Shape: (num_contexts, context_len, D)
            modeled_context = output_dict['modeled_passage']
            modeled_contexts.append(modeled_context)

        return (ques_encoded_final_state,
                encoded_ques_tensor, question_mask_tensor,
                embedded_questions, questions_mask,
                embedded_contexts, contexts_mask,
                encoded_questions,
                encoded_contexts, modeled_contexts)

