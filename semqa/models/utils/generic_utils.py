from typing import List, Dict, Tuple
import torch
import allennlp.nn.util as allenutil
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


def embed_and_encode_ques_contexts(text_field_embedder: TextFieldEmbedder,
                                   qencoder: Seq2SeqEncoder,
                                   batch_size: int,
                                   question: Dict[str, torch.LongTensor],
                                   contexts: Dict[str, torch.LongTensor]):
        """ Embed and Encode question and contexts

        Parameters:
        -----------
        text_field_embedder: ``TextFieldEmbedder``
        qencoder: ``Seq2SeqEncoder``
        question: Dict[str, torch.LongTensor]
            Output of a TextField. Should yield tensors of shape (B, ques_length, D)
        contexts: Dict[str, torch.LongTensor]
            Output of a TextField. Should yield tensors of shape (B, num_contexts, ques_length, D)

        Returns:
        ---------
        embedded_questions: List[(ques_length, D)]
            Batch-sized list of embedded questions from the text_field_embedder
        encoded_questions: List[(ques_length, D)]
            Batch-sized list of encoded questions from the qencoder
        questions_mask: List[(ques_length)]
            Batch-sized list of questions masks
        encoded_ques_tensor: Shape: (batch_size, ques_len, D)
            Output of the qencoder
        questions_mask_tensor: Shape: (batch_size, ques_length)
            Questions mask as a tensor
        ques_encoded_final_state: Shape: (batch_size, D)
            For each question, the final state of the qencoder
        embedded_contexts: List[(num_contexts, context_length, D)]
            Batch-sized list of embedded contexts for each instance from the text_field_embedder
        contexts_mask: List[(num_contexts, context_length)]
            Batch-sized list of contexts_mask for each context in the instance

        """
        # Shape: (B, question_length, D)
        embedded_questions_tensor = text_field_embedder(question)
        # Shape: (B, question_length)
        questions_mask_tensor = allenutil.get_text_field_mask(question).float()
        embedded_questions = [embedded_questions_tensor[i] for i in range(batch_size)]
        questions_mask = [questions_mask_tensor[i] for i in range(batch_size)]

        # Shape: (B, ques_len, D)
        encoded_ques_tensor = qencoder(embedded_questions_tensor, questions_mask_tensor)
        # Shape: (B, D)
        ques_encoded_final_state = allenutil.get_final_encoder_states(encoded_ques_tensor,
                                                                      questions_mask_tensor,
                                                                      qencoder.is_bidirectional())
        encoded_questions = [encoded_ques_tensor[i] for i in range(batch_size)]

        # # contexts is a (B, num_contexts, context_length, *) tensors
        # (tokenindexer, indexed_tensor) = next(iter(contexts.items()))
        # num_contexts = indexed_tensor.size()[1]
        # # Making a separate batched token_indexer_dict for each context -- [{token_inderxer: (C, T, *)}]
        # contexts_indices_list: List[Dict[str, torch.LongTensor]] = [{} for _ in range(batch_size)]
        # for token_indexer_name, token_indices_tensor in contexts.items():
        #         print(f"{token_indexer_name}: {token_indices_tensor.size()}")
        #         for i in range(batch_size):
        #                 contexts_indices_list[i][token_indexer_name] = token_indices_tensor[i, ...]
        #
        # # Each tensor of shape (num_contexts, context_len, D)
        # embedded_contexts = []
        # contexts_mask = []
        # # Shape: (num_contexts, context_length, D)
        # for i in range(batch_size):
        #         embedded_contexts_i = text_field_embedder(contexts_indices_list[i])
        #         embedded_contexts.append(embedded_contexts_i)
        #         contexts_mask_i = allenutil.get_text_field_mask(contexts_indices_list[i]).float()
        #         contexts_mask.append(contexts_mask_i)

        embedded_contexts_tensor = text_field_embedder(contexts, num_wrapping_dims=1)
        contexts_mask_tensor = allenutil.get_text_field_mask(contexts, num_wrapping_dims=1).float()

        embedded_contexts = [embedded_contexts_tensor[i] for i in range(batch_size)]
        contexts_mask = [contexts_mask_tensor[i] for i in range(batch_size)]

        return (embedded_questions, encoded_questions, questions_mask,
                encoded_ques_tensor, questions_mask_tensor, ques_encoded_final_state,
                embedded_contexts, contexts_mask)