from typing import List
import logging

import torch
import torch.nn.functional
from torch import Tensor

import allennlp.nn.util as allenutil
from allennlp.semparse import (DomainLanguage, ExecutionError, predicate, predicate_with_side_args)
from allennlp.semparse.domain_languages.domain_language import PredicateType

import datasets.hotpotqa.utils.constants as hpcons
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters

class Bool():
    def __init__(self, value: Tensor):
        self._value = self.clamp(value)
        self._bool_val = (self._value >= 0.5).float()

    def clamp(self, value: torch.Tensor):
        new_val = value.clamp(min=1e-3, max=1.0 - 1e-3)
        return new_val


class Bool1():
    def __init__(self, value: Tensor):
        self._value = value
        self.clamp()
        self._bool_val = (self._value >= 0.5).float()

    def clamp(self):
        self._value = self._value.clamp(min=1e-3, max=1.0 - 1e-3)


class Qstr():
    def __init__(self, value):
        self._value = value


class Qent():
    def __init__(self, value):
        self._value = value


class HotpotQALanguage(DomainLanguage):
    def __init__(self, start_types):
        super().__init__(start_types=start_types)

        self._execution_parameters: ExecutorParameters = None

        # These are mappings from Type predicates to the name of the type in the preprocessed data
        # Useful for (1) Finding starting rules that result in a particular type. (2) Finding the type of a logical prog.
        self.LANGTYPE_TO_TYPENAME_MAP = {
            PredicateType.get_type(Bool).name: hpcons.BOOL_TYPE
        }

        self.TYPENAME_TO_LANGTYPE_MAP = {
            hpcons.BOOL_TYPE: PredicateType.get_type(Bool).name
        }

        self.device_id = -1
        self._execution_parameters: ExecutorParameters = None

        # key telling which execution function to use to implement bool_qstr_qent function
        self.bool_qstr_qent_func = None

        self.q_nemenspan2entidx = None
        # Shape: (QLen, Q_d)
        self.ques_embedded = None
        # Shape: (QLen, Q_d)
        self.ques_encoded = None
        # Shape: (Qlen)
        self.ques_mask = None
        # Shape: (C, T, D) - The three token-level context representations
        self.contexts_embedded = None
        self.contexts_encoded = None
        self.contexts_modeled = None
        # Shape: (C, D)
        self.contexts_vec = None
        # Shape: (C, T)
        self.contexts_mask = None
        # Shape: (E, C, M, 2)
        self.ne_ent_mens = None
        # Shape: (E, C, M, 2)
        self.num_ent_mens = None
        # Shape: (E, C, M, 2)
        self.date_ent_mens = None
        # Dict from Qent -> Idx into self.q_nemens_grounding -- TODO(nitish) -- not used, but keep around
        self.q_nemens2groundingidx = None
        # Dict from Q_NE_men idx to EntityIdx corresonding to self.ne_ent_mens
        self.q_nemenspan2entidx = None

        # Dictionary from QStr/Qent span to it's tokens' tensor representation which is implemented by diff languages.
        # Should be a mapping from QuesSpan tensor to a tensor of shape (SpanLen, D)
        # These can be embeded or encoded repr. based on the question_token_repr_key
        self.questionspan2tokensembed = None
        self.questionspan2tokensrepr = None
        self.questionspan2mask = None
        # Map from Qent span_str to LongTensor(2) of it's span in the question
        self.entidx2spans = None
        # Map from Qent span_str to LongTensor(Qlen) - a multi-hot vector with 1s at span token locations
        self.entidx2spanvecs = None
        self.entitymention_idxs_vec = None

        # Used for debug printing while decoding in prediction mode
        self.debug = None
        self.metadata = None

    def typeobj_to_typename(self, obj):
        ''' For a object of a return-type of the language, return the name of type used in the dataset.

        The obj input here should be a non-terminal type of the language defined, ideally the output of an execution
        '''
        # This is the name of the equivalent BasicType
        class_name =  PredicateType.get_type(obj.__class__).name
        if class_name in self.LANGTYPE_TO_TYPENAME_MAP:
            return self.LANGTYPE_TO_TYPENAME_MAP[class_name]
        else:
            raise ExecutionError(message="TypeClass to Type mapping not found! ClassName:{}".format(class_name))

    def typename_to_langtypename(self, typename):
        ''' For a given name of a type used in the dataset, return the mapped name of the type (BasicType) in the lang.

        The typename entered here should have a mapping to a non-terminal type of the language defined.
        '''
        # This is the name of the type from the dataset
        if typename in self.TYPENAME_TO_LANGTYPE_MAP:
            return self.TYPENAME_TO_LANGTYPE_MAP[typename]
        else:
            raise ExecutionError(message="Mapping from type name to Type not found! TypeName:{}".format(typename))

    def set_arguments(self, **kwargs):
        self.ques_embedded = kwargs["ques_embedded"]
        self.ques_encoded = kwargs["ques_encoded"]
        self.ques_mask = kwargs["ques_mask"]
        self.contexts_embedded = kwargs["contexts_embedded"]
        self.contexts_encoded = kwargs["contexts_encoded"]
        self.contexts_modeled = kwargs["contexts_modeled"]
        self.contexts_vec = kwargs["contexts_vec"]
        self.contexts_mask = kwargs["contexts_mask"]
        self.ne_ent_mens = kwargs["ne_ent_mens"]
        self.num_ent_mens = kwargs["num_ent_mens"]
        self.date_ent_mens = kwargs["date_ent_mens"]
        self.q_nemenspan2entidx = kwargs["q_nemenspan2entidx"]
        self.device_id = allenutil.get_device_of(self.ques_encoded)
        self.bool_qstr_qent_func = kwargs["bool_qstr_qent_func"]
        # self.snli_ques = kwargs["snli_ques"]
        # self.snli_contexts = kwargs["snli_contexts"]
        # self.snliques_mask = kwargs["snliques_mask"]
        # self.snlicontexts_mask = kwargs["snlicontexts_mask"]
        # Keep commented for use later
        # print(f"self.ques_embedded: {self.ques_embedded.size()}")
        # print(f"self.ques_mask: {self.ques_mask.size()}")
        # print(f"self.contexts: {self.contexts.size()}")
        # print(f"self.contexts_mask: {self.contexts_mask.size()}")
        # print(f"self.ne_ent_mens: {self.ne_ent_mens.size()}")
        # print(f"self.date_ent_mens: {self.date_ent_mens.size()}")
        # print(f"self.num_ent_mens: {self.num_ent_mens.size()}")
        # print(f"self.q_qstr2idx: {self.q_qstr2idx}")
        # print(f"self.q_qstr_spans: {self.q_qstr_spans.size()}")
        # print(f"self.q_nemenspan2entidx: {self.q_nemenspan2entidx}")


    def set_execution_parameters(self, execution_parameters: ExecutorParameters):
        self._execution_parameters: ExecutorParameters = execution_parameters

    @predicate
    def bool_and(self, bool1: Bool1, bool2: Bool1) -> Bool:
        """ AND operation between two booleans """
        # return_val = bool1._value * bool2._value
        return_val = torch.min(torch.cat([bool1._value.unsqueeze(0), bool2._value.unsqueeze(0)], 0))
        return Bool(value=return_val)

    # return_val = torch.sigmoid(10.0 * bool1._value + 10.0 * bool2._value - 15.0)


    def closest_context(self,
                        ques_token_repr,
                        ques_mask,
                        contexts_token_repr,
                        contexts_mask,
                        question_attention=None):
        """ Finds the context_idx for the context most similar to the question.
            For each context, find the question_token to context_token dot-product similarity
            For each question_token, find the maximum similarity and sum over question tokens.
            Idx for the context with the maximum similarity is returned.

            In the case of a
                1) Language w/o sideargs: this is the usually the token_embeddings of the QEnt ques_span.
                2) Language w sideargs: ques_token_repr is the complete embedded question with an additional
                    argument, i.e. the question_attention, which is a normalized attention over question tokens.

            Parameters:
            -----------
            ques_token_repr: ``torch.FloatTensor``
                Shape: (ques_length, D)
            ques_mask: ``torch.FloatTensor``
                Shape: (ques_length)
            context_token_repr: ``torch.FloatTensor``
                Shape: (num_contexts, context_length, D)
            contexts_mask: ``torch.FloatTensor``
                Shape: (num_contexts, context_length)
            question_attention: `Optional` ``torch.FloatTensor``
                 Shape: (ques_length)
        """

        num_contexts = contexts_mask.size()[0]
        # (C, Qlen, D)
        ques_token_repr_ex = ques_token_repr.unsqueeze(0).expand(num_contexts, *ques_token_repr.size())
        # (C, Qlen, Clen)
        ques_context_token_similarity = self._execution_parameters._dotprod_matrixattn(ques_token_repr_ex,
                                                                                       contexts_token_repr)
        # (C, Qlen, Clen)
        ques_context_token_similarity = ques_context_token_similarity * contexts_mask.unsqueeze(1)
        ques_context_token_similarity = ques_context_token_similarity * ques_mask.unsqueeze(0).unsqueeze(2)

        masked_ques_context_similarity = allenutil.replace_masked_values(tensor=ques_context_token_similarity,
                                                                         mask=contexts_mask.unsqueeze(1),
                                                                         replace_with=-1e7)
        # (C, Qlen, Clen)
        masked_ques_context_similarity = allenutil.replace_masked_values(tensor=masked_ques_context_similarity,
                                                                         mask=ques_mask.unsqueeze(0).unsqueeze(2),
                                                                         replace_with=-1e7)
        # (C, Qlen)
        cwise_maxquestoken_context_similarity, _ = torch.max(masked_ques_context_similarity, dim=2)

        if question_attention is not None:
            cwise_maxquestoken_context_similarity = cwise_maxquestoken_context_similarity * \
                                                        question_attention.unsqueeze(0)

        # Shape: (C)
        question_context_similarity = cwise_maxquestoken_context_similarity.sum(1)

        closest_context_idx = torch.argmax(question_context_similarity)

        return closest_context_idx


    # @predicate
    # def bool_or(self, bool1: Bool1, bool2: Bool1) -> Bool:
    #     """ OR operation between two booleans """
    #     return_val = torch.max(torch.cat([bool1._value.unsqueeze(0), bool2._value.unsqueeze(0)], 0))
    #     return Bool(value=return_val)

