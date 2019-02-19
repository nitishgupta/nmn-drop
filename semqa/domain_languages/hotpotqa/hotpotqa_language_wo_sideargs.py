from typing import Callable, List, Dict, Tuple
import logging
from functools import cmp_to_key

import torch
import torch.nn.functional
from torch import Tensor

from allennlp.common.registrable import Registrable
import allennlp.nn.util as allenutil
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.modules.similarity_functions import SimilarityFunction, BilinearSimilarity
from allennlp.semparse import (DomainLanguage, ExecutionError, ParsingError,
                               predicate, predicate_with_side_args, util)
from allennlp.semparse.domain_languages.domain_language import PredicateType
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention

from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters
from semqa.domain_languages.hotpotqa.hotpotqa_language import Qstr, Qent, Bool, Bool1, HotpotQALanguage

import datasets.hotpotqa.utils.constants as hpcons


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def getspanfromaction(actionstr):
    """ Action string of the type Qent -> QENT:Tokens@DELIM@START@DELIM@END or just RHS
        Return (START, END)
    """
    splittokens = actionstr.split(hpcons.SPAN_DELIM)
    start, end = int(splittokens[-2]), int(splittokens[-1])
    return (start, end)


class HotpotQALanguageWOSideArgs(HotpotQALanguage):
    def __init__(self, qstr_qent_spans: List[str]):
        super().__init__(start_types={Bool})

        self._add_constants(qstr_qent_spans=qstr_qent_spans)

    def _add_constants(self, qstr_qent_spans: List[str]):
        for span in qstr_qent_spans:
            if span.startswith(hpcons.QSTR_PREFIX):
                self.add_constant(name=span, value=Qstr(span), type_=Qstr)
            elif span.startswith(hpcons.QENT_PREFIX):
                # Question NE men span
                self.add_constant(name=span, value=Qent(span), type_=Qent)


    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """
        (self.questionspan2tokensrepr,
         self.questionspan2tokensembed,
         self.questionspan2mask) = self._preprocess_ques_representations()


    def _preprocess_ques_representations(self):
        """ Makes questionspan 2 tokenrepr (Shape: (SpanLen, D)) dictionaries.
            First is based on the question_token_repr_key and the second question_token_embedding
            A dictionary from questionspan 2 mask (Shape: (SpanLen)) is also made
        """

        questionspan2tokensrepr = {}
        questionspan2tokensembed = {}
        questionspan2mask = {}

        ques_tokenrepr_key = self._execution_parameters._question_token_repr_key
        # This tensor should be (QLen, D)
        if ques_tokenrepr_key == 'embeded':
            question_token_repr = self.ques_embedded
        elif ques_tokenrepr_key == 'encoded':
            question_token_repr = self.ques_encoded
        else:
            raise NotImplementedError(f"Question Token Repr Key not recognized: {ques_tokenrepr_key}")

        ques_tokenembed = self.ques_embedded
        ques_mask = self.ques_mask

        qstr_actions = self.get_nonterminal_productions()['Qstr']
        qent_actions = self.get_nonterminal_productions()['Qent']

        for qstraction in qstr_actions:
            (start, end) = getspanfromaction(qstraction)
            span_str = qstraction.split(' -> ')[1]
            questionspan2tokensrepr[span_str] = question_token_repr[start:end + 1]
            questionspan2tokensembed[span_str] = ques_tokenembed[start:end + 1]
            questionspan2mask[span_str] = ques_mask[start: end + 1]

        for qentaction in qent_actions:
            (start, end) = getspanfromaction(qentaction)
            span_str = qentaction.split(' -> ')[1]
            questionspan2tokensrepr[span_str] = question_token_repr[start:end + 1]
            questionspan2tokensembed[span_str] = ques_tokenembed[start:end + 1]
            questionspan2mask[span_str] = ques_mask[start: end + 1]

        return questionspan2tokensrepr, questionspan2tokensembed, questionspan2mask


    def _get_gold_actions(self) -> Tuple[str, str, str]:
        def longestfirst_cmp(t1, t2):
            # when arranging in ascending, longest with greater end should come first
            len1 = t1[1] - t1[0]
            len2 = t2[1] - t2[0]
            if len2 < len1:
                return -1
            elif len1 < len2:
                return 1
            else:
                if t1[1] < t2[1]:
                    return 1
                elif t1[1] > t2[1]:
                    return -1
                else:
                    return 0
        def startfirst_cmp(t1, t2):
            # When arranging in ascending, lower start but longest within should come first
            if t1[0] < t2[0]:
                return -1
            elif t1[0] > t2[0]:
                return 1
            else:
                if t1[1] < t2[1]:
                    return 1
                elif t1[1] > t2[1]:
                    return -1
                else:
                    return 0



        """ For boolean questions of the kind: XXX E1 yyy E2 zzz zzz zzz

        We want
            Two Qent -> QENT:Tokens@DELIM@START@DELIM@END' style actions
            One Qstr -> QSTR:Tokens@DELIM@START@DELIM@END' action for zz zzz zzz

        Assume that the first two entities are first/second Qent actions, and longest span that starts right after
        second entity is the Qstr action
        """
        qent_actions = self.get_nonterminal_productions()['Qent']
        qstr_actions = self.get_nonterminal_productions()['Qstr']

        # Dict from span tuple to action str. this will make sorting easier.
        span2qentactions = {}
        span2qstractions = {}
        for a in qent_actions:
            span = getspanfromaction(a)
            span2qentactions[span] = a
        for a in qstr_actions:
            span = getspanfromaction(a)
            span2qstractions[span] = a

        sorted_qentspans = sorted(span2qentactions.keys(),
                                  key=cmp_to_key(startfirst_cmp),
                                  reverse=False)
        sorted_qstrspans = sorted(span2qstractions.keys(),
                                  key=cmp_to_key(longestfirst_cmp),
                                  reverse=False)
        if len(sorted_qentspans) >= 2:
            gold_qent1span = sorted_qentspans[0]
            gold_qent2span = sorted_qentspans[1]
        else:
            gold_qent1span = sorted_qentspans[0]
            gold_qent2span = sorted_qentspans[0]


        gold_qent2end = gold_qent2span[1]
        gold_qstr_span = None
        for qstrspan in sorted_qstrspans:
            if qstrspan[0] > gold_qent2end:
                gold_qstr_span = qstrspan
                break

        if gold_qstr_span is None:
            gold_qstr_span = sorted_qstrspans[-1]

        qent1_action = span2qentactions[gold_qent1span]
        qent2_action = span2qentactions[gold_qent2span]
        qstr_action = span2qstractions[gold_qstr_span]

        # print(qent1_action)
        # print(qent2_action)
        # print(qstr_action)

        return qent1_action, qent2_action, qstr_action

    @predicate
    def bool_qent_qstr(self, qent: Qent, qstr: Qstr) -> Bool1:
        returnval = None
        if self.bool_qstr_qent_func == 'mentions':
            # returnval = self.bool_qent_qstr_wmens(qent=qent, qstr=qstr)
            raise NotImplementedError
        elif self.bool_qstr_qent_func == 'slicebidaf':
            returnval = self.bool_qent_qstr_slicebidaf(qent_obj=qent, qstr_obj=qstr)
        elif self.bool_qstr_qent_func == 'rawbidaf':
            returnval = self.bool_qent_qstr_rawbidaf(qent_obj=qent, qstr_obj=qstr)
        elif self.bool_qstr_qent_func == 'snli':
            returnval = self.bool_qent_qstr_snli(qent_obj=qent, qstr_obj=qstr)
        else:
            raise NotImplementedError

        return returnval


    def bool_qent_qstr_snli(self, qent_obj: Qent, qstr_obj: Qstr) -> Bool1:
        qent = qent_obj._value
        qstr = qstr_obj._value

        (qent_start, qent_end) = getspanfromaction(qent)
        (qstr_start, qstr_end) = getspanfromaction(qstr)

        qent_embed = self.ques_embedded[qent_start: qent_end + 1]
        qent_mask = self.ques_mask[qent_start: qent_end + 1]
        qstr_embed = self.ques_embedded[qstr_start: qstr_end + 1]
        qstr_mask = self.ques_mask[qstr_start: qstr_end + 1]

        # Shape: (Span1len + Span2len, D)
        q_ent_str_repr = torch.cat([qent_embed, qstr_embed], dim=0)
        q_ent_str_mask = torch.cat([qent_mask, qstr_mask], dim=0)

        contexts_mask = self.contexts_mask

        num_contexts = contexts_mask.size()[0]
        # (C, Qlen, D)
        ques_token_repr_ex = q_ent_str_repr.unsqueeze(0).expand(num_contexts, *q_ent_str_repr.size())
        q_ent_str_mask_ex = q_ent_str_mask.unsqueeze(0).expand(num_contexts, *q_ent_str_mask.size())

        # snli_output = self._execution_parameters._snli_model.forward_from_embeddings(
        #         embedded_premise=self.snli_contexts, embedded_hypothesis=ques_token_repr_ex,
        #         premise_mask=contexts_mask, hypothesis_mask=q_ent_str_mask_ex)
        snli_output = self._execution_parameters._decompatt.forward(
            embedded_premise=self.contexts_embedded, embedded_hypothesis=ques_token_repr_ex,
            premise_mask=contexts_mask, hypothesis_mask=q_ent_str_mask_ex)

        # C, 3
        output_probs = snli_output['label_probs']
        # print(f"OutputProbs:{output_probs}")
        # C
        boolean_probs = output_probs[:, 0]

        closest_context = self.closest_context(qent_embed, qent_mask, self.contexts_embedded, contexts_mask)
        # print(f"Closest ContextIdx: {closest_context}")

        ans_prob = boolean_probs[closest_context]

        # print(f"Ansprob: {ans_prob}")
        # print()

        return Bool1(value=ans_prob)


    def bool_qent_qstr_slicebidaf(self, qent_obj: Qent, qstr_obj: Qstr) -> Bool1:
        qent = qent_obj._value
        qstr = qstr_obj._value

        # Already sliced representation of token repr based on question_token_repr_key
        # Shape: (SpanLen, D)
        qent_embedded = self.questionspan2tokensembed[qent]
        qstr_embedded = self.questionspan2tokensembed[qstr]

        qent_token_repr = self.questionspan2tokensrepr[qent]
        qstr_token_repr = self.questionspan2tokensrepr[qstr]
        qent_mask = self.questionspan2mask[qent]
        qstr_mask = self.questionspan2mask[qstr]

        # Concatenation of spliced Qent and Qstr repr
        # Shape: (Span1len + Span2len, D)
        q_ent_str_repr = torch.cat([qent_token_repr, qstr_token_repr], dim=0)
        q_ent_str_mask = torch.cat([qent_mask, qstr_mask], dim=0)

        context_token_repr = None
        if self._execution_parameters._context_token_repr_key == 'embeded':
            context_token_repr = self.contexts_embedded
        elif self._execution_parameters._context_token_repr_key == 'encoded':
            context_token_repr = self.contexts_encoded
        elif self._execution_parameters._context_token_repr_key == 'modeled':
            context_token_repr = self.contexts_modeled
        context_mask = self.contexts_mask

        boolean_prob = self.boolean_q_c_token_repr(ques_token_repr=q_ent_str_repr,
                                                   ques_mask=q_ent_str_mask,
                                                   contexts_token_repr=context_token_repr,
                                                   contexts_mask=context_mask)
        return Bool1(value=boolean_prob)


    def bool_qent_qstr_rawbidaf(self, qent_obj: Qent, qstr_obj: Qstr) -> Bool1:
        qent = qent_obj._value
        qstr = qstr_obj._value

        # print(f"QENT: {qent.split(hpcons.SPAN_DELIM)}")
        # print(f"QSTR: {qstr.split(hpcons.SPAN_DELIM)}")

        # For are (*, D)
        qent_embedded = self.questionspan2tokensembed[qent]
        qent_mask = self.questionspan2mask[qent]

        qstr_embedded = self.questionspan2tokensembed[qstr]
        qstr_mask = self.questionspan2mask[qstr]

        # This can go into bidaf utils forward
        # Shape: (Span1len + Span2len, D)
        q_ent_str_embed = torch.cat([qent_embedded, qstr_embedded], dim=0)
        q_ent_str_mask = torch.cat([qent_mask, qstr_mask], dim=0)

        # Shape : (C, T, D)
        context_embeded = self.contexts_embedded
        context_encoded = self.contexts_encoded

        # (C, T)
        context_mask = self.contexts_mask
        num_contexts = context_encoded.size(0)

        # Shape: (1, ques_len, D)
        encoded_q_ent_str = self._execution_parameters._bidafutils.encode_question(
            embedded_question=q_ent_str_embed.unsqueeze(0),
            question_lstm_mask=q_ent_str_mask.unsqueeze(0))

        # (C, Qlen, D) -- copy of the question C times to get independent reprs
        q_ent_str_encoded_ex = encoded_q_ent_str.expand([num_contexts, *q_ent_str_embed.size()])
        q_ent_str_mask_ex = q_ent_str_mask.unsqueeze(0).expand([num_contexts, *q_ent_str_mask.size()])


        output_dict = self._execution_parameters._bidafutils.forward_bidaf(encoded_question=q_ent_str_encoded_ex,
                                                                           encoded_passage=context_encoded,
                                                                           question_lstm_mask=q_ent_str_mask_ex,
                                                                           passage_lstm_mask=context_mask)

            # bidaf_utils.forward_bidaf(bidaf_model=self._execution_parameters._bidafmodel,
            #                                     embedded_question=q_ent_str_embed_ex,
            #                                     encoded_passage=context_encoded,
            #                                     question_lstm_mask=q_ent_str_mask_ex,
            #                                     passage_lstm_mask=context_mask)

        # (C, T1, D) -- T1 = Len of Qent + Qstr
        q_embeded = q_ent_str_embed
        q_encoded = output_dict["encoded_question"][0]
        # (T1)
        q_mask = q_ent_str_mask

        # Shape: (C, T2, D) -- T2 is the length of the context. This is modeled based on the truncated question
        context_modeled = output_dict["modeled_passage"]
        # Shape: (C, D) == (C, 200) for bidaf
        context_vec = output_dict["passage_vector"]    # Since only one passage

        ques_token_repr = None
        if self._execution_parameters._question_token_repr_key == 'embeded':
            ques_token_repr = q_embeded
        elif self._execution_parameters._question_token_repr_key == 'encoded':
            ques_token_repr = q_encoded
        else:
            raise NotImplementedError

        context_token_repr = None
        if self._execution_parameters._context_token_repr_key == 'embeded':
            context_token_repr = context_embeded
        elif self._execution_parameters._context_token_repr_key == 'encoded':
            context_token_repr = context_encoded
        elif self._execution_parameters._context_token_repr_key == 'modeled':
            context_token_repr = context_modeled
        else:
            raise NotImplementedError

        boolean_prob = self.boolean_q_c_token_repr(ques_token_repr=ques_token_repr,
                                                   ques_mask=q_mask,
                                                   contexts_token_repr=context_token_repr,
                                                   contexts_mask=context_mask)

        return Bool1(value=boolean_prob)


    def boolean_q_c_token_repr(self,
                               ques_token_repr: torch.FloatTensor,
                               ques_mask: torch.FloatTensor,
                               contexts_token_repr: torch.FloatTensor,
                               contexts_mask: torch.FloatTensor):
        """ Given token reprs for question and context, return a boolean probability value

        Parameters:
        -----------
        ques_token_repr: Shape (Qlen, D)
        ques_mask: Shape (Qlen)
        contexts_token_repr: Shape: (C, Clen, D)
        contexts_mask: Shape: (C, Clen)
        """

        num_contexts = contexts_mask.size()[0]
        # (C, Qlen, D)
        ques_token_repr_ex = ques_token_repr.unsqueeze(0).expand(num_contexts, *ques_token_repr.size())
        # (C, Qlen, Clen)
        ques_context_token_similarity = self._execution_parameters._matrix_attention(ques_token_repr_ex,
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

        # Below the tensors store,
        # 1) max ques-context token similarity for each question token
        # 2) max context-ques token similarity for each context token
        # Shape: (C, Qlen)
        contextwise_qtoken_maxsimilarity = torch.max(masked_ques_context_similarity, dim=2)[0]
        # Shape: (C, Clen)
        contextwise_ctoken_maxsimilarity = torch.max(masked_ques_context_similarity, dim=1)[0]

        # Shape: (C, Qlen)
        cwise_qtoken_attention = allenutil.masked_softmax(contextwise_qtoken_maxsimilarity,
                                                          mask=ques_mask.unsqueeze(0))
        # Shape: (C, Clen)
        cwise_ctoken_attention = allenutil.masked_softmax(contextwise_ctoken_maxsimilarity,
                                                          mask=contexts_mask)

        # Shape: (C, D)
        cwise_averaged_quesrepr = (ques_token_repr.unsqueeze(0) * cwise_qtoken_attention.unsqueeze(2)).sum(1)

        # Shape: (C, D)
        cwise_averaged_contextrepr = (contexts_token_repr * cwise_ctoken_attention.unsqueeze(2)).sum(1)

        # Shape: C
        context_scores = self._execution_parameters._quescontext_bilinear(cwise_averaged_quesrepr,
                                                                          cwise_averaged_contextrepr)

        # print(f"context_avg_attention: {context_avg_attention}")
        contextwise_prob_values = torch.sigmoid(context_scores)
        # print(f"contextwise_prob_values: {contextwise_prob_values}")

        boolean_prob = torch.max(contextwise_prob_values)
        return boolean_prob



    def bool_qent_qstr_wmens(self, qent: Qent, qstr: Qstr) -> Bool1:
        qent_att = qent._value * self.ques_mask
        qstr_att = qstr._value * self.ques_mask

        # Shape: (Q_d)
        qstr_repr = (self.ques_encoded * qstr_att.unsqueeze(1)).sum(0)

        # Computing score/distribution over all entities mentioned in the question based on the qent ques_attention
        # Entity_prob is proportional to the sum of the attention values of the tokens in the entity's mention
        entidx2prob = {}
        entidx2score = {}
        total_score = 0.0
        for entidx, span_onehot_vecs in self.entidx2spanvecs.items():
            entity_score = sum([(spanvec * qent_att).sum() for spanvec in span_onehot_vecs])
            total_score += entity_score
            entidx2score[entidx] = entity_score
            # entidx2prob[entidx] = entity_score
        for entidx, entity_score in entidx2score.items():
            entidx2prob[entidx] = entity_score/total_score

        # Find the answer prob based on each entity mentioned in the question, and compute the expection later ...
        entidx2ansprob = {}
        for entidx in entidx2prob.keys():
            # Shape: (C, M, 2)
            qent_mens = self.ne_ent_mens[entidx]
            qent_mens_mask = (qent_mens[..., 0] >= 0).long()

            # Shape: (C, M, 2*D)
            qent_men_repr = self._execution_parameters._span_extractor(self.contexts_encoded, qent_mens,
                                                                       self.contexts_mask,
                                                                       qent_mens_mask)
            q_repr_cat = torch.cat([qstr_repr, qstr_repr], 0)
            qstr_repr_ex = q_repr_cat.unsqueeze(0).unsqueeze(1)
            scores = torch.sum(qstr_repr_ex * qent_men_repr, 2)
            probs = torch.sigmoid(scores) * qent_mens_mask.float()

            boolean_prob = torch.max(probs)  # .unsqueeze(0)
            entidx2ansprob[entidx] = boolean_prob

        # Computing the expected boolean_answer based on the qent probs
        expected_prob = 0.0
        for entidx, ent_ans_prob in entidx2ansprob.items():
            # ent_prob = entidx2prob[entidx]
            ent_prob = entidx2score[entidx]
            expected_prob += ent_prob * ent_ans_prob

        return Bool1(value=expected_prob)


if __name__=="__main__":

    # b = Bool(value=torch.FloatTensor([0.65]))
    # print(b._value)
    # print(b._bool_val)
    # print(PredicateType.get_type(b.__class__).name)
    # print(PredicateType.get_type(Bool).name)

    language = HotpotQALanguageWOSideArgs(qstr_qent_spans=["QSTR:Hi", "QENT:Nitish"])
    print(language._functions)

    all_prods = language.all_possible_productions()

    print("All prods:\n{}\n".format(all_prods))

    print(f"\n {language.get_nonterminal_productions()}")
    #
    # nonterm_prods = language.get_nonterminal_productions()
    # print("Non terminal prods:\n{}\n".format(nonterm_prods))
    #
    # functions = language._functions
    # print("Functions:\n{}\n".format(functions))
    #
    # function_types = language._function_types
    # print("Function Types:\n{}\n".format(function_types))
    #
    # logical_form = "(bool_qent_qstr QSTR:Hi QENT:Nitish)"
    # action_seq = language.logical_form_to_action_sequence(logical_form=logical_form)
    # print(action_seq)




