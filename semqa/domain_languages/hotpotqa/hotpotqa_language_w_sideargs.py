from typing import List, Tuple
import logging

import torch
import torch.nn.functional
from torch import Tensor

import allennlp.nn.util as allenutil
from allennlp.semparse import (DomainLanguage, ExecutionError, predicate, predicate_with_side_args)
from allennlp.semparse.domain_languages.domain_language import PredicateType

import datasets.hotpotqa.utils.constants as hpcons
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters
from semqa.domain_languages.hotpotqa.hotpotqa_language import Qstr, Qent, Bool, Bool1, HotpotQALanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class HotpotQALanguageWSideArgs(HotpotQALanguage):
    def __init__(self):
        super().__init__(start_types={Bool})


    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """
        self._preprocess_ques_NE_menspans()


    def _get_gold_actions(self) -> Tuple[torch.FloatTensor, torch.FloatTensor , torch.FloatTensor]:
        return self._make_gold_attentions()

    def _make_gold_attentions(self) -> Tuple[torch.FloatTensor, torch.FloatTensor , torch.FloatTensor]:
        """ For questions of the kind: XXX E1 yyy E2 zzz zzz zzz

        We want
            Two qent attentions, one for E1 and E2
            One qstr attention, for zz zzz zzz

        Assume that the first entity is  Qent_1, second is Qent_2, and all tokens after that are Qstr
        """
        qlen = self.ques_mask.size()
        if self.device_id > -1:
            qent1 = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
            qent2 = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
            qstr = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
        else:
            qent1 = torch.FloatTensor(qlen).fill_(0)
            qent2 = torch.FloatTensor(qlen).fill_(0)
            qstr = torch.FloatTensor(qlen).fill_(0)

        # List of (start,end) tuples
        ne_mens = []
        for span in self.q_nemenspan2entidx.keys():
            # span is tokens@DELIM@START@END
            split_tokens = span.split(hpcons.SPAN_DELIM)
            start, end = int(split_tokens[-2]), int(split_tokens[-1])
            ne_mens.append((start, end))
        # Sort based on start token - end is exclusive
        sorted_ne_mens: List[Tuple[int, int]] = sorted(ne_mens, key=lambda x: x[0])

        # Assume that first mention is E1 and second is E2.
        if len(sorted_ne_mens) < 2:
            e1 = sorted_ne_mens[0]
            e2 = sorted_ne_mens[0]
        else:
            e1 = sorted_ne_mens[0]
            e2 = sorted_ne_mens[1]


        qent1[e1[0]:e1[1] + 1] = 1.0/float(e1[1]-e1[0]+1)
        qent2[e2[0]:e2[1] + 1] = 1.0/float(e2[1]-e2[0]+1)
        # Doesn't take mask into account -- but maybe that's alright
        qstr_len = torch.sum(self.ques_mask) - (e2[1] + 1)
        qstr_len = 1.0 if qstr_len == 0 else qstr_len
        qstr[e2[1] + 1:] = 1.0/float(qstr_len)

        qent1 = qent1 * self.ques_mask
        qent2 = qent2 * self.ques_mask
        qstr = qstr * self.ques_mask

        return qent1, qent2, qstr


    def _preprocess_ques_NE_menspans(self):
        """ Preprocess Ques NE mens to extract spans for each of the entity mentioned.
        Makes two dictionaries, (1) Stores a list of mention spans (Tensor(2)) in the question for each entity
        mentioned. (2) For each entity mentioned, a list of binary-vectors the size of Qlen, with 1s for tokens in the
        span. For example, if the question (len=5) has two mentions, [1,2] and [3,4] linking to entities e1 and e2.
        The two dictionaries made will be {e1: [Tensor([1,2])], e2: [Tensor([3,4])]} and
        {e1: [FloatTensor([0,..,1,1,.0])], e2: [FloatTensor([0...,0,1,1])]}

        Additionally, self.entitymention_idxs_vec -- a vector of QLen is made that contains 1 at all locations
        that are NE mens is made; this is used for Qent attention loss.
        """
        q_len = self.ques_mask.size()
        # Use the qent attention to get a distribution over entities.
        # Use the self.q_nemenspan2entidx map to extract spans for entities in the question.
        # Since an entity can be mentioned multiple times, we'll create a dictionary from entityidx2spans
        self.entidx2spans = {}
        self.entidx2spanvecs = {}

        if self.device_id > -1:
            self.entitymention_idxs_vec = torch.cuda.FloatTensor(q_len, device=self.device_id).fill_(0)
        else:
            self.entitymention_idxs_vec = torch.FloatTensor(q_len).fill_(0)

        for span, entity_idx in self.q_nemenspan2entidx.items():
            # Span: QENT:TOKEN_1@DELIM@TOKEN_2...TOKEN_N@DELIM@SPAN_START@DELIM@SPAN_END
            # Start and end are inclusive
            split_tokens = span.split(hpcons.SPAN_DELIM)
            start, end = int(split_tokens[-2]), int(split_tokens[-1])
            if self.device_id > -1:
                span_tensor = torch.cuda.LongTensor([start, end], device=self.device_id)
                onehot_span_tensor = torch.cuda.FloatTensor(q_len, device=self.device_id).fill_(0)
                onehot_span_tensor[start:end + 1] = 1
            else:
                span_tensor = torch.LongTensor([start, end])
                onehot_span_tensor = torch.FloatTensor(q_len).fill_(0)
                onehot_span_tensor[start:end + 1] = 1
            if entity_idx not in self.entidx2spans:
                self.entidx2spans[entity_idx] = []
                self.entidx2spanvecs[entity_idx] = []
            self.entidx2spans[entity_idx].append(span_tensor)
            self.entidx2spanvecs[entity_idx].append(onehot_span_tensor)
            self.entitymention_idxs_vec += onehot_span_tensor


    @predicate_with_side_args(['question_attention'])
    def find_Qstr(self, question_attention: torch.FloatTensor) -> Qstr:
        return Qstr(value=question_attention)


    @predicate_with_side_args(['question_attention'])
    def find_Qent(self, question_attention: torch.FloatTensor) -> Qent:
        return Qent(value=question_attention)

    @predicate
    def bool_qent_qstr(self, qent: Qent, qstr: Qstr) -> Bool:

        returnval = None
        if self.bool_qstr_qent_func == 'mentions':
            returnval = self.bool_qent_qstr_wmens(qent=qent, qstr=qstr)
        elif self.bool_qstr_qent_func == 'context':
            returnval = self.bool_qent_qstr_wcontext(qent=qent, qstr=qstr)
        elif self.bool_qstr_qent_func == 'snli':
            returnval = self.bool_qent_qstr_snli(qent=qent, qstr=qstr)
        else:
            raise NotImplementedError
        return returnval


    def bool_qent_qstr_snli(self, qent: Qent, qstr: Qstr) -> Bool:
        qent_att = qent._value * self.ques_mask
        qstr_att = qstr._value * self.ques_mask

        # Shape: (ques_len, D)
        ques_embedded = self.ques_embedded
        ques_mask = self.ques_mask

        # Shape: (num_contexts, context_len, D)
        contexts_embedded = self.contexts_embedded
        contexts_mask = self.contexts_mask

        closest_context, context_similarity_dist = self.closest_context(ques_embedded,
                                                                        ques_mask,
                                                                        contexts_embedded,
                                                                        contexts_mask,
                                                                        question_attention=qent_att)

        ques_att = qent_att + qstr_att

        num_contexts = contexts_mask.size()[0]
        # (C, Qlen, D)
        ques_token_repr_ex = ques_embedded.unsqueeze(0).expand(num_contexts, *ques_embedded.size())
        q_mask_ex = ques_mask.unsqueeze(0).expand(num_contexts, *ques_mask.size())
        ques_att_ex = ques_att.unsqueeze(0).expand(num_contexts, *ques_att.size())

        debug_kwargs = {}

        snli_output = self._execution_parameters._decompatt.forward(
            embedded_premise=contexts_embedded, embedded_hypothesis=ques_token_repr_ex,
            premise_mask=contexts_mask, hypothesis_mask=q_mask_ex,
            debug=False, question_attention=ques_att_ex, **debug_kwargs)
            # self.debug

        # Shape: (C, 2)
        output_probs = snli_output['label_probs']

        # C
        boolean_probs = output_probs[:, 0]

        # ans_prob = boolean_probs[closest_context]
        ans_prob = (context_similarity_dist * boolean_probs).sum()

        if self.debug:
            print(self.metadata['question'])
            context_texts = self.metadata['contexts']
            for c in context_texts:
                print(f"{c}")
            print(f"QStr: {qstr_att}")
            print(f"QEnt: {qent_att}")
            print(f"ContextSimDist: {context_similarity_dist}")
            print(f"BoolProbs :{boolean_probs}")
            print(f"AnsProb: {ans_prob}")
            print()

        return Bool(value=ans_prob)


    def bool_qent_qstr_wcontext(self, qent: Qent, qstr: Qstr) -> Bool1:
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
            entidx2prob[entidx] = entity_score / total_score

        # Find the answer prob based on each entity mentioned in the question, and compute the expection later ...
        entidx2ansprob = {}
        for entidx, span_onehot_vecs in self.entidx2spanvecs.items():
            # Shape: (Qlen)
            entity_token_att = sum([(spanvec * qent_att) for spanvec in span_onehot_vecs])
            # Shape: Q_d
            qent_repr = (self.ques_encoded * entity_token_att.unsqueeze(1)).sum(0)
            # Shape: 2*Q_d
            qent_qstr_repr = torch.cat([qent_repr, qstr_repr])

            qent_qstr_repr = self._execution_parameters._dropout(qent_qstr_repr)
            contexts_vec = self._execution_parameters._dropout(self.contexts_vec)

            # Shape: C
            dot_prod = self._execution_parameters._bool_bilinear(qent_qstr_repr, contexts_vec)
            # dot_prod = (qent_qstr_repr.unsqueeze(0) * context_vecs).sum(1)

            probs = torch.sigmoid(dot_prod)

            boolean_prob = torch.max(probs)  # .unsqueeze(0)
            entidx2ansprob[entidx] = boolean_prob

        # Computing the expected boolean_answer based on the qent probs
        expected_prob = 0.0
        for entidx, ent_ans_prob in entidx2ansprob.items():
            # ent_prob = entidx2prob[entidx]
            ent_prob = entidx2score[entidx]
            expected_prob += ent_prob * ent_ans_prob

        return Bool1(value=expected_prob)

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

    language = HotpotQALanguageWSideArgs()
    print(language._functions)

    all_prods = language.all_possible_productions()

    print("All prods:\n{}\n".format(all_prods))




