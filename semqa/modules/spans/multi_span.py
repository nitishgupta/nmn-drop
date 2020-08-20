from typing import List, Dict, Tuple

import warnings
import torch
from itertools import product
from collections import OrderedDict


from allennlp.nn.util import replace_masked_values, logsumexp

from semqa.modules.spans.span_answer import SpanAnswer
from semqa.modules.viterbi_decoding import allowed_transitions, viterbi_tags

class MultiSpanAnswer(SpanAnswer):
    def __init__(self,
                 ignore_question: bool,
                 prediction_method: str,
                 decoding_style: str,
                 training_style: str,
                 labels: Dict[str, int],
                 empty_decoding: bool = False,
                 generation_top_k: int = 0,
                 unique_decoding: bool = True,
                 substring_unique_decoding: bool = True) -> None:
        super().__init__()
        self._ignore_question = ignore_question
        self._generation_top_k = generation_top_k
        self._unique_decoding = unique_decoding
        self._substring_unique_decoding = substring_unique_decoding
        self._prediction_method = prediction_method     # "argmax", "viterbi"
        self.empty_decoding = empty_decoding   # if False, do not allow empty-span answer, i.e. predict atleast one span
        self._decoding_style = decoding_style
        self._training_style = training_style
        self._labels = labels

        assert (labels['O'] == 0)  # must have O as 0 as there are assumptions about it
        self._labels_scheme = ''.join(sorted(labels.keys()))
        if self._labels_scheme == 'BILOU':
            self._labels_scheme = 'BIOUL'
            self._span_start_label = self._labels['U']
        else:
            self._span_start_label = self._labels['B'] if self._labels_scheme == 'BIO' else self._labels['I']

        if self._prediction_method == 'viterbi':
            num_tags = len(labels)
            # (num_tags, num_tags) - viterbi_tags handles the START and END
            self._transitions = torch.ones(num_tags, num_tags)
            if self._labels_scheme == 'BIO' or self._labels_scheme == 'BIOUL':
                constraints = allowed_transitions(self._labels_scheme, {value: key for key, value in labels.items()})
            else:
                # For IO scheme, All transitions are allowed between tags.
                constraints = list(product(range(num_tags), range(num_tags)))
                constraints += [(num_tags, i) for i in range(num_tags)]       # START can transition to {I,O}
                constraints += [(i, num_tags + 1) for i in range(num_tags)]   # {I,O} can transition to END

            self._constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)
            for i, j in constraints:
                self._constraint_mask[i, j] = 1.0

    def gold_log_marginal_likelihood(self,
                                     passage_span_answer: torch.LongTensor,
                                     passage_span_answer_mask: torch.LongTensor,
                                     log_probs: torch.FloatTensor,
                                     passage_mask: torch.FloatTensor):
        """ Compute log-marginal likelihood for the gold-seqs

        Parameters:
        ----------
        passage_span_answer: `torch.LongTensor` (batch_size, num_taggings, passage_len)
        # answer_as_list_of_bios: `(num_gold_seqs, passage_length)`
        #     Various gold tag-seqs per instance.
        #     This can be completely masked (all 0s), in that case use span_bio_labels.
        # span_bio_labels: `(passage_length, )`
        #     Since BIO gold-tagging for the instance.
        #     If answer_as_list_of_bios is masked, use this. Otherwise, this would be masked (all 0s)
        log_probs: `(passage_length, num_tags)`
            Log-probabilities for the tags (maybe unmasked)
        passage_mask: `(passage_length, )`
            Float mask for the passage
        """
        # if answer_as_list_of_bios.sum() > 0:
        #     gold_bio_seqs = answer_as_list_of_bios
        # elif span_bio_labels.sum() > 0:
        #     gold_bio_seqs = span_bio_labels.unsqueeze(0)
        # else:
        #     #  If log-loss is being computed during validation for an instance with no gold-spans
        #     gold_bio_seqs = span_bio_labels.unsqueeze(0)
        #     warnings.warn('One of answer_as_list_of_bios or span_bio_labels need to be un-masked')

        gold_bio_seqs = passage_span_answer
        gold_bio_seqs_mask = passage_span_answer_mask
        if gold_bio_seqs.sum() == 0:
            #  If log-loss is being computed during validation for an instance with no gold-spans
            warnings.warn('One of answer_as_list_of_bios or span_bio_labels need to be un-masked. M:{}'.format(
                gold_bio_seqs_mask.sum()
            ))


        log_marginal_likelihood = self._marginal_likelihood(gold_bio_seqs=gold_bio_seqs,
                                                            gold_bio_seqs_mask=gold_bio_seqs_mask,
                                                            log_probs=log_probs,
                                                            seq_mask=passage_mask)

        return log_marginal_likelihood


    def _marginal_likelihood(self,
                             gold_bio_seqs: torch.LongTensor,
                             gold_bio_seqs_mask: torch.LongTensor,
                             log_probs: torch.FloatTensor,
                             seq_mask: torch.FloatTensor):
        """ Compute marginal log-likelihood for the gold-seqs given the log-probabilities for tags

        Parameters:
        -----------
        gold_bio_seqs: `(num_gold_seqs, seq_length)`
            Multiple gold BIO-tag gold seqs for a single instance.
            /// Some of the sequences can be empty (all 0s) and need to be masked ///
        gold_bio_seqs_mask: `(num_gold_seqs, )`
            0/1 mask to indicate if gold-seq is masked or not. An empty (all 0s) seq may still be unmasked and may be
            used to supervise a NO-spans tagging
        log_probs: `(seq_length, num_tags)
            Need to be masked with (-1e7)
        seq_mask: `(seq_length)`
        """

        # Replacing masked log-probs with 0; Now seq-log-likelihood can be computed by summing over the seq
        log_probs = log_probs * seq_mask.unsqueeze(-1)

        # Shape: (num_gold_seqs, seq_length, num_tags)
        expanded_log_probs = log_probs.unsqueeze(0).expand(gold_bio_seqs.size()[0], -1, -1)

        # Shape: (num_gold_seqs, seq_length)
        log_likelihoods = torch.gather(expanded_log_probs, dim=-1, index=gold_bio_seqs.unsqueeze(-1)).squeeze(-1)

        # # Shape: (num_gold_seqs)
        # correct_sequences_pad_mask = (gold_bio_seqs.sum(-1) > 0)

        # Shape: (num_gold_seqs)
        seqs_log_likelihood = log_likelihoods.sum(dim=-1)
        seqs_log_likelihood = replace_masked_values(seqs_log_likelihood, gold_bio_seqs_mask.bool(), -1e32)

        # This would compute log \sum \exp(seq-log-prob) = \log\sum(seq-prob)
        log_marginal_likelihood = logsumexp(seqs_log_likelihood)

        # If all gold-seqs are masked, then mask the log_marginal_likelihood
        combined_gold_seqs_mask = (gold_bio_seqs_mask.sum() > 0).float()
        log_marginal_likelihood = log_marginal_likelihood * combined_gold_seqs_mask

        return log_marginal_likelihood

    def get_predicted_tags(self,
                           log_probs: torch.LongTensor,
                           passage_mask: torch.FloatTensor):

        # prediction_method = 'argmax' if self.training else self._prediction_method
        prediction_method = 'viterbi'  # if self.training else self._prediction_method

        # Shape: (non-masked_passage_length)
        masked_indices = passage_mask.nonzero().squeeze()
        masked_log_probs = log_probs[masked_indices]

        if prediction_method == 'viterbi':
            # Shape should be (1, 2, non-masked_passage_length) -- 1 for batch_size, top_k=2
            top_two_masked_predicted_tags = torch.Tensor(viterbi_tags(logits=masked_log_probs.unsqueeze(0),
                                                                      transitions=self._transitions,
                                                                      constraint_mask=self._constraint_mask, top_k=2))
            masked_predicted_tags = top_two_masked_predicted_tags[0, 0, :]
            # Empty output is allowed now; given that we're pretraining with zero-count
            if masked_predicted_tags.sum(dim=-1) == 0 and not self.empty_decoding:
                masked_predicted_tags = top_two_masked_predicted_tags[0, 1, :]
        elif prediction_method == 'argmax':
            masked_predicted_tags = torch.argmax(masked_log_probs, dim=-1)
        else:
            raise Exception("Illegal prediction_method")

        return masked_predicted_tags

    def decode_answer(self,
                      log_probs: torch.LongTensor,
                      passage_mask: torch.FloatTensor,
                      p_text: str,
                      passage_token_charidxs: List[int],
                      passage_tokens: List[str],
                      p_tokenidx2wpidx: List[List[int]] = None,
                      original_question: str = None) -> List[str]:

        """ Decode answer from tag-log-probs

        Parameters:
        -----------
        log_probs: `(sequence_length, num_tags)`
        passage_mask: `(seq_length)`


        """
        # Sequence containing indices for BIO tags
        masked_predicted_tags: torch.LongTensor = self.get_predicted_tags(log_probs=log_probs,
                                                                          passage_mask=passage_mask)

        if p_tokenidx2wpidx is not None:
            predicted_token_tags = masked_predicted_tags.cpu().tolist()
            predicted_token_tags = self.convert_wp_tags_to_token_tags(predicted_token_tags, p_tokenidx2wpidx)
        else:
            predicted_token_tags = masked_predicted_tags.cpu().tolist()

        # Convert BIO tag indices to spans
        predicted_span_idxs: List[Tuple[int, int]] = self.convert_tags_to_spans(predicted_token_tags)
        predicted_answer_spans: List[str] = []
        for span in predicted_span_idxs:
            start_char_idx = passage_token_charidxs[span[0]]    # start-char of start-token-idx
            end_char_idx = passage_token_charidxs[span[1]] + len(passage_tokens[span[1]])   # This is _exclusive_
            predicted_answer_spans.append(p_text[start_char_idx:end_char_idx])

        if not predicted_answer_spans:
            predicted_answer_spans = [""]

        if self._unique_decoding:
            predicted_answer_spans = list(OrderedDict.fromkeys(predicted_answer_spans))

        # retrying = False
        # while True:
        #     spans_text, spans_indices = self._decode_spans_from_tags(masked_predicted_tags, masked_qp_indices,
        #                                                              qp_tokens, question_passage_wordpieces,
        #                                                              p_text, q_text)
        #     if (not retrying and len(spans_text) == 0 and prediction_method == 'argmax'):
        #         retrying = True
        #         max_start_index = torch.argmax(masked_log_probs[:, self._span_start_label], dim=0)
        #         masked_predicted_tags[max_start_index] = self._span_start_label
        #     else:
        #         break
        #
        # if self._unique_decoding:
        #     spans_text = list(OrderedDict.fromkeys(spans_text))
        #
        #     if self._substring_unique_decoding:
        #         spans_text = self._remove_substring_from_decoded_output(spans_text)
        #
        # answer_dict = {
        #     'value': spans_text,
        #     'spans': spans_indices
        # }
        # return answer_dict

        return predicted_answer_spans


    def convert_wp_tags_to_token_tags(self, predicted_tags: List[int], p_tokenidx2wpidx: List[List[int]]):
        """Convert tags over wordpieces to tags over tokens.

        Since a single token's wps could be assigned different tags, we hard-code some logic into this transformation.
        Along the possible tags for a token, we have a preference for the assigned tag depending on the previous tag.
        Constraints:
        O: [B, O]       - if previous token is O, prefer B, then O
                          (if current token is only allowed I, it's an error in BIO labeling scheme; resort to O)
        I: [I, B, O]    - if previous token is I, prefer I (continue span), B (start new), then O
        B: [I, B, O]    - if previous token is B, prefer I (continue span), B (start new), then O
        """
        labels = self._labels
        if self._labels_scheme == "BIO":
            tag_preference = {
                'O': ['B', 'O'],
                'I': ['I', 'B', 'O'],
                'B': ['I', 'B', 'O'],
            }
        elif self._labels_scheme == "IO":
            tag_preference = {
                'O': ['I', 'O'],
                'I': ['I', 'O'],
            }
        else:
            raise Exception("Labeling scheme not supported: {}".format(self._labels_scheme))

        tag_preference = {labels[k]: [labels[x] for x in v] for (k, v) in tag_preference.items()}

        passage_wp_len = len(predicted_tags)
        passage_token_len = len(p_tokenidx2wpidx)
        token_tags = [labels['O']] * passage_token_len
        prev_tag = labels['O']

        for token_idx in range(passage_token_len):
            wp_idxs = p_tokenidx2wpidx[token_idx]
            if not wp_idxs:     # wps can be empty for a token, eg. " "
                prev_tag = labels['O']
                continue
            wp_idxs = [x for x in wp_idxs if 0 <= x < passage_wp_len]
            possible_tags = [predicted_tags[x] for x in wp_idxs]    # tags assigned to this token
            tag_preference_order = tag_preference[prev_tag]         # preference order of tags based on previous tag
            assigned_tag = None
            for tag in tag_preference_order:
                if tag in possible_tags:   # if a possible tag is in preference list
                    assigned_tag = tag
                    break
            if assigned_tag is None:       # if none of preference was in possible tags, assign 'O'
                assigned_tag = labels['O']

            token_tags[token_idx] = assigned_tag
            prev_tag = assigned_tag

        return token_tags


    def convert_tags_to_spans(self, token_tags: List[int]) -> List[Tuple[int, int]]:
        labels = self._labels
        idx2label = {idx: label for label, idx in labels.items()}
        token_tags: List[str] = [idx2label[idx] for idx in token_tags]

        labels_scheme = self._labels_scheme

        spans = []
        prev = 'O'
        current = []
        for tokenidx, tag in enumerate(token_tags):
            if tag == "I":
                if labels_scheme == "BIO":
                    if prev == "B" or prev == "I":
                        current.append(tokenidx)
                        prev = "I"
                    else:
                        # Illegal I, treat it as O
                        prev = "O"
                elif labels_scheme == "IO":
                    if prev == "I":
                        current.append(tokenidx)    # continue span
                    else:
                        if current:
                            spans.append((current[0], current[-1]))
                        current = [tokenidx]
                        prev = "I"
                else:
                    raise NotImplementedError

            if tag == "O":
                if prev == "O":
                    continue
                elif prev == "B" or prev == "I":
                    if current:
                        spans.append((current[0], current[-1]))
                    current = []
                    prev = "O"
            if tag == "B":
                if current:
                    spans.append((current[0], current[-1]))
                current = [tokenidx]
                prev = "B"

        if current:
            # residual span
            spans.append((current[0], current[-1]))

        return spans











    # def _decode_spans_from_tags(self, masked_tags, masked_qp_indices,
    #                             qp_tokens, qp_wordpieces,
    #                             passage_text, question_text):
    #     """
    #     decoding_style: str - The options are:
    #                 "single_word_representation" - Each word's wordpieces are aggregated somehow.
    #                                                 The only decoding_style that requires masking of wordpieces.
    #                 "at_least_one" - If at least one of the wordpieces is tagged with B,
    #                                 then the whole word is taken. This is approach yielding the best results
    #                                 for the non-masked wordpieces models.
    #                 "forget_wordpieces" - Each wordpiece is regarded as an independent token,
    #                                 which means partial words predictions are valid. This is the most natural decoding.
    #                 "strict_wordpieces" - all of the wordpieces should be tagged as they would have been in the reader
    #     """
    #     decoding_style = self._decoding_style
    #     labels = self._labels
    #     labels_scheme = self._labels_scheme
    #
    #     ingested_token_indices = []
    #     spans_tokens = []
    #     prev = labels['O']
    #     current_tokens = []
    #
    #     context = ''
    #     for i in range(len(masked_qp_indices)):
    #         tag = masked_tags[i]
    #         token_index = masked_qp_indices[i]
    #         token = qp_tokens[token_index]
    #         relevant_wordpieces = qp_wordpieces[token_index]
    #
    #         if token_index in ingested_token_indices:
    #             continue
    #
    #         if decoding_style == 'single_word_representation' or decoding_style == 'at_least_one':
    #             tokens = [qp_tokens[j] for j in qp_wordpieces[token_index]]
    #             token_indices = qp_wordpieces[token_index]
    #         elif decoding_style == 'forget_wordpieces':
    #             tokens = [qp_tokens[token_index]]
    #             token_indices = [token_index]
    #         elif decoding_style == 'strict_wordpieces':
    #             num_of_prev_wordpieces = len(relevant_wordpieces) - 1
    #             if len(relevant_wordpieces) == 1:
    #                 tokens = [qp_tokens[token_index]]
    #             elif (token_index == relevant_wordpieces[-1] and  # if token is the last wordpiece
    #                   len(ingested_token_indices) >= len(
    #                         relevant_wordpieces) - 1  # and the number of ingested is at least the number of previous wordpieces
    #                   and ingested_token_indices[-num_of_prev_wordpieces:] == relevant_wordpieces[
    #                                                                           :-1]):  # and all the last ingested are exactly the previous wordpieces
    #                 tokens = [qp_tokens[j] for j in qp_wordpieces[token_index]]
    #             else:
    #                 tokens = []
    #             token_indices = [token_index]
    #         else:
    #             raise Exception("Illegal decoding_style")
    #
    #         add_span = False
    #         ingest_token = False
    #
    #         if labels_scheme == 'BIO' or labels_scheme == 'IO':
    #             if tag == labels['I']:
    #                 if prev != labels['O'] or labels_scheme == 'IO':
    #                     ingest_token = True
    #                     prev = labels['I']
    #                 else:
    #                     # Illegal I, treat it as O
    #                     # Won't occur with Viterbi or constrained beam search, only with argmax
    #                     prev = labels['O']
    #
    #             elif labels_scheme != 'IO' and tag == labels['B']:
    #                 add_span = True
    #                 ingest_token = True
    #                 prev = labels['B']
    #
    #                 if decoding_style == 'strict_wordpieces':
    #                     if token_index != relevant_wordpieces[0]:
    #                         ingest_token = False
    #                         prev = labels['O']
    #
    #             elif tag == labels['O'] and prev != labels['O']:
    #                 # Examples: "B O" or "B I O"
    #                 # consume previously accumulated tokens as a span
    #                 add_span = True
    #                 prev = labels['O']
    #         # elif labels_scheme == 'BIOUL':
    #         #     if tag == labels['I']:
    #         #         if prev == labels['B'] or prev == labels['I']:
    #         #             ingest_token = True
    #         #             prev = labels['I']
    #         #         else:
    #         #             # Illegal I, treat it as O
    #         #             # Won't occur with Viterbi or constrained beam search, only with argmax
    #         #             prev = labels['O']
    #         #
    #         #     elif tag == labels['B']:
    #         #         if prev == labels['O'] or prev == labels['L'] or prev == labels['U']:
    #         #             ingest_token = True
    #         #             prev = labels['B']
    #         #         else:
    #         #             prev = labels['O']
    #         #
    #         #         if decoding_style == 'strict_wordpieces':
    #         #             if token_index != relevant_wordpieces[0]:
    #         #                 ingest_token = False
    #         #                 prev = labels['O']
    #         #
    #         #     elif tag == labels['U']:
    #         #         if prev == labels['O'] or prev == labels['L'] or prev == labels['U']:
    #         #             add_span = True
    #         #             ingest_token = True
    #         #             prev = labels['U']
    #         #         else:
    #         #             prev = labels['O']
    #         #
    #         #         if decoding_style == 'strict_wordpieces':
    #         #             if token_index != relevant_wordpieces[0]:
    #         #                 ingest_token = False
    #         #                 prev = labels['O']
    #         #
    #         #     elif tag == labels['L']:
    #         #         if prev == labels['I'] or prev == labels['B']:
    #         #             add_span = True
    #         #             ingest_token = True
    #         #             prev = labels['L']
    #         #         else:
    #         #             prev = labels['O']
    #         #
    #         #         if decoding_style == 'strict_wordpieces':
    #         #             if token_index != relevant_wordpieces[-1]:
    #         #                 ingest_token = False
    #         #                 prev = labels['O']
    #         #
    #         #     elif tag == labels['O']:
    #         #         prev = labels['O']
    #
    #         else:
    #             raise Exception("Illegal labeling scheme")
    #
    #         if labels_scheme == 'BIOUL' and ingest_token:
    #             current_tokens.extend(tokens)
    #             ingested_token_indices.extend(token_indices)
    #             context = get_token_context(token)
    #         if add_span and current_tokens:
    #             context = get_token_context(current_tokens[0])
    #             # consume previously accumulated tokens as a span
    #             spans_tokens.append((context, current_tokens))
    #             # start accumulating for a new span
    #             current_tokens = []
    #         if labels_scheme != 'BIOUL' and ingest_token:
    #             current_tokens.extend(tokens)
    #             ingested_token_indices.extend(token_indices)
    #
    #     if current_tokens:
    #         # Examples: # "B [EOS]", "B I [EOS]"
    #         context = get_token_context(current_tokens[0])
    #         spans_tokens.append((context, current_tokens))
    #
    #     spans_text, spans_indices = decode_token_spans(spans_tokens, passage_text, question_text)
    #
    #     return spans_text, spans_indices

    @staticmethod
    def _remove_substring_from_decoded_output(spans):
        new_spans = []
        lspans = [s.lower() for s in spans]

        for span in spans:
            lspan = span.lower()

            # remove duplicates due to casing
            if lspans.count(lspan) > 1:
                lspans.remove(lspan)
                continue

            # remove some kinds of substrings
            if not any((lspan + ' ' in s or ' ' + lspan in s or lspan + 's' in s or lspan + 'n' in s or (
                    lspan in s and not s.startswith(lspan) and not s.endswith(lspan))) and lspan != s for s in lspans):
                new_spans.append(span)

        return new_spans

