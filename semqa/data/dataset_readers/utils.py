from typing import Dict, List, Tuple, Any
import string
import itertools
from collections import defaultdict
import numpy as np

from allennlp.data import Token
from allennlp.data.fields import ListField, SpanField, Field, LabelField, SequenceField, ArrayField
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.data.fields.labels_field import LabelsField

# Copied from allennlp_models/rc/dataset_readers/utils.py
# These are tokens and characters that are stripped by the standard SQuAD and TriviaQA evaluation
# scripts.
IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])

Span = Tuple[int, int]

def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def find_valid_spans(
        passage_tokens: List[Token], answer_texts: List[str]
) -> List[Tuple[int, int]]:
    """Find instances of answer_texts in the input tokens.

    Returns:
    --------
    spans: `List[Tuple[int, int]]`
        List of (start, end) spans where end is _inclusive_
    """
    normalized_tokens = [
        token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens
    ]
    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans


def extract_answer_info_from_annotation(
        answer_annotation: Dict[str, Any]
) -> Tuple[str, List[str]]:
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts: List[str] = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key]
            for key in ["month", "day", "year"]
            if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def get_single_answer_span_fields(
        passage_tokens: List[Token],
        max_passage_token_len: int,
        answer_annotation: Dict[str, Any],
        spacy_tokenizer: SpacyTokenizer,
        passage_field: SequenceField,
        p_tokenidx2wpidx: List[List[int]]=None
) -> Tuple[ListField, ArrayField, ListField, List[Tuple[int, int]], bool]:
    """ Get span-answer fields for a single-span prediction from a start/end predictor(tagger).

    For each answer-text, find all possible answer token-spans (convert to wp-spans if needed). This would be
    returned as a List[SpanField].

    For addition passage-attention loss, also return these spans as List[List[SpanField]] where
    - outer list denotes different possible answer-taggings (which is all possible spans in this case, len = num_spans)
    - inner list denotes all the spans in a given tagging (which is a single span in this case; len = 1)
    """

    passage_tokens = passage_tokens[:max_passage_token_len]
    answer_type, answer_texts = extract_answer_info_from_annotation(answer_annotation)
    tokenized_answer_texts = []
    for answer_text in answer_texts:
        answer_tokens = spacy_tokenizer.tokenize(answer_text)
        answer_tokens = split_tokens_by_hyphen(answer_tokens)
        tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))

    # These are token-idxs
    answer_spans: List[Tuple[int, int]] = find_valid_spans(passage_tokens,
                                                           answer_texts=tokenized_answer_texts)  # end is _inclusive_
    if len(answer_spans) == 0:
        answer_spans_field = ListField([SpanField(-1, -1, passage_field)])
        answer_spans_mask = ArrayField(np.array([0], dtype=np.int32), padding_value=0, dtype=np.int32)
        answer_spans_for_possible_taggings_field = ListField([ListField([SpanField(-1, -1, passage_field)])])
        return answer_spans_field, answer_spans_mask, answer_spans_for_possible_taggings_field, answer_spans, False

    if p_tokenidx2wpidx is not None:
        answer_spans: List[Tuple[int, int]] = convert_token_spans_to_wp_spans(answer_spans, p_tokenidx2wpidx)

    # ListField[SpanField]
    answer_spans_field = ListField([SpanField(x, y, passage_field) for (x, y) in answer_spans])
    answer_spans_mask = ArrayField(np.array([1]*len(answer_spans), dtype=np.int32), padding_value=0, dtype=np.int32)
    # ListField[ListField[SpanField]]
    answer_spans_for_possible_taggings_field = ListField([ListField([SpanField(x, y, passage_field)])
                                                          for (x, y) in answer_spans])

    return answer_spans_field, answer_spans_mask, answer_spans_for_possible_taggings_field, answer_spans, True


def index_text_to_tokens(text: str, tokens: List[Token]):
    text_index_to_token_index = []
    token_index = 0
    next_token_index = token_index + 1
    index = tokens[token_index].idx
    next_index = tokens[next_token_index].idx
    for i in range(len(text)):
        while True:
            while next_index == index and next_token_index < len(tokens) - 1:
                next_token_index += 1
                next_index = tokens[next_token_index].idx
            if next_index == index and next_token_index == len(tokens) - 1:
                next_token_index += 1
                next_index = len(text)

            if i >= index and i < next_index:
                text_index_to_token_index.append(token_index)
                break
            else:
                token_index = next_token_index
                index = next_index

                if next_token_index < len(tokens) - 1:
                    next_token_index += 1
                    next_index = tokens[next_token_index].idx
                else:
                    next_token_index += 1
                    next_index = len(text)
                if (next_token_index > len(tokens)):
                    raise Exception("Error in " + text)
    return text_index_to_token_index


def convert_token_spans_to_wp_spans(spans: List[Tuple[int, int]],
                                    tokenidx2wpidx: List[List[int]]) -> List[Tuple[int, int]]:
    """Convert span-token-idxs to wp-idxs.

    Note: Entries in tokenidx2wpidx can be empty (if the underlying token was " " for which wp=''). Take care of that.
    """
    if tokenidx2wpidx is None:
        return spans
    wp_spans = []
    for span in spans:
        token_start_idx, token_end_idx = span[0], span[1]
        if len(tokenidx2wpidx[token_start_idx]) == 0:   # if empty wp
            if token_start_idx == len(tokenidx2wpidx) - 1:   # if token is last
                token_start_idx -= 1
            else:
                token_start_idx += 1
        if len(tokenidx2wpidx[token_end_idx]) == 0:
            token_end_idx -= 1

        if token_end_idx < token_start_idx:
            token_end_idx = token_start_idx

        start = tokenidx2wpidx[token_start_idx][0]
        end = tokenidx2wpidx[token_end_idx][-1]  # span end is _inclusive_
        wp_spans.append((start, end))
    return wp_spans


class BIOAnswerGenerator:
    def __init__(self,
                 ignore_question: bool = True,
                 flexibility_threshold: int = 1000,
                 labels: Dict[str, int] = {
                     'O': 0,
                     'B': 1,
                     'I': 2}):
        self._ignore_question = ignore_question
        self._flexibility_threshold = flexibility_threshold
        self._labels = labels
        self._labels_scheme = ''.join(sorted(labels.keys()))
        if self._labels_scheme == 'BILOU':
            self._labels_scheme = 'BIOUL'
        self.spacy_tokenizer = SpacyTokenizer()

    def _create_sequence_labels(self, spans: List[Tuple[int, int]], seq_len: int) -> List[int]:
        labels = self._labels
        labels_scheme = self._labels_scheme
        # initialize all labels to O
        labeling = [labels['O']] * seq_len

        for start, end in spans:
            if labels_scheme == 'BIO':
                # create B labels
                labeling[start] = labels['B']
                # create I labels
                labeling[start + 1: end + 1] = [labels['I']] * (end - start)
            elif labels_scheme == 'IO':
                # create I labels
                labeling[start: end + 1] = [labels['I']] * (end - start + 1)
            elif labels_scheme == 'BIOUL':
                if end - start == 0:
                    labeling[start] = labels['U']
                else:
                    labeling[start] = labels['B']
                    labeling[start + 1: end] = [labels['I']] * (end - start - 1)
                    labeling[end] = labels['L']
            else:
                raise Exception("Illegal labeling scheme")

        return labeling


    def get_bio_labels(self,
                       answer_annotation: Dict[str, Any],
                       passage_tokens: List[Token],
                       max_passage_len: int,
                       passage_field: SequenceField,
                       p_tokenidx2wpidx: List[List[int]] = None,
                       passage_wps_len: int = None):
        """

        Most of this is based on
        `https://github.com/eladsegal/tag-based-multi-span-extraction/blob/
         0911e72dfc9f4473f46b154f9090de0a1a2b943b/
         src/data/dataset_readers/answer_field_generators/tagged_answer_generator.py#L30`
        """
        # First find answer-spans based passage tokens
        passage_tokens = passage_tokens[:max_passage_len]
        answer_type, answer_texts = extract_answer_info_from_annotation(answer_annotation)
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self.spacy_tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(" ".join(token.text for token in answer_tokens))

        all_spans = []      # Based on wp-idx if provided, else token-idxs
        spans_dict = {}
        for i, answer_text in enumerate(tokenized_answer_texts):
            answer_spans: List[Tuple[int, int]] = find_valid_spans(passage_tokens,
                                                                   answer_texts=[answer_text])   # end is _inclusive_
            if len(answer_spans) == 0:
                continue
            if p_tokenidx2wpidx is not None:
                answer_spans = convert_token_spans_to_wp_spans(answer_spans, p_tokenidx2wpidx)
            spans_dict[answer_text] = answer_spans
            all_spans.extend(answer_spans)

        passage_len = len(passage_tokens) if passage_wps_len is None else passage_wps_len

        # Only gets instantiated in one case -- find below
        packed_gold_spans_list: List[Tuple[List[Span]]] = None

        if len(all_spans) > 0:
            has_answer = True
            # Create BIO label-sequences
            flexibility_count = 1
            for answer_text in tokenized_answer_texts:
                spans = spans_dict[answer_text] if answer_text in spans_dict else []
                if len(spans) == 0:
                    continue
                flexibility_count *= ((2 ** len(spans)) - 1)    # 2^n_spans - 1 = all non-empty combinations of this ans

            if (flexibility_count < self._flexibility_threshold):
                # generate all non-empty span combinations for each answer_text
                spans_combinations_dict = {}    # answer_text: List of (list of spans)
                for answer_text, spans in spans_dict.items():
                    # power-set (all possible subsets of spans) for this answer_text
                    spans_combinations_dict[answer_text] = all_combinations = []
                    for i in range(1, len(spans) + 1):
                        # all combination sets of spans of size = i
                        all_combinations += list(itertools.combinations(spans, i))

                # calculate cartesian product between all span-combinations for each answer-text
                # Each element in this list is a possible combination of spans from different answer-texts, i.e.
                # each tagging is a Tuple[List[Span]], where size of this tuple is len(answer-texts) and each List[Span]
                # is a combination of spans for that answer-text.
                packed_gold_spans_list: List[Tuple[List[Span]]] = \
                    itertools.product(*list(spans_combinations_dict.values()))
                bios_list: List[LabelsField] = []
                answer_spans_for_possible_taggings = []     # List[ListField[SpanField]]
                for packed_gold_spans in packed_gold_spans_list:
                    gold_spans: List[Span] = [s for sublist in packed_gold_spans for s in sublist]
                    answer_spans_for_possible_taggings.append(
                        ListField([SpanField(x, y, passage_field) for (x, y) in gold_spans]))
                    bios = self._create_sequence_labels(gold_spans, passage_len)
                    bios_list.append(LabelsField(bios))

                # ListField[LabelsField]
                answer_spans_as_bios_field = ListField(bios_list)
                bios_mask = [1] * len(bios_list)
                bios_field_mask = ArrayField(np.array(bios_mask, dtype=np.int32), padding_value=0, dtype=np.int32)
                # ListField[ListField[SpanField]]
                answer_spans_for_possible_taggings_field = ListField(answer_spans_for_possible_taggings)

                # recomputing since generator gets used up above
                packed_gold_spans_list: List[Tuple[List[Span]]] = \
                    itertools.product(*list(spans_combinations_dict.values()))

                # fields['answer_as_list_of_bios'] = ListField(bios_list)
                # fields['answer_as_text_to_disjoint_bios'] = ListField([ListField([no_answer_bios])])
            else:
                # Create a single tagging with all spans
                all_spans_bio_labels = LabelsField(self._create_sequence_labels(all_spans, passage_len))
                answer_spans_as_bios_field = ListField([all_spans_bio_labels])   # single tagging
                bios_mask = [1]
                bios_field_mask = ArrayField(np.array(bios_mask, dtype=np.int32), padding_value=0, dtype=np.int32)
                answer_spans_for_possible_taggings = [
                    ListField([SpanField(x, y, passage_field) for (x, y) in all_spans])]
                answer_spans_for_possible_taggings_field = ListField(answer_spans_for_possible_taggings)
                # fields['answer_as_list_of_bios'] = ListField([no_answer_bios])

            # bio_labels: List[int] = self._create_sequence_labels(all_spans, passage_len)
            # fields['span_bio_labels'] = LabelsField(bio_labels)
            # fields['is_bio_mask'] = LabelField(1, skip_indexing=True)
        else:
            answer_spans_as_bios_field = ListField([self._get_empty_answer(passage_len)])
            bios_mask = [0]
            bios_field_mask = ArrayField(np.array(bios_mask, dtype=np.int32), padding_value=0, dtype=np.int32)
            answer_spans_for_possible_taggings_field = ListField([ListField([SpanField(-1, -1, passage_field)])])
            has_answer = False

        return (answer_spans_as_bios_field, bios_field_mask, answer_spans_for_possible_taggings_field, all_spans,
                packed_gold_spans_list, spans_dict, has_answer)
        # return fields, has_answer

    @staticmethod
    def _get_empty_answer(seq_len: int) -> LabelsField:
        return LabelsField([0] * seq_len)

    def get_empty_answer_fields(self, passage_len: int) -> Dict[str, Field]:
        fields: Dict[str, Field] = {}

        no_answer_bios = self._get_empty_answer(passage_len)

        fields['answer_as_text_to_disjoint_bios'] = ListField([ListField([no_answer_bios])])
        fields['answer_as_list_of_bios'] = ListField([no_answer_bios])
        fields['span_bio_labels'] = no_answer_bios
        fields['is_bio_mask'] = LabelField(0, skip_indexing=True)

        return fields






