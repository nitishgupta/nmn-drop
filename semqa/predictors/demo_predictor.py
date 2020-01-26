from typing import List, Union, Dict

import json
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import utils.util as myutils

import unicodedata
from utils import util, spacyutils
from allennlp.data.tokenizers import Token
from datasets.drop.preprocess import ner_process

from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.tools.drop_eval import get_metrics as drop_em_and_f1, answer_json_to_strings


def f1metric(prediction: Union[str, List], ground_truths: List):  # type: ignore
    """
    Parameters
    ----------a
    prediction: ``Union[str, List]``
        The predicted answer from the model evaluated. This could be a string, or a list of string
        when multiple spans are predicted as answer.
    ground_truths: ``List``
        All the ground truth answer annotations.
    """
    # If you wanted to split this out by answer type, you could look at [1] here and group by
    # that, instead of only keeping [0].
    ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
    exact_match, f1_score = metric_max_over_ground_truths(drop_em_and_f1, prediction, ground_truth_answer_strings)

    return (exact_match, f1_score)


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


@Predictor.register("drop_demo_predictor")
class DROPDemoPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]

        spacy_nlp = spacyutils.getSpacyNLP()
        spacy_whitespacetokenizer = spacyutils.getWhiteTokenizerSpacyNLP()

        # From datasets.drop.preprocess.tokenize
        # Parse passage
        cleaned_passage_text = unicodedata.normalize("NFKD", passage_text)
        cleaned_passage_text = util.pruneMultipleSpaces(cleaned_passage_text)
        passage_spacydoc = spacyutils.getSpacyDoc(cleaned_passage_text, spacy_nlp)
        passage_tokens = [t for t in passage_spacydoc]
        passage_tokens: List[Token] = split_tokens_by_hyphen(passage_tokens)

        passage_token_charidxs = [token.idx for token in passage_tokens]
        passage_token_texts: List[str] = [t.text for t in passage_tokens]

        # Remaking the doc for running NER on new tokenization
        new_passage_doc = spacyutils.getSpacyDoc(" ".join(passage_token_texts), spacy_whitespacetokenizer)

        assert len(passage_tokens) == len(" ".join(passage_token_texts).split(" "))
        assert len(new_passage_doc) == len(passage_tokens)

        # List[Tuple[int, int]] -- start (inclusive) and end (exclusive) token idxs for sentence boundaries
        passage_sent_idxs = sorted([(sentence.start, sentence.end) for sentence in new_passage_doc.sents],
                                   key=lambda x: x[0])

        passage_ners = spacyutils.getNER(new_passage_doc)

        (p_parsed_dates, p_normalized_date_idxs,
         p_normalized_date_values, _) = ner_process.parseDateNERS(passage_ners, passage_token_texts)
        (p_parsed_nums, p_normalized_num_idxs,
         p_normalized_number_values, _) = ner_process.parseNumNERS(passage_ners, passage_token_texts)

        # Parse question
        question: str = question_text.strip()
        cleaned_question = unicodedata.normalize("NFKD", question)
        cleaned_question = util.pruneMultipleSpaces(cleaned_question)

        q_spacydoc = spacyutils.getSpacyDoc(cleaned_question, spacy_nlp)
        question_tokens = [t for t in q_spacydoc]
        question_tokens = split_tokens_by_hyphen(question_tokens)
        question_token_charidxs = [token.idx for token in question_tokens]
        question_token_texts = [t.text for t in question_tokens]

        # Remaking the doc for running NER on new tokenization
        new_question_doc = spacyutils.getSpacyDoc(" ".join(question_token_texts), spacy_whitespacetokenizer)
        assert len(new_question_doc) == len(question_tokens)

        q_ners = spacyutils.getNER(new_question_doc)
        (q_parsed_dates, q_normalized_date_idxs,
         q_normalized_date_values, q_num_date_entities) = ner_process.parseDateNERS(q_ners, question_token_texts)
        (q_parsed_nums, q_normalized_num_idxs,
         q_normalized_number_values, q_num_num_entities) = ner_process.parseNumNERS(q_ners, question_token_texts)

        return self._dataset_reader.text_to_instance(
            question_text=" ".join(question_token_texts),
            original_ques_text=question_text,
            question_charidxs=question_token_charidxs,
            passage_text=" ".join(passage_token_texts),
            original_passage_text=passage_text,
            passage_charidxs=passage_token_charidxs,
            p_sent_boundaries=passage_sent_idxs,
            p_date_mens=p_parsed_dates,
            p_date_entidxs=p_normalized_date_idxs,
            p_date_normvals=p_normalized_date_values,
            p_num_mens=p_parsed_nums,
            p_num_entidxs=p_normalized_num_idxs,
            p_num_normvals=p_normalized_number_values,
            qtype="UNK",
            program_supervised=False,
            qattn_supervised=False,
            execution_supervised=False,
            pattn_supervised=False,
            strongly_supervised=False,
            ques_attn_supervision=None,
            date_grounding_supervision=None,
            num_grounding_supervision=None,
            passage_attn_supervision=None,
            synthetic_numground_metadata=None,
            answer_passage_spans=None,
            answer_question_spans=None,
            question_id="demo_question",
            passage_id="demo_passage",
            answer_annotations=None,
            max_question_len=50,
        )

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = self.predict_instance(instance)
        metadata = outputs["metadata"]
        question = metadata["original_question"]
        passage = metadata["original_passage"]
        predicted_ans = outputs["predicted_answer"]

        output_dict = {
            "question": question,
            "passage": passage,
            "predicted_ans": predicted_ans,
            "answer": predicted_ans
        }
        return output_dict

    def _print_ExecutionValTree(self, exval_tree, depth=0):
        """
        exval_tree: [[root_func_name, value], [], [], []]
        """
        tabs = "\t" * depth
        func_name = str(exval_tree[0][0])
        debug_value = str(exval_tree[0][1])
        debug_value = debug_value.replace("\n", "\n" + tabs)
        outstr = f"{tabs}{func_name}  :\n {tabs}{debug_value}\n"
        if len(exval_tree) > 1:
            for child in exval_tree[1:]:
                outstr += self._print_ExecutionValTree(child, depth + 1)
        return outstr

    # @overrides
    # def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
    #     # Use json.dumps(outputs) + "\n" to dump a dictionary
    #
    #     out_str = ""
    #     metadata = outputs["metadata"]
    #     predicted_ans = outputs["predicted_answer"]
    #     module_debug_infos = outputs["modules_debug_infos"]
    #
    #     gold_passage_span_ans = metadata["answer_passage_spans"] if "answer_passage_spans" in metadata else []
    #     gold_question_span_ans = metadata["answer_question_spans"] if "answer_question_spans" in metadata else []
    #
    #     # instance_spans_for_all_progs = outputs['predicted_spans']
    #     # best_span = instance_spans_for_all_progs[0]
    #     question_id = metadata["question_id"]
    #     question = metadata["original_question"]
    #     passage = metadata["original_passage"]
    #     passage_tokens = metadata["passage_orig_tokens"]
    #     passage_wps = metadata["passage_tokens"]
    #     passage_wpidx2tokenidx = metadata["passage_wpidx2tokenidx"]
    #     answer_annotation_dicts = metadata["answer_annotations"]
    #     passage_date_values = metadata["passage_date_values"]
    #     passage_num_values = metadata["passage_number_values"]
    #     composed_numbers = metadata["composed_numbers"]
    #     passage_year_diffs = metadata["passage_year_diffs"]
    #     # passage_num_diffs = metadata['passagenum_diffs']
    #     (exact_match, f1_score) = f1metric(predicted_ans, answer_annotation_dicts)
    #
    #     out_str += "qid: {}".format(question_id) + "\n"
    #     out_str += question + "\n"
    #     out_str += passage + "\n"
    #
    #     out_str += f"GoldAnswer: {answer_annotation_dicts}" + "\n"
    #     out_str += f"GoldPassageSpans:{gold_passage_span_ans}  GoldQuesSpans:{gold_question_span_ans}\n"
    #     # out_str += f"GoldPassageSpans:{answer_as_passage_spans}" + '\n'
    #
    #     # out_str += f"PredPassageSpan: {best_span}" + '\n'
    #     out_str += f"PredictedAnswer: {predicted_ans}" + "\n"
    #     out_str += f"F1:{f1_score} EM:{exact_match}" + "\n"
    #     out_str += f"Top-Prog: {outputs['logical_forms'][0]}" + "\n"
    #     out_str += f"Top-Prog-Prob: {outputs['batch_actionseq_probs'][0]}" + "\n"
    #     out_str += f"Dates: {passage_date_values}" + "\n"
    #     out_str += f"PassageNums: {passage_num_values}" + "\n"
    #     out_str += f"ComposedNumbers: {composed_numbers}" + "\n"
    #     # out_str += f'PassageNumDiffs: {passage_num_diffs}' + '\n'
    #     out_str += f"YearDiffs: {passage_year_diffs}" + "\n"
    #
    #     logical_forms = outputs["logical_forms"]
    #     program_probs = outputs["batch_actionseq_probs"]
    #     execution_vals = outputs["execution_vals"]
    #     program_logprobs = outputs["batch_actionseq_logprobs"]
    #     all_predicted_answers = outputs["all_predicted_answers"]
    #     if "logical_forms":
    #         for lf, d, ex_vals, prog_logprob, prog_prob in zip(
    #             logical_forms, all_predicted_answers, execution_vals, program_logprobs, program_probs
    #         ):
    #             ex_vals = myutils.round_all(ex_vals, 1)
    #             # Stripping the trailing new line
    #             ex_vals_str = self._print_ExecutionValTree(ex_vals, 0).strip()
    #             out_str += f"LogicalForm: {lf}\n"
    #             out_str += f"Prog_LogProb: {prog_logprob}\n"
    #             out_str += f"Prog_Prob: {prog_prob}\n"
    #             out_str += f"Answer: {d}\n"
    #             out_str += f"ExecutionTree:\n{ex_vals_str}"
    #             out_str += f"\n"
    #             # NUM_PROGS_TO_PRINT -= 1
    #             # if NUM_PROGS_TO_PRINT == 0:
    #             #     break
    #
    #     # # This is the top scoring program
    #     # # List of dictionary where each dictionary contains a single module_name: pattn-value pair
    #     # module_debug_info: List[Dict] = module_debug_infos[0]
    #     # for module_dict in module_debug_info:
    #     #     module_name, pattn = list(module_dict.items())[0]
    #     #     print(module_name)
    #     #     print(f"{len(pattn)}  {len(passage_wpidx2tokenidx)}")
    #     #     assert len(pattn) == len(passage_wpidx2tokenidx)
    #     #
    #     # # print(module_debug_infos)
    #     # # print(passage_wpidx2tokenidx)
    #
    #     out_str += "--------------------------------------------------\n"
    #
    #     return out_str


