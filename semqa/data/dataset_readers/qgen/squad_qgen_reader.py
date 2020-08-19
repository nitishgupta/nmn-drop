from typing import Any, Dict, Iterable, Optional, List
import json
import logging
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides

from datasets.drop import constants
from semqa.utils.qdmr_utils import Node, node_from_dict, nested_expression_to_lisp, \
    get_domainlang_function2returntype_mapping, get_inorder_function_list, function_to_action_string_alignment
from semqa.models.qgen.constants import SPAN_START_TOKEN, SPAN_END_TOKEN, ALL_SPECIAL_TOKENS
from utils.util import _KnuthMorrisPratt

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('squad_conditional_qgen')
class SquadConditionalQuestionGenerationReader(DatasetReader):
    def __init__(self,
                 model_name: str,
                 add_masked_question: bool = True,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        # Setting this to false to encode pairs of text
        self.tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)
        self.token_indexers = {'tokens': PretrainedTransformerIndexer(model_name, namespace='tokens')}

        # Add the tokens which will mark the answer span
        self.tokenizer.tokenizer.add_tokens(ALL_SPECIAL_TOKENS)
        self.mask_token = self.tokenizer.tokenizer.mask_token

        self.add_masked_question = add_masked_question

        self.total_instances = 0

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        num_read = 0

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")

        total_ques, instances_read = 0, 0

        for passage_id, passage_info in dataset.items():
            passage = passage_info[constants.passage]

            for qa in passage_info[constants.qa_pairs]:
                total_ques += 1
                question_id = qa[constants.query_id]
                question = qa[constants.question]
                question_tokens = qa[constants.question_tokens]
                question_charidxs = qa[constants.question_charidxs]

                answer_dict = qa["answer"]
                answer_text = answer_dict["spans"][0]   # SQuAD only has a single span answer
                if not answer_text:
                    continue

                answer_start_charoffsets = list(_KnuthMorrisPratt(passage, answer_text))
                if not answer_start_charoffsets:
                    continue

                if self.add_masked_question:
                    program_supervision: Dict = qa.get(constants.program_supervision, None)
                    if program_supervision is None:
                        continue

                    program_node = node_from_dict(program_supervision)
                    # Since this is a SQuAD program we know it is a project(select)
                    # (start, end) _inclusive_ token indices
                    project_start_end_indices = program_node.extras.get("project_start_end_indices", None)
                    select_start_end_indices = program_node.extras.get("select_start_end_indices", None)

                    if project_start_end_indices is None or select_start_end_indices is None:
                        continue

                    # Start char-index of the project's start token
                    mask_start_charoffset = question_charidxs[project_start_end_indices[0]]
                    mask_end_charoffset = question_charidxs[project_start_end_indices[1]] + \
                                            len(question_tokens[project_start_end_indices[1]]) - 1

                    masked_question = question[0:mask_start_charoffset] + self.mask_token + \
                                        question[mask_end_charoffset + 1:]
                else:
                    masked_question = None

                instances_read += 1
                # if instances_read > 100:
                #     break

                yield self.text_to_instance(passage=passage,
                                            masked_question=masked_question,
                                            answer_text=answer_text,
                                            answer_start_charoffsets=answer_start_charoffsets,
                                            passage_id=passage_id,
                                            query_id=question_id,
                                            question=question)

        logger.info("Total questions: {} Instances read: {}".format(total_ques, instances_read))

    def _insert_span_symbols(self, context: str, start: int, end: int) -> str:
        return f'{context[:start]}{SPAN_START_TOKEN} {context[start:end]} {SPAN_END_TOKEN}{context[end:]}'

    @overrides
    def text_to_instance(self,
                         passage: str,
                         answer_text: str,
                         answer_start_charoffsets: List[int],
                         masked_question: str = None,
                         passage_id: str = None,
                         query_id: str = None,
                         question: Optional[str] = None) -> Instance:
        fields = {}

        ans_start_offset, ans_end_offset = answer_start_charoffsets[0], answer_start_charoffsets[0] + len(answer_text)

        ans_marked_passage = self._insert_span_symbols(passage, ans_start_offset, ans_end_offset)
        ans_marked_passage_tokens: List[Token] = self.tokenizer.tokenize(ans_marked_passage)

        if masked_question is not None:
            # source: masked-question </s> paragraph
            masked_question_tokens: List[Token] = self.tokenizer.tokenize(masked_question)
            source_tokens = self.tokenizer.add_special_tokens(masked_question_tokens, ans_marked_passage_tokens)
        else:
            # source: paragraph
            source_tokens = self.tokenizer.add_special_tokens(ans_marked_passage_tokens)
        fields['source_tokens'] = TextField(source_tokens, self.token_indexers)

        metadata = {}
        metadata['answer'] = answer_text
        metadata['answer_start'] = ans_start_offset
        metadata['answer_end'] = ans_end_offset
        metadata['passage'] = passage
        metadata['ans_marked_passage'] = ans_marked_passage
        metadata['source_tokens'] = source_tokens

        if question is not None:
            target_tokens = self.tokenizer.tokenize(question)
            target_tokens = self.tokenizer.add_special_tokens(target_tokens)
            fields['target_tokens'] = TextField(target_tokens, self.token_indexers)
            metadata['question'] = question
            metadata['target_tokens'] = target_tokens

        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)