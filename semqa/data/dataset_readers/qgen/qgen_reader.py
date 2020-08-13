import json
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides
from typing import Any, Dict, Iterable, Optional

from semqa.models.qgen.constants import SPAN_START_TOKEN, SPAN_END_TOKEN


@DatasetReader.register('question_generation')
class QuestionGenerationDatasetReader(DatasetReader):
    def __init__(self,
                 model_name: str,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.tokenizer = PretrainedTransformerTokenizer(model_name)
        self.token_indexers = {'tokens': PretrainedTransformerIndexer(model_name, namespace='tokens')}

        # Add the tokens which will mark the answer span
        self.tokenizer.tokenizer.add_tokens([SPAN_START_TOKEN, SPAN_END_TOKEN])

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        num_read = 0
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                context = data['context']
                start = data['answer_start']
                end = data['answer_end']
                question = data.pop('question', None)
                metadata = data.pop('metadata', {})

                num_read += 1
                # if num_read > 100:
                #     break

                yield self.text_to_instance(context, start, end, question, metadata)

    def _insert_span_symbols(self, context: str, start: int, end: int) -> str:
        return f'{context[:start]}{SPAN_START_TOKEN} {context[start:end]} {SPAN_END_TOKEN}{context[end:]}'

    @overrides
    def text_to_instance(self,
                         context: str,
                         start: int,
                         end: int,
                         question: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Instance:
        fields = {}
        metadata = metadata or {}

        answer = context[start:end]
        marked_context = self._insert_span_symbols(context, start, end)
        source_tokens = self.tokenizer.tokenize(marked_context)
        fields['source_tokens'] = TextField(source_tokens, self.token_indexers)
        metadata['answer'] = answer
        metadata['answer_start'] = start
        metadata['answer_end'] = end
        metadata['context'] = context
        metadata['marked_context'] = marked_context
        metadata['source_tokens'] = source_tokens

        if question is not None:
            target_tokens = self.tokenizer.tokenize(question)
            fields['target_tokens'] = TextField(target_tokens, self.token_indexers)
            metadata['question'] = question
            metadata['target_tokens'] = target_tokens

        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)