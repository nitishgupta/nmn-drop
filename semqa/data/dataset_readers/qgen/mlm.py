import json
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides
from typing import Any, Dict, Iterable, Optional

SPAN_START_TOKEN = '<m>'
SPAN_END_TOKEN = '</m>'


@DatasetReader.register('bart_mlm')
class BartMLMReader(DatasetReader):
    def __init__(self,
                 model_name: str,
                 lazy: bool = False):
        super().__init__(lazy=lazy)
        self.tokenizer = PretrainedTransformerTokenizer(model_name)
        self.token_indexers = {'tokens': PretrainedTransformerIndexer(model_name, namespace='tokens')}

        self.mask_token = self.tokenizer.tokenizer.mask_token

        # # Add the tokens which will mark the answer span
        # self.tokenizer.tokenizer.add_tokens([SPAN_START_TOKEN, SPAN_END_TOKEN])

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                masked_line = line.strip()
                yield self.text_to_instance(masked_line)

    @overrides
    def text_to_instance(self,
                         masked_line: str) -> Instance:
        fields = {}
        metadata = {}

        source_tokens = self.tokenizer.tokenize(masked_line)
        fields['source_tokens'] = TextField(source_tokens, self.token_indexers)
        metadata['masked_line'] = masked_line
        metadata['source_tokens'] = source_tokens

        # fields['metadata'] = MetadataField(metadata)

        return Instance(fields)