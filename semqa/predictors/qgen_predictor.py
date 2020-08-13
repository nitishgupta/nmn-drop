import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('question_generation')
class QuestionGenerationPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context = json_dict['context']

        sent_start = json_dict['sent_start']
        sent_end = json_dict['sent_end']
        sentence = context[sent_start:sent_end]

        answer_start = json_dict['answer_start'] - sent_start
        answer_end = json_dict['answer_end'] - sent_start

        metadata = {}
        return self._dataset_reader.text_to_instance(context=sentence,
                                                     start=answer_start,
                                                     end=answer_end,
                                                     metadata=metadata)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        metadata = outputs['metadata']
        outputs.update(metadata)

        outputs['predicted_question'] = outputs['predicted_question']

        return json.dumps(outputs) + '\n'
