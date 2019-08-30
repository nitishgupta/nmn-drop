from typing import Dict, List, Union

from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.tools.drop_eval import (get_metrics as drop_em_and_f1,
                                      answer_json_to_strings)




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
    print(ground_truth_answer_strings)
    exact_match, f1_score = metric_max_over_ground_truths(
        drop_em_and_f1,
        prediction,
        ground_truth_answer_strings
    )

    return (exact_match, f1_score)




gold_dicts = [{'number': '', 'date': {'day': '', 'month': '', 'year': ''},
               'spans': ['Macromedia disclosed the Flash Version 3']},
              {'number': '', 'date': {'day': '', 'month': '', 'year': ''},
               'spans': ['disclosed the Flash Version 3']},
              {'number': '', 'date': {'day': '', 'month': '', 'year': ''},
               'spans': ['Flash Version 3']}]

predicted_answer = "Macromedia disclosed the Flash Version 3"


print(f1metric(predicted_answer, gold_dicts))