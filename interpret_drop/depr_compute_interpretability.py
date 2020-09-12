from typing import List, Dict, Tuple
import json
import argparse
import numpy as np
from collections import defaultdict
import interpret_drop.logical_form_supervision as lfs


QTYPES = ['date_comparison', 'number_comparison', 'num_find_qtype', 'num_filterfind_qtype', 'min_find_qtype',
          'min_filterfind_qtype', 'max_find_qtype', 'max_filterfind_qtype', 'count_find_qtype',
          'count_filterfind_qtype', 'relocate_find_qtype', 'relocate_filterfind_qtype', 'relocate_maxfind_qtype',
          'relocate_maxfilterfind_qtype', 'relocate_minfind_qtype', 'relocate_minfilterfind_qtype',
          'yeardiff_find_qtype', 'yeardiff_find2_qtype']

# Merging different question-types that have the same annotation and module-output structure
# This will help us reduce the number of functions we implement to compute interpretability
# In the worst case, we'll have to write a separate function for each question type.
SUPERVISION_TO_QTYPES = {
    'TWO_FINDS_TWO_SYMBOL': ['date_comparison', 'number_comparison', 'yeardiff_find2_qtype'],
    'FIND_TWO_SYMBOL': ['yeardiff_find_qtype'],
    'ONE': ['count_find_qtype'],
    'TWO': ['num_find_qtype', 'relocate_find_qtype', 'count_filterfind_qtype'],
    'THREE': ['max_find_qtype', 'min_find_qtype', 'num_filterfind_qtype', 'relocate_filterfind_qtype'],
    'FOUR': ['max_filterfind_qtype', 'min_filterfind_qtype', 'relocate_maxfind_qtype', 'relocate_minfind_qtype'],
    'FIVE': ['relocate_maxfilterfind_qtype', 'relocate_minfilterfind_qtype'],
}

# Mapping from QTYPE to what gold/predicted passage-attentions to expect
QTYPE_TO_SUPERVISION: Dict[str, str] = {}
for supervision_type, ques_types in SUPERVISION_TO_QTYPES.items():
    for qtype in ques_types:
        if qtype in QTYPE_TO_SUPERVISION:
            raise RuntimeError("Duplicate Q-type already in the map: {}".format(qtype))
        QTYPE_TO_SUPERVISION[qtype] = supervision_type


MODULEWISE_INTERPRETABILITY = defaultdict(float)
MODULEWISE_COUNT = defaultdict(float)

# This would contain all scores. This will be used for significance testing
MODULE_SCORE = []

def compute_interpretability_loss(passage_attention: np.array, spans: List[Tuple]):
    interpretability_loss = 0.0
    for span in spans:
        span_prob = np.sum(passage_attention[span[0]:span[1]])
        # span_prob = max(1e-20, span_prob)
        if span_prob > 1e-20:
            span_neg_log_prob = -1.0 * np.log(span_prob)
            interpretability_loss += span_neg_log_prob
        else:
            span_neg_log_prob = -1.0 * np.log(1e-20)
            interpretability_loss += span_neg_log_prob
    # interpretability_loss /= float(len(spans))
    return interpretability_loss


"""
All interpretability functions get as input predicted_module_outputs & gold_module_outputs
Each is a list of tuples -- (module_name, module_output)
"""
def interpretability_FIND_TWO_SYMBOL(predicted_module_outputs, gold_module_outputs):
    """ The gold for this would contain two modules, "find-one-event" and "ground-two-{dates, nums}"
        The predicted for this would contain a single "find" and two "date1" and "date2" / "num1" and "num2"

        The predicted "find" can be directly compared to gold "find-one-event", but
        the two predicted symbol distributions need to be merged into one and compared to "ground-two-{dates, nums}"
    """
    gold_find_distribution = gold_module_outputs[0][1]
    gold_symbol_distribution = gold_module_outputs[1][1]

    predicted_find_distribution = predicted_module_outputs[0][1]
    predicted_symbol_1_distribution = predicted_module_outputs[1][1]
    predicted_symbol_2_distribution = predicted_module_outputs[2][1]
    predicted_symbol_distribution = np.maximum(predicted_symbol_1_distribution, predicted_symbol_2_distribution)

    find_interpretability_loss = compute_interpretability_loss(predicted_find_distribution, gold_find_distribution)
    symbols_interpretability_loss = compute_interpretability_loss(predicted_symbol_distribution,
                                                                  gold_symbol_distribution)
    final_interpretability_loss = find_interpretability_loss + symbols_interpretability_loss

    # Removing the training 1 in num1/date1
    symbolname = predicted_module_outputs[1][0][:-1]
    MODULEWISE_INTERPRETABILITY[symbolname] += symbols_interpretability_loss
    MODULEWISE_COUNT[symbolname] += 1
    MODULEWISE_INTERPRETABILITY["find"] += find_interpretability_loss
    MODULEWISE_COUNT["find"] += 1

    MODULE_SCORE.append(find_interpretability_loss)

    return final_interpretability_loss


def interpretability_TWO_FINDS_TWO_SYMBOL(predicted_module_outputs, gold_module_outputs):
    """ The gold for this would contain two modules, "find-two-events" and "ground-two-{dates, nums}"
        The predicted for this would first contain a two "finds" and then two "{date,num}{1,2}"

        The predicted "find"s need to be combined to be compared with the gold "find-two-events"
        the two predicted symbol distributions need to be merged into one and compared to "ground-two-{dates, nums}"
    """
    gold_find_distribution = gold_module_outputs[0][1]
    gold_symbol_distribution = gold_module_outputs[1][1]

    predicted_find_1_distribution = predicted_module_outputs[0][1]
    predicted_find_2_distribution = predicted_module_outputs[1][1]
    predicted_symbol_1_distribution = predicted_module_outputs[2][1]
    predicted_symbol_2_distribution = predicted_module_outputs[3][1]

    predicted_find_distribution = np.maximum(predicted_find_1_distribution, predicted_find_2_distribution)
    predicted_symbol_distribution = np.maximum(predicted_symbol_1_distribution, predicted_symbol_2_distribution)

    find_interpretability_loss = compute_interpretability_loss(predicted_find_distribution, gold_find_distribution)
    symbols_interpretability_loss = compute_interpretability_loss(predicted_symbol_distribution,
                                                                  gold_symbol_distribution)
    final_interpretability_loss = find_interpretability_loss + symbols_interpretability_loss

    symbolname = predicted_module_outputs[2][0][:-1]
    MODULEWISE_INTERPRETABILITY[symbolname] += symbols_interpretability_loss
    MODULEWISE_COUNT[symbolname] += 1
    MODULEWISE_INTERPRETABILITY["find"] += find_interpretability_loss
    MODULEWISE_COUNT["find"] += 1

    MODULE_SCORE.append(find_interpretability_loss)

    return final_interpretability_loss


def interpretability_N(predicted_module_outputs, gold_module_outputs, N: str):
    """ The first modules are used for interpretability evaluation """
    num_mapping = {'ONE':1, 'TWO':2, 'THREE':3, 'FOUR':4, 'FIVE':5}
    N: int = num_mapping[N]
    final_interpretability_loss = 0.0
    for n in range(N):
        predicted_module_name = predicted_module_outputs[n][0]
        gold_module_name = gold_module_outputs[n][0]
        predicted_find_attention = predicted_module_outputs[n][1]
        gold_find_spans = gold_module_outputs[n][1]
        interpretability_loss = compute_interpretability_loss(predicted_find_attention, gold_find_spans)
        final_interpretability_loss += interpretability_loss
        # print(f"{gold_module_name}  {predicted_module_name}")
        MODULEWISE_INTERPRETABILITY[predicted_module_name] += interpretability_loss
        MODULEWISE_COUNT[predicted_module_name] += 1

        MODULE_SCORE.append(interpretability_loss)

    return final_interpretability_loss


class Example:
    def __init__(self, example_dict: Dict):
        self.passage_id = example_dict["passage_id"]
        self.query_id = example_dict["query_id"]
        self.question = example_dict["question"]
        self.qtype = example_dict["qtype"]
        self.predicted_program = example_dict["predicted_logical_form"]
        self.module_outputs = self.convert_attention_to_np(example_dict["module_outputs"])


    def convert_attention_to_np(self, module_outputs):
        module_outputs_np = []
        for (module_name, attention) in module_outputs:
            np_attention = convert_to_nparray(attention)
            module_outputs_np.append((module_name, np_attention))
        return module_outputs_np


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def read_jsonl(input_jsonl):
    dicts = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            dicts.append(json.loads(line))
    return dicts


def convert_to_nparray(attention_score):
    return np.array(attention_score, dtype=np.float32)


def get_queryid2annotations(module_output_gold: Dict):
    qid2annotations = {}
    for pid, pinfo in module_output_gold.items():
        qa_pairs = pinfo["qa_pairs"]
        for qa in qa_pairs:
            query_id = qa["query_id"]
            annotations = qa["module_output_annotations"]
            qid2annotations[query_id] = annotations

    return qid2annotations


def compute_interpretability_score(module_output_predictions: List[Dict], module_output_gold: Dict):
    """ Compute interpretability scores given module predictions and gold-annotations

    For each question we assume the prediction is made for the gold program. Each prediction hence would contain a
    passage-attention for the relevant modules in the program. Similarly, the gold data would contain gold passage
    attentions for the relevant modules. Interpretability score will be computed using the predicted and gold p-attn.

    The gold data sometimes combines p-attn for multiple modules, mainly for the find-module. For e.g. date-compare
    requires two calls to the find module and hence the predicted attentions would contain two calls to the same.
    Whereas, the gold annotation would contain a single annotation under the name of "find-events".

    Interpretability score for a (predicted, gold) passage attention pair is the cross-entropy loss.


    Args:
    module_output_predictions: `List[Dict]`
        Each example contains "query_id", "question", "qtype", "f1", "em", "predicted_logical_form", and
        "module_outputs" as keys. "module_outputs" is a list of (module_name, pattn) tuples

    module_output_gold: `Dict`
        Subset of the DROP dev data which is Interpret-dev gold data. Each qa_pair contains an additional key,
        "module_output_annotations" which is a List containing ("module_name", List[Span]) annotations.
        Each span is token-offset tuple (start, end) with exclusive-end.
    """
    skipped_due_to_predicted = 0
    qid2annotations = get_queryid2annotations(module_output_gold)
    interpretability_loss = 0.0
    num_examples = 0
    for example_dict in module_output_predictions:
        example = Example(example_dict)
        gold_annotations = qid2annotations[example.query_id]
        qtype = example.qtype
        gold_logical_forms = lfs.qtype2logicalforms[qtype]

        # TODO: Add an assert that the predicted logical form conforms to the gold-qtype
        if example.predicted_program not in gold_logical_forms:
            skipped_due_to_predicted += 1
            print(example.question)
            print(example.qtype)
            print(example.predicted_program)
            continue

        num_examples += 1
        # Depending on the qtype, we have expectations on the modules that the annotation and gold would contain.
        # We would now compute the interpretability score based on that
        supervision_type = QTYPE_TO_SUPERVISION[qtype]
        if supervision_type == 'TWO_FINDS_TWO_SYMBOL':
            interpretability_loss += interpretability_TWO_FINDS_TWO_SYMBOL(
                predicted_module_outputs=example.module_outputs, gold_module_outputs=gold_annotations)
        elif supervision_type == 'FIND_TWO_SYMBOL':
            interpretability_loss += interpretability_FIND_TWO_SYMBOL(predicted_module_outputs=example.module_outputs,
                                                                      gold_module_outputs=gold_annotations)
        elif supervision_type in ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']:
            interpretability_loss += interpretability_N(predicted_module_outputs=example.module_outputs,
                                                        gold_module_outputs=gold_annotations, N=supervision_type)
        else:
            print(supervision_type)
            # raise NotImplementedError("Supervision Type is unknown: {}".format(supervision_type))

    interpretability_loss = float(interpretability_loss)/num_examples
    print("Skipped due to predicted: {}".format(skipped_due_to_predicted))
    print("Interpretability Loss (lower is better): {}".format(interpretability_loss))


    # Merge min and max interpretability scores --
    keys_to_merge = ['max-pattn', 'min-pattn']
    MODULEWISE_INTERPRETABILITY['minmax-pattn'] = sum([MODULEWISE_INTERPRETABILITY[x] for x in keys_to_merge])
    MODULEWISE_COUNT['minmax-pattn'] = sum([MODULEWISE_COUNT[x] for x in keys_to_merge])
    for key in keys_to_merge:
        MODULEWISE_INTERPRETABILITY.pop(key)
        MODULEWISE_COUNT.pop(key)

    keys_to_merge = ['find-date', 'find-num']
    MODULEWISE_INTERPRETABILITY['find-arg'] = sum([MODULEWISE_INTERPRETABILITY[x] for x in keys_to_merge])
    MODULEWISE_COUNT['find-arg'] = sum([MODULEWISE_COUNT[x] for x in keys_to_merge])
    # for key in keys_to_merge:
    #     MODULEWISE_INTERPRETABILITY.pop(key)
    #     MODULEWISE_COUNT.pop(key)

    MODULEWISE_INTERPRETABILITY_AVG = {}
    micro_total_score = 0.0
    micro_sum = 0


    for module, int_score in MODULEWISE_INTERPRETABILITY.items():
        print(module)
        MODULEWISE_INTERPRETABILITY_AVG[module] = int_score/MODULEWISE_COUNT[module]
        micro_total_score += int_score
        micro_sum += MODULEWISE_COUNT[module]

    micro_avg = micro_total_score/micro_sum

    print("Interpretability Micro Avg (lower is better): {}".format(micro_avg))

    print(MODULEWISE_INTERPRETABILITY_AVG)

    with open("interpret_drop/strong.txt", 'w') as outf:
        for score in MODULE_SCORE:
            outf.write(str(score))
            outf.write("\n")


def main(args):
    module_output_pred = read_jsonl(args.module_output_pred_jsonl)
    module_output_gold = readDataset(args.module_output_anno_json)

    compute_interpretability_score(module_output_predictions=module_output_pred,
                                   module_output_gold=module_output_gold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_output_pred_jsonl")
    parser.add_argument("--module_output_anno_json")
    args = parser.parse_args()

    main(args)
