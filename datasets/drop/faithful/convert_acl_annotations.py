from typing import List, Dict, Tuple
import json
import argparse

import datasets.drop.constants as constants
from semqa.domain_languages.drop_language import Output
from semqa.utils.qdmr_utils import Node, lisp_to_nested_expression, nested_expression_to_tree, \
    read_json_dataset
from utils.util import _KnuthMorrisPratt

QTYPE2LISP = {
    "yeardiff_find_qtype": "(year_difference_single_event select_passage)",
    "yeardiff_find2_qtype": "(year_difference_two_events select_passage select_passage)",
    "count_find_qtype": "(aggregate_count select_passage)",
    "count_filterfind_qtype": "(aggregate_count select_passage)",
    "relocate_find_qtype": "(select_passagespan_answer (project_passage select_passage))",
    "relocate_maxfind_qtype": "(select_passagespan_answer (project_passage (select_max_num select_passage)))",
    "relocate_minfind_qtype": "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
    "relocate_maxfilterfind_qtype": "(select_passagespan_answer (project_passage (select_max_num select_passage)))",
    "relocate_minfilterfind_qtype": "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
    "number_comparison": "(select_passagespan_answer (compare_num_gt select_passage select_passage))",
    "date_comparison": "(select_passagespan_answer (compare_date_gt select_passage select_passage))",
    "num_find_qtype": "(select_num select_passage)",
    "num_filterfind_qtype": "(select_num select_passage)",
    "min_find_qtype": "(select_num (select_min_num select_passage))",
    "max_find_qtype": "(select_num (select_max_num select_passage))",
    "max_filterfind_qtype": "(select_num (select_max_num select_passage))",
    "min_filterfind_qtype": "(select_num (select_min_num select_passage))",
}


# ('yeardiff_find_qtype', 'find-one-event', 'ground-two-dates')
# ('yeardiff_find2_qtype', 'find-two-events', 'ground-two-dates')

# ('count_find_qtype', 'find-events')
# ('count_filterfind_qtype', 'find-events', 'after-filter-events')

# ('relocate_find_qtype', 'find-events', 'relocate')
# ('relocate_maxfind_qtype', 'find-events', 'find-nums', 'after-max-event', 'relocate')
# ('relocate_minfind_qtype', 'find-events', 'find-nums', 'after-min-event', 'relocate')
# ('relocate_maxfilterfind_qtype', 'find-events', 'after-filter-events', 'find-nums', 'after-max-event', 'relocate')
# ('relocate_minfilterfind_qtype', 'find-events', 'after-filter-events', 'find-nums', 'after-min-event', 'relocate')

# ('number_comparison', 'find-two-events', 'ground-two-nums')
# ('date_comparison', 'find-two-events', 'ground-two-dates')

# ('num_find_qtype', 'find-events', 'find-nums')
# ('num_filterfind_qtype', 'find-events', 'after-filter-events', 'find-nums')
# ('max_find_qtype', 'find-events', 'find-nums', 'after-max-event')
# ('min_find_qtype', 'find-events', 'find-nums', 'after-min-event')
# ('max_filterfind_qtype', 'find-events', 'after-filter-events', 'find-nums', 'after-max-event')
# ('min_filterfind_qtype', 'find-events', 'after-filter-events', 'find-nums', 'after-min-event')

def get_token_idx(charoffset, charidxs, tokens):
    tokenidx = None
    for tidx, _ in enumerate(charidxs):
        # charoffset is inclusive; so match starting from first char
        if charidxs[tidx] <= charoffset < charidxs[tidx] + len(tokens[tidx]):
            return tidx

    return None

def get_drop_tokenidxs(span, acl_passage, acl_charidxs, acl_passage_tokens,
                       drop_passage, drop_charidxs, drop_passage_tokens):
    global total, unequal
    start_token_idx, end_token_idx = span[0], span[1] - 1
    acl_start_charidx = acl_charidxs[start_token_idx]   # inclusive
    acl_end_charidx = acl_charidxs[end_token_idx] + len(acl_passage_tokens[end_token_idx]) # exclusive

    span_text = acl_passage[acl_start_charidx:acl_end_charidx]
    # Since ACL text is cleaned (chars removed), this span would be found at this position or after
    # We are hoping to find the span text within 5 extra characters
    potential_start = max(0, acl_start_charidx - 5)
    potential_end = acl_end_charidx + 10
    drop_potential_text = drop_passage[potential_start:potential_end]
    find_idxs = list(_KnuthMorrisPratt(drop_potential_text, span_text))

    if len(find_idxs) == 1:
        find_idx = find_idxs[0]
        drop_start_charidx = potential_start + find_idx
        drop_end_charidx = drop_start_charidx + len(span_text)   # exclusive
        # Both token-idxs inclusive
        start_token_idx = get_token_idx(charoffset=drop_start_charidx, charidxs=drop_charidxs,
                                        tokens=drop_passage_tokens)
        end_token_idx = get_token_idx(charoffset=drop_end_charidx-1, charidxs=drop_charidxs,
                                      tokens=drop_passage_tokens)
        if start_token_idx is None or end_token_idx is None:
            import pdb
            pdb.set_trace()
    else:
        start_token_idx = get_token_idx(charoffset=acl_start_charidx, charidxs=drop_charidxs,
                                        tokens=drop_passage_tokens)
        while start_token_idx is None:
            acl_start_charidx = acl_start_charidx + 1
            start_token_idx = get_token_idx(charoffset=acl_start_charidx, charidxs=drop_charidxs,
                                            tokens=drop_passage_tokens)

        end_token_idx = get_token_idx(charoffset=acl_end_charidx - 1, charidxs=drop_charidxs,
                                      tokens=drop_passage_tokens)
        while end_token_idx is None:
            acl_end_charidx -= 1
            end_token_idx = get_token_idx(charoffset=acl_end_charidx - 1, charidxs=drop_charidxs,
                                          tokens=drop_passage_tokens)

    return (start_token_idx, end_token_idx)


def get_drop_spans(acl_spans, acl_passage, acl_charidxs, acl_passage_tokens,
                   drop_passage, drop_charidxs, drop_passage_tokens):
    """Convert from ACL token-spans to DROP token-spans. Output token-spans are inclusive start/end."""
    drop_spans = []
    for span in acl_spans:
        drop_span = get_drop_tokenidxs(span, acl_passage, acl_charidxs, acl_passage_tokens, drop_passage, drop_charidxs,
                                       drop_passage_tokens)
        drop_spans.append(drop_span)
    return drop_spans


def print_spans(spans, tokens):
    for span in spans:
        print(tokens[span[0]:span[1] + 1])
    print()

FAITHFUL_ANNOTATION_KEY = "faithful_annotation"

def convert_annotations(faithful_dataset, drop_dataset):
    """Convert ACL annotations to newer format of using Node for programs and drop_language.Output for module outputs.

    In ACL annotations, each qa_pair contains an additional key, "module_output_annotations" which is a List containing
    ("module_name", List[Span]) tuples. Each span is token-offset tuple (start, end) with exclusive-end.
    The key "qtype": str contains the question-type that needs to be converted to a lisp -> Node program.

    Example, qtype = "max_find_qtype" --> (select_number (select_max_num select_passage))
    The annotations for this program are "find-events" (select_passage), "find-nums" (number_input in select_max_num),
    and "after-max-event"
    """

    output_dataset = {}
    total_qas = 0

    qtype_modules = set()
    for passage_id, passage_info in faithful_dataset.items():
        acl_passage_tokens = passage_info["tokenized_passage"].split(" ")
        acl_charidxs = passage_info[constants.passage_charidxs]
        acl_passage = passage_info["cleaned_passage"]   # This is the text that is used for tokenization :/

        drop_pinfo = drop_dataset[passage_id]
        drop_qas = drop_pinfo[constants.qa_pairs]
        drop_passage_tokens = drop_dataset[passage_id]["passage_tokens"]
        drop_charidxs = drop_dataset[passage_id][constants.passage_charidxs]
        drop_passage = drop_dataset[passage_id]["passage"]

        qas_to_add = []

        for qa in passage_info[constants.qa_pairs]:
            if "module_output_annotations" not in qa:
                continue
            qid = qa[constants.query_id]
            found = False
            drop_qa = None
            for dropqa in drop_qas:
                if qid == dropqa[constants.query_id]:
                    found = True
                    drop_qa = dropqa
                    break

            if found == False:
                print("NOT FOUND!")
                continue

            question = qa[constants.question]
            qtype = qa["qtype"]
            program_lisp = QTYPE2LISP[qtype]
            nested_exp = lisp_to_nested_expression(program_lisp)
            program_node: Node = nested_expression_to_tree(nested_exp)

            module_output_annotations: List[Tuple] = qa["module_output_annotations"]
            drop_module_output_annotations = []
            for module_name, acl_spans in module_output_annotations:
                drop_spans = get_drop_spans(acl_spans, acl_passage, acl_charidxs, acl_passage_tokens,
                                            drop_passage, drop_charidxs, drop_passage_tokens)
                drop_module_output_annotations.append((module_name, drop_spans))

            if qtype == "yeardiff_find_qtype":
                # (year_difference_single_event select_passage)
                select_spans = drop_module_output_annotations[0][1]
                date_spans = drop_module_output_annotations[1][1]
                # program_node.children[0].extras[FAITHFUL_ANNOTATION_KEY] = {"select": select_spans}
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"dates": date_spans, "select": select_spans}

            elif qtype == "yeardiff_find2_qtype":
                # (year_difference_two_events select_passage select_passage)
                select_spans = drop_module_output_annotations[0][1]
                date_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"dates": date_spans, "select": select_spans}

            elif qtype == "count_find_qtype":
                # (aggregate_count select_passage)
                select_spans = drop_module_output_annotations[0][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"select": select_spans}

            elif qtype == "count_filterfind_qtype":
                # (aggregate_count select_passage)
                select_spans = drop_module_output_annotations[0][1]
                filter_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"select": filter_spans}

            elif qtype == "relocate_find_qtype":
                # (select_passagespan_answer (project_passage select_passage))
                # 'find-events', 'relocate'
                select_spans = drop_module_output_annotations[0][1]
                project_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"project": project_spans, "select": select_spans}

            elif qtype == "relocate_maxfind_qtype" or qtype == "relocate_minfind_qtype":
                # (select_passagespan_answer (project_passage (select_max_num select_passage)))
                # 'find-events', 'find-nums', 'after-max-event', 'relocate'
                select_spans = drop_module_output_annotations[0][1]
                number_spans = drop_module_output_annotations[1][1]
                minmax_spans = drop_module_output_annotations[2][1]
                project_spans = drop_module_output_annotations[3][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"project": project_spans,
                                                                "input_numbers": number_spans,
                                                                "minmax": minmax_spans,
                                                                "select": select_spans}

            elif qtype == "relocate_maxfilterfind_qtype" or qtype == "relocate_minfilterfind_qtype":
                # (select_passagespan_answer (project_passage (select_max_num select_passage)))
                # 'find-events', 'after-filter-events', 'find-nums', 'after-max-event', 'relocate'
                select_spans = drop_module_output_annotations[0][1]
                filter_spans = drop_module_output_annotations[1][1]
                number_spans = drop_module_output_annotations[2][1]
                minmax_spans = drop_module_output_annotations[3][1]
                project_spans = drop_module_output_annotations[4][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"project": project_spans,
                                                                "input_numbers": number_spans,
                                                                "minmax": minmax_spans,
                                                                "select": filter_spans}

            elif qtype == "num_find_qtype":
                # (select_num select_passage)
                # 'find-events', 'find-nums'
                select_spans = drop_module_output_annotations[0][1]
                number_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"numbers": number_spans,
                                                                "select": select_spans}

            elif qtype == "num_filterfind_qtype":
                # (select_num select_passage)
                # 'find-events', 'after-filter-events', 'find-nums'
                select_spans = drop_module_output_annotations[0][1]
                filter_spans = drop_module_output_annotations[1][1]
                number_spans = drop_module_output_annotations[2][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"numbers": number_spans,
                                                                "select": filter_spans}

            elif qtype == "max_find_qtype" or qtype == "min_find_qtype":
                # (select_num (select_min_num select_passage))
                # 'find-events', 'find-nums', 'after-max-event'
                select_spans = drop_module_output_annotations[0][1]
                number_spans = drop_module_output_annotations[1][1]
                minmax_spans = drop_module_output_annotations[2][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"input_numbers": number_spans,
                                                                "minmax": minmax_spans,
                                                                "select": select_spans}

            elif qtype == "max_filterfind_qtype" or qtype == "min_filterfind_qtype":
                # (select_num (select_min_num select_passage))
                # 'find-events', 'after-filter-events', 'find-nums', 'after-max-event'
                select_spans = drop_module_output_annotations[0][1]
                filter_spans = drop_module_output_annotations[1][1]
                number_spans = drop_module_output_annotations[2][1]
                minmax_spans = drop_module_output_annotations[3][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"input_numbers": number_spans,
                                                                "minmax": minmax_spans,
                                                                "select": filter_spans}

            elif qtype == "number_comparison":
                # (select_passagespan_answer (compare_num_gt select_passage select_passage))
                # 'find-two-events', 'ground-two-nums'
                lesser_than_tokens = ["smaller", "fewer", "lowest", "smallest", "less", "least", "fewest", "lower"]
                if any([x in question for x in lesser_than_tokens]):
                    num_operator = "compare_num_lt"
                else:
                    num_operator = "compare_num_gt"
                program_lisp = "(select_passagespan_answer ({} select_passage select_passage))".format(num_operator)
                program_node = nested_expression_to_tree(lisp_to_nested_expression(program_lisp))
                select_spans = drop_module_output_annotations[0][1]
                number_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"numbers": number_spans,
                                                                "select": select_spans}

            elif qtype == "date_comparison":
                # (select_passagespan_answer (compare_date_gt select_passage select_passage))
                # 'find-two-events', 'ground-two-dates'
                lesser_tokens = ["first", "earlier", "forst", "firts"]
                if any([x in question for x in lesser_tokens]):
                    date_operator = "compare_date_lt"
                else:
                    date_operator = "compare_date_gt"
                program_lisp = "(select_passagespan_answer ({} select_passage select_passage))".format(date_operator)
                program_node = nested_expression_to_tree(lisp_to_nested_expression(program_lisp))

                select_spans = drop_module_output_annotations[0][1]
                date_spans = drop_module_output_annotations[1][1]
                program_node.extras[FAITHFUL_ANNOTATION_KEY] = {"dates": date_spans,
                                                                "select": select_spans}

            drop_qa[constants.program_supervision] = program_node.to_dict()
            qas_to_add.append(drop_qa)

        if qas_to_add:
            total_qas += len(qas_to_add)
            pinfo = drop_dataset[passage_id]
            pinfo[constants.qa_pairs] = qas_to_add
            output_dataset[passage_id] = pinfo

    print("Passages: {} Qa: {}".format(len(output_dataset), total_qas))

    return output_dataset


def main(args):

    faithfulness_annotation_dataset = read_json_dataset(args.acl_json)

    # In the ICLR21 splits (e.g. iclr_qdmr-v2-noexc) this is the test set, since acl annotations were from drop-dev
    # which is the test set in the iclr21 i.i.d. splits
    drop_dataset = read_json_dataset(args.drop_input_json)

    output_dataset = convert_annotations(faithful_dataset=faithfulness_annotation_dataset, drop_dataset=drop_dataset)

    output_json = args.output_json
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl_json")
    parser.add_argument("--drop_input_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)



