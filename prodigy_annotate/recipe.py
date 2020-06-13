import prodigy
from prodigy.components.loaders import JSONL
from semqa.utils.qdmr_utils import nested_expression_to_tree
import random
random.seed(42)

# "question": question,
# "passage": passage,
# "query_id": query_id,
# "nested_expr": nested_expr,
# "nested_tuple": nested_tuple,
# "lisp": lisp,
# "prodigy_lisp": prodigy_lisp,
# "answer_annotation": answer_annotation,
# "answer_list": answers,
# "answer_passage_spans": answer_passage_spans,
# "passage_number_values": passage_number_values,
# "passage_date_values": passage_date_values,
# "year_differences": year_differences,


DROP_FUNCTIONS_LIST = ['select_passagespan_answer', 'select_num', 'select_passage', 'compare_date_gt',
                       'compare_date_lt', 'compare_num_gt', 'compare_num_lt', 'filter_passage',
                       'passagenumber_addition', 'passagenumber_difference', 'project_passage', 'select_implicit_num',
                       'select_max_num', 'select_min_num', 'year_difference_single_event', 'year_difference_two_events',
                       'extract_passagespan_answer', 'aggregate_count']

DROP_FUNCTIONS = "  ||  ".join(DROP_FUNCTIONS_LIST)
DROP_FUNCTIONS = DROP_FUNCTIONS.upper()

COMMON_TEMPLATES_LIST = [
    "(SELECT_PASSAGESPAN_ANSWER SELECT_PASSAGE)",
    "(PASSAGENUMBER_DIFFERENCE (SELECT_NUM SELECT_PASSAGE) (SELECT_NUM SELECT_PASSAGE))",
    "(AGGREGATE_COUNT SELECT_PASSAGE)",
    "(SELECT_PASSAGESPAN_ANSWER (COMPARE_NUM_GT SELECT_PASSAGE SELECT_PASSAGE))",
    "(SELECT_PASSAGESPAN_ANSWER (PROJECT_PASSAGE SELECT_PASSAGE))",
    "(YEAR_DIFFERENCE_TWO_EVENTS SELECT_PASSAGE SELECT_PASSAGE)",
    "(SELECT_PASSAGESPAN_ANSWER (COMPARE_DATE_LT SELECT_PASSAGE SELECT_PASSAGE))",
    "(YEAR_DIFFERENCE_SINGLE_EVENT SELECT_PASSAGE)",
    "(SELECT_NUM SELECT_PASSAGE)",
    "(SELECT_PASSAGESPAN_ANSWER (FILTER_PASSAGE SELECT_PASSAGE))",
    "(AGGREGATE_COUNT (FILTER_PASSAGE SELECT_PASSAGE))",
    "(SELECT_PASSAGESPAN_ANSWER (COMPARE_NUM_LT SELECT_PASSAGE SELECT_PASSAGE))",
    "(PASSAGENUMBER_DIFFERENCE SELECT_IMPLICIT_NUM (SELECT_NUM SELECT_PASSAGE))",
    "(PASSAGENUMBER_ADDITION (SELECT_NUM SELECT_PASSAGE) (SELECT_NUM SELECT_PASSAGE))",
    "(SELECT_PASSAGESPAN_ANSWER (COMPARE_DATE_GT SELECT_PASSAGE SELECT_PASSAGE))"
]

COMMON_TEMPLATES = "\n".join(COMMON_TEMPLATES_LIST)
COMMON_TEMPLATES = COMMON_TEMPLATES.upper()


def add_task_info(stream):
    tasks = []
    for drop_dict in stream:
        if "prodigy_lisp" not in drop_dict:
            continue

        drop_dict.update(
            {
                # This is an input field name to pre-populate; check recipe return.
                "program": drop_dict["prodigy_lisp"],
            })
        # yield task
        tasks.append(drop_dict)
    return tasks


@prodigy.recipe(
    "drop-recipe",
    dataset=("Dataset to save answers to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
    view_id=("Annotation interface", "option", "v", str),
)
def my_custom_recipe(dataset, file_path, view_id="text_input"):
    def update(examples):
        nonlocal total_annotated
        total_annotated += len(examples)
        print(f"Received {len(examples)} annotations!")
        return total_annotated

    def get_progress(*args, **kwargs):
        nonlocal total_examples
        progress = float(total_annotated)/float(total_examples)
        return progress

    # Load your own streams from anywhere you want
    stream = JSONL(file_path)     # load in the JSONL file

    tasks = add_task_info(stream)
    # random.shuffle(tasks)

    total_examples = len(tasks)
    total_annotated = 0

    stream = tasks
    print(f"Total number of examples: {stream.__len__()}")

    return {
        "dataset": dataset,
        "stream": stream,
        "update": update,
        "view_id": "blocks",
        "config": {
            "blocks": [
                {"view_id": "html",
                 "drop_functions": DROP_FUNCTIONS,
                 "common_templates": COMMON_TEMPLATES,
                 "html_template": get_html_template()
                },
                {"view_id": "text_input",
                 "field_id": "program",
                 "field_label": "Lisp Program",
                 "field_placeholder": "Type here...",
                 "field_rows": 3,
                },
                {"view_id": "text_input",
                 "field_id": "remarks",
                 "field_label": "Remarks",
                 "field_placeholder": "",
                 "field_rows": 1,
                },
            ],
        },
        "progress": get_progress,
    }

def get_html_template():
    html_template = (
        "<div style='text-align: center; width: 100%'>"
        "<div style='padding: 10px; border-bottom: 1px solid #ccc'><strong>Functions:</strong><br> {{drop_functions}}</div>"
        "<div style='padding-top: 10px; padding-bottom: 20px; border-bottom: 2px solid #2b2b2b'><strong>Common Templates:</strong><br> {{common_templates}}</div>"
        "<div style='padding: 10px;'><strong>Question:</strong> {{question}}</div>"
        "<div style='padding: 10px;'><strong>Passage:</strong> {{passage}}</div>"
        "<div style='padding: 10px;'><strong>Answer:</strong> {{answer_list}}</div>"        
        "<div style='padding: 10px;'><strong>Lisp:</strong> {{lisp}}</div>"
        "</div>"
    )
    return html_template


# def filter_instances(tasks):
#     filtered = []
#     print("Before filtering: {}".format(len(tasks)))
#     for task in tasks:
#         nested_expr = task["nested_expr"]
#         program_node = nested_expression_to_tree(nested_expr)
#         if program_node.predicate == "select_passagespan_answer" and \
#                 len(program_node.children) == 1 and program_node.children[0].predicate == "select_passage":
#             continue
#         else:
#             filtered.append(task)
#
#     print("After filtering: {}".format(len(filtered)))
#     return filtered


# def get_task_stream(tasks):
#     global x
#     while True and x < len(tasks):
#         print(x)
#         yield tasks[x]
#         x += 1


