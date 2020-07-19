import re
import json
from typing import List, Tuple, Set, Dict, Union, Any
from allennlp_semparse.common.util import lisp_to_nested_expression
from allennlp_semparse.domain_languages import DomainLanguage


class Node():
    def __init__(self, predicate, string_arg=None):
        self.predicate: str = predicate
        self.string_arg: str = string_arg
        # Empty list indicates leaf node
        self.children: List[Node] = []
        self.parent: Node = None    # None indicates root

        self.supervision: Dict[str, Any] = {}

    def add_child(self, obj):
        assert isinstance(obj, Node)
        obj.parent = self
        self.children.append(obj)

    def is_leaf(self):
        leaf = True if not len(self.children) else False
        return leaf

    def get_nested_expression(self):
        if not self.is_leaf():
            nested_expression = [self.predicate]
            for child in self.children:
                nested_expression.append(child.get_nested_expression())
            return nested_expression
        else:
            return self.predicate

    def get_nested_expression_with_strings(self):
        """ Nested expression where predicates w/ string_arg are written as PREDICATE(string_arg)
            This is introduced with drop_language since there are multiple predicates that select string_arg from text.
            Therefore, if we just write the string-arg to data (as done previously with typed_nested_expression), it
            would not be sufficient that some node is not a predicate in the language and hence get_ques_span pred.
            We can write the output of this function in json data and parse it back by removing split on ( and )
        """
        node_name = self.predicate
        if self.string_arg is not None:
            # GET_QUESTION_NUMBER(25)
            node_name = node_name + "(" + self.string_arg + ")"
        if self.is_leaf():
            return node_name
        else:
            nested_expression = [node_name]
            for child in self.children:
                nested_expression.append(child.get_nested_expression_with_strings())
            return nested_expression


    def get_prodigy_lisp(self):
        def get_string(node: Node):
            if node.string_arg is None:
                return node.predicate.upper()
            else:
                return node.predicate.upper() + "[[" + node.string_arg + "]]"

        if self.is_leaf():
            return get_string(self)
        else:
            lisps = []
            for child in self.children:
                lisps.append(child.get_prodigy_lisp())

            self_lisp = get_string(self)

            return "(" + " ".join([self_lisp] + lisps) + ")"

    def to_dict(self):
        json_dict = {
            "predicate": self.predicate,
            "string_arg": self.string_arg,
            "children": [c.to_dict() for c in self.children],
            "supervision": self.supervision,
        }
        return json_dict


def node_from_dict(dict: Dict) -> Node:
    predicate = dict["predicate"]
    string_arg = dict["string_arg"]
    node = Node(predicate=predicate, string_arg=string_arg)
    for c in dict["children"]:
        child_node = node_from_dict(c)
        node.add_child(child_node)
    node.supervision = dict["supervision"]
    return node


class QDMRExample(object):
    def __init__(self, q_decomp):
        self.query_id = q_decomp["question_id"]
        self.question = q_decomp["question_text"]
        self.split = q_decomp["split"]
        self.decomposition = q_decomp["decomposition"]
        self.program: List[str] = q_decomp["program"]
        self.nested_expression: List = q_decomp["nested_expression"]
        self.operators = q_decomp["operators"]
        # Filled by parse_dataset/qdmr_grammar_program.py if transformation to QDMR-language is successful
        # This contains string-args as it is
        self.typed_nested_expression: List = []
        if "typed_nested_expression" in q_decomp:
            self.typed_nested_expression = q_decomp["typed_nested_expression"]

        # This was added after completing parse_dataset/drop_grammar_program. This class is a moving target.
        # This nested_expression should be s.t. string-arg grounding predicates occur as PREDICATE(string-arg).
        # e.g. ['FILTER_NUM_EQ', ['SELECT', 'GET_QUESTION_SPAN(field goals of Mason)'], 'GET_QUESTION_NUMBER(37)']
        self.drop_nested_expression: List = []
        if "drop_nested_expression" in q_decomp:
            self.drop_nested_expression = q_decomp["drop_nested_expression"]

        self.program_tree: Node = None
        self.typed_masked_nested_expr = []
        if self.drop_nested_expression:
            self.program_tree: Node = nested_expression_to_tree(self.drop_nested_expression,
                                                                predicates_with_strings=True)
            # This contains string-args masked as GET_QUESTION_SPAN predicate
            # self.typed_masked_nested_expr = self.program_tree.get_nested_expression()

        elif self.typed_nested_expression:
            self.program_tree: Node = nested_expression_to_tree(self.typed_nested_expression,
                                                                predicates_with_strings=True)
            # This contains string-args masked as GET_QUESTION_SPAN predicate
            self.typed_masked_nested_expr = self.program_tree.get_nested_expression()

    def to_json(self):
        json_dict = {
            "question_id": self.query_id,
            "question_text": self.question,
            "split": self.split,
            "decomposition": self.decomposition,
            "program": self.program,
            "nested_expression": self.nested_expression,
            "typed_nested_expression": self.typed_nested_expression,
            "drop_nested_expression": self.drop_nested_expression,
            "operators": self.operators
        }
        return json_dict


def read_qdmr_json_to_examples(qdmr_json: str) -> List[QDMRExample]:
    """Parse processed qdmr json (from parse_dataset/parse_qdmr.py or qdmr_grammar_program.py into List[QDMRExample]"""
    qdmr_examples = []
    with open(qdmr_json, 'r') as f:
        dataset = json.load(f)
    for q_decomp in dataset:
        qdmr_example = QDMRExample(q_decomp)
        qdmr_examples.append(qdmr_example)
    return qdmr_examples


def write_qdmr_examples_to_json(qdmr_examples: List[QDMRExample], qdmr_json: str):
    examples_as_json_dicts = [example.to_json() for example in qdmr_examples]
    with open(qdmr_json, 'w') as outf:
        json.dump(examples_as_json_dicts, outf, indent=4)


def nested_expression_to_lisp(nested_expression):
    if isinstance(nested_expression, str):
        return nested_expression

    elif isinstance(nested_expression, List):
        lisp_expressions = [nested_expression_to_lisp(x) for x in nested_expression]
        return "(" + " ".join(lisp_expressions) + ")"
    else:
        raise NotImplementedError


def convert_nestedexpr_to_tuple(nested_expression):
    """Converts a nested expression list into a nested expression tuple to make the program hashable."""
    new_nested = []
    for i, argument in enumerate(nested_expression):
        if i == 0:
            new_nested.append(argument)
        else:
            if isinstance(argument, list):
                tupled_nested = convert_nestedexpr_to_tuple(argument)
                new_nested.append(tupled_nested)
            else:
                new_nested.append(argument)
    return tuple(new_nested)


def nested_expression_to_linearized_list(nested_expression, open_bracket: str = "(",
                                         close_bracket: str = ")") -> List[str]:
    """Convert the program (as nested expression) into a linearized expression.

        The natural language arguments in the program are kept intact as a single program `token` and it is the onus of
        the processing step after this to tokenize them
    """
    if isinstance(nested_expression, str):
        # If the string is not a predicate but a NL argument, it is the onus of the models to tokenize it appropriately
        return [nested_expression]

    elif isinstance(nested_expression, List):
        # Flattened list of tokens for each element in the list
        program_tokens = []
        for x in nested_expression:
            program_tokens.extend(nested_expression_to_linearized_list(x, open_bracket, close_bracket))
        # Inserting a bracket around the program tokens
        program_tokens.insert(0, open_bracket)
        program_tokens.append(close_bracket)
        return program_tokens
    else:
        raise NotImplementedError


def nested_expression_to_tree(nested_expression, predicates_with_strings: bool = True) -> Node:
    """ There are two types of expressions, one which have string-arg as it is and without a predicate (True)
    and other, newer DROP style, where string-arg nodes in expression are PREDICATE(string-arg)
    """
    if isinstance(nested_expression, str):
        if not predicates_with_strings:
            current_node = Node(predicate=nested_expression)
        else:
            predicate_w_stringarg = nested_expression
            # This can either be a plain predicate (e.g. `SELECT`) or with a string-arg (e.g. `GET_Q_SPAN(string-arg)`)
            start_paranthesis_index = predicate_w_stringarg.find("(")
            if start_paranthesis_index == -1:
                predicate = predicate_w_stringarg
                current_node = Node(predicate=predicate)
            else:
                predicate = predicate_w_stringarg[0:start_paranthesis_index]
                # +1 to avoid (, and -1 to avoid )
                string_arg = predicate_w_stringarg[start_paranthesis_index+1:-1]
                current_node = Node(predicate=predicate, string_arg=string_arg)

    elif isinstance(nested_expression, list):
        current_node = Node(nested_expression[0])
        for i in range(1, len(nested_expression)):
            child_node = nested_expression_to_tree(nested_expression[i], predicates_with_strings)
            current_node.add_child(child_node)
    else:
        raise NotImplementedError

    return current_node


def prodigy_lisp_to_nested_expression(prodigy_lisp: str) -> List:
    """
    Prodigy lisp example -- "(count[[how many]] (division[[2]] first[[field goal]]))", i.e.
    string arguments of predicates are written as [[string with spaces]]
    """
    def clean_nested(nested_expression):
        if isinstance(nested_expression, str):
            nested_expression = nested_expression.replace("[[", "(")
            nested_expression = nested_expression.replace("]]", ")")
            nested_expression = nested_expression.replace("@@@", " ")
            return nested_expression
        elif isinstance(nested_expression, List):
            # Flattened list of tokens for each element in the list
            program_tokens = []
            for x in nested_expression:
                program_tokens.append(clean_nested(x))
            return program_tokens
        else:
            raise NotImplementedError

    # replace space within [[string-arg]] with @@@
    prodigy_lisp = re.sub("\[\[[^]]*\]\]", lambda x: x.group(0).replace(' ', '@@@'), prodigy_lisp)
    # Convert this lisp into nested expression.
    prodigy_nested_expr = lisp_to_nested_expression(prodigy_lisp)
    # above would contain nodes like 'predicate[[yards@@@of@@@TD@@@passes]]'; need to replace [[, ]], @@@ with (, ), ' '
    nested_expr = clean_nested(prodigy_nested_expr)
    return nested_expr


def read_drop_dataset(input_json: str):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def convert_answer(answer_annotation: Dict[str, Union[str, Dict, List]]) -> Tuple[str, List]:
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key] for key in ["month", "day", "year"] if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def get_domainlang_function2returntype_mapping(language: DomainLanguage) -> Dict[str, str]:
    function2returntype_mapping = {}
    for name, function_type_list in language._function_types.items():
        for function_type in function_type_list:
            # {function_type} -> {name} are actions
            function_type_split = str(function_type).split(":")
            if len(function_type_split) == 1:  # e.g. PassageAttention -> select_passage
                return_type = function_type_split[0]
            else:  # e.g. <PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_addition
                return_type = function_type_split[-1][:-1]  # removing '>' from the end
            function2returntype_mapping[name] = return_type
    return function2returntype_mapping


def get_inorder_function_list(node: Node) -> List[str]:
    inorder_func_list = [node.predicate]
    for c in node.children:
        inorder_func_list.extend(get_inorder_function_list(c))
    return inorder_func_list


def get_postorder_function_list(node: Node) -> List[str]:
    postorder_func_list = []
    for c in node.children:
        postorder_func_list.extend(get_postorder_function_list(c))

    postorder_func_list.append(node.predicate)
    return postorder_func_list


def get_postorder_function_and_arg_list(node: Node) -> Tuple[List[str], List[str]]:
    postorder_func_list = []
    postorder_arg_list = []
    for c in node.children:
        c_func_list, c_arg_list = get_postorder_function_and_arg_list(c)
        postorder_func_list.extend(c_func_list)
        postorder_arg_list.extend(c_arg_list)

    postorder_func_list.append(node.predicate)
    postorder_arg_list.append(node.string_arg)
    return postorder_func_list, postorder_arg_list


def get_inorder_supervision_list(node: Node) -> List[Dict]:
    inorder_supervision_list = [node.supervision]
    for c in node.children:
        inorder_supervision_list.extend(get_inorder_supervision_list(c))
    return inorder_supervision_list


def function_to_action_string_alignment(program_node: Node, action_strings: List[str]) -> List[int]:
    """ Action strings is an in-order linearization of the program_node. It contains a intermediate non-terminal actions
    and terminal-function producing actions. The idea is to map functions (in an in-order list) to those actions
    that produce them. This can be used to map aux-function-supervision to the correct action's side-args in a NMN.

    E.g.
    nested-program:
    ['passagenumber_difference', ['select_num', 'select_passage'], ['select_num', 'select_passage']]
    linearized-function-list:
    ['passagenumber_difference', 'select_num', 'select_passage', 'select_num', 'select_passage']
    action-strings:
    ['@start@ -> ComposedNumber',
     'ComposedNumber -> [<PassageNumber,PassageNumber:ComposedNumber>, PassageNumber, PassageNumber]',
     '<PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_difference',
     ........... ]

    The return value should be a list of size = linearized-function-list that contains [2, ...] since
    passagenumber_difference gets decoded in the action-idx = 2.
    """
    # List of function-names in in-order form
    function_list: List[str] = get_inorder_function_list(program_node)

    function2action_idx_mapping: List[int] = []

    function_idx = 0
    for action_idx, action in enumerate(action_strings):
        rhs = action.split(" -> ")[1]
        if rhs == function_list[function_idx]:
            function2action_idx_mapping.append(action_idx)
            function_idx += 1

    assert function_idx == len(function_list), "All functions from function_list not found in action_strings"
    assert len(function2action_idx_mapping) == len(function_list)

    return function2action_idx_mapping


def write_jsonl(output_jsonl, output_json_dicts):
    with open(output_jsonl, 'w') as outf:
        for json_dict in output_json_dicts:
            outf.write(json.dumps(json_dict))
            outf.write("\n")


def read_jsonl(input_jsonl) -> List[Dict]:
    json_dicts = []
    with open(input_jsonl) as f:
        for line in f:
            json_dicts.append(json.loads(line))
    return json_dicts


if __name__ == "__main__":
    p = ['FILTER_NUM_GT', ['FILTER', ['SELECT', 'GET_QUESTION_SPAN(yards of TD passes)'],
                           'GET_QUESTION_SPAN(in the first half)'], 'GET_QUESTION_NUMBER(70)']

    node: Node = nested_expression_to_tree(p, predicates_with_strings=True)
    print(node.get_nested_expression_with_strings())
    print(node.get_nested_expression())

    prodigy_lisp = node.get_prodigy_lisp()
    print(prodigy_lisp)

    nested_expr = prodigy_lisp_to_nested_expression(prodigy_lisp)
    print(nested_expr)





    #
    # with open("test/node.json", 'w') as fp:
    #     json.dump(node.to_dict(), fp)
    #
    #
    # with open("test/node.json", 'r') as fp:
    #     node_dict = json.load(fp)
    #
    # print()
    # print(node_dict)
    # print()
    # n1: Node = node_from_dict(node_dict)
    # print(n1.get_nested_expression_with_strings())
    # print(n1.get_nested_expression())