from semqa.utils.qdmr_utils import Node, nested_expression_to_tree, nested_expression_to_lisp
from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object

droplang = get_empty_language_object()

nested_expr = ['passagenumber_difference', ['select_num1', 'select_passage'], ['select_num2', 'select_passage']]
program_node: Node = nested_expression_to_tree(nested_expr)
lisp = nested_expression_to_lisp(program_node.get_nested_expression())
action_seq = droplang.logical_form_to_action_sequence(lisp)

print(lisp)
print(action_seq)


def compute_postorder_position(node: Node, linear_length: int):
    for c in node.children:
        linear_length = compute_postorder_position(c, linear_length)

    node.post_order = linear_length
    linear_length += 1
    return linear_length


def compute_inorder_position(node: Node, linear_length: int):
    node.in_order = linear_length
    linear_length += 1

    for c in node.children:
        linear_length = compute_inorder_position(c, linear_length)

    return linear_length


def compute_postorder_position_in_inorder_traversal(node: Node):
    postorder_positions = [node.post_order]

    for c in node.children:
        c_postorder_positions = compute_postorder_position_in_inorder_traversal(c)
        postorder_positions.extend(c_postorder_positions)

    return postorder_positions

compute_postorder_position(program_node, 0)
# compute_inorder_position(program_node, 0)
postorder_position_in_inorder_traversal = compute_postorder_position_in_inorder_traversal(program_node)


print(postorder_position_in_inorder_traversal)