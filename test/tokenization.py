import json
from collections import namedtuple

nested_expression = ["passagenumber_difference", ["find-num", "find"], ["find-num", "find"]]

module = namedtuple('Module', ['name', 'identifier'])

m1 = module(name="find-num", identifier=1)


class Module:
    def __init__(self, name:str, identifier: int):
        self.name = name
        self.identifier = identifier

    def to_dict(self):
        json_dict = {
            "name": self.name,
            "identifier": self.identifier
        }
        return json_dict


def add_identifier(nested_expression, count):
    """Convert the nested_expression into a representation that contains the order in which the modules are executed.

    This function converts the nested_expression of module-as-str into expression with module-as-Module class where the
    class stores an `identifier` key which is the number at which the module was executed.

    Since the program-tree is executed in a left-to-right post-traversal order we will traverse the tree in a similar
    manner to number the modules in the nested-expression.
    """
    # If expression is not a list (hence a str) it's a Module
    if not isinstance(nested_expression, list):
        return Module(name=nested_expression, identifier=count), count + 1
    # If expression is tree
    else:
        sub_expression = []
        # Performing left-to-right post traversal of the tree
        for i in range(1, len(nested_expression)):
            arg_i, count = add_identifier(nested_expression[i], count)
            sub_expression.append(arg_i)
        # Then add the root-module of the tree
        arg_0 = Module(name=nested_expression[0], identifier=count)
        sub_expression.insert(0, arg_0)

        return sub_expression, count + 1


def convert_module_expression_tree_to_dict(module_expression):
    mapped_expression = []
    for i, argument in enumerate(module_expression):
        if isinstance(argument, list):
            mapped_expression.append(convert_module_expression_tree_to_dict(argument))
        elif isinstance(argument, Module):
            mapped_expression.append(argument.to_dict())
        else:
            raise NotImplementedError
    return mapped_expression



module_expression, _ = add_identifier(nested_expression, count=1)
expression_as_dict = convert_module_expression_tree_to_dict(module_expression)
print(expression_as_dict)

