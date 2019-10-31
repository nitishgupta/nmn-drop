from typing import Callable, List

from allennlp_semparse import DomainLanguage, predicate

class Arithmetic(DomainLanguage):
    def __init__(self):
        super().__init__(
            start_types={int},
            allowed_constants={
                # We unfortunately have to explicitly enumerate all allowed constants in the
                # grammar.  Because we'll be inducing a grammar for this language for use with a
                # semantic parser, we need the grammar to be finite, which means we can't allow
                # arbitrary constants (you can't parameterize an infinite categorical
                # distribution).  So our Arithmetic language will have to only operate on simple
                # numbers.
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
                "10": 10,
                "20": 20,
                "-5": -5,
                "-2": -2,
            },
        )

    @predicate
    def add(self, num1: int, num2: int) -> int:
        return num1 + num2

    @predicate
    def sum(self, numbers: List[int]) -> int:
        return sum(numbers)

    # Unfortunately, to make lists, we need to have some function with a fixed number of list
    # elements that we can predict.  No variable number of arguments - that gives us an infinite
    # number of production rules in our grammar.
    @predicate
    def list1(self, num1: int) -> List[int]:
        return [num1]

    @predicate
    def list2(self, num1: int, num2: int) -> List[int]:
        return [num1, num2]

    @predicate
    def list3(self, num1: int, num2: int, num3: int) -> List[int]:
        return [num1, num2, num3]

    @predicate
    def list4(self, num1: int, num2: int, num3: int, num4: int) -> List[int]:
        return [num1, num2, num3, num4]

    @predicate
    def subtract(self, num1: int, num2: int) -> int:
        return num1 - num2

    @predicate
    def power(self, num1: int, num2: int) -> int:
        return num1 ** num2

    @predicate
    def multiply(self, num1: int, num2: int) -> int:
        return num1 * num2

    @predicate
    def divide(self, num1: int, num2: int) -> int:
        return num1 // num2

    @predicate
    def halve(self, num1: int) -> int:
        return num1 // 2

    @predicate
    def three(self) -> int:
        return 3

    @predicate
    def three_less(self, function: Callable[[int, int], int]) -> Callable[[int, int], int]:
        """
        Wraps a function into a new function that always returns three less than what the original
        function would.  Totally senseless function that's just here to test higher-order
        functions.
        """

        def new_function(num1: int, num2: int) -> int:
            return function(num1, num2) - 3

        return new_function


if __name__ == "__main__":
    language = Arithmetic()
    all_prods = language.all_possible_productions()

    print("All prods:\n{}\n".format(all_prods))

    nonterm_prods = language.get_nonterminal_productions()
    print("Non terminal prods:\n{}\n".format(nonterm_prods))

    functions = language._functions
    print("Functions:\n{}\n".format(functions))

    function_types = language._function_types
    print("Function Types:\n{}\n".format(function_types))
