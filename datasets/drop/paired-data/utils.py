from typing import List, Tuple, Dict, Union, Set

import numpy as np
import itertools
from collections import defaultdict

from semqa.domain_languages.drop_language import Date


def compute_number_support(
        numbers: List[Union[int, float]],
        implicit_numbers: List[Union[int, float]] = None,
        max_number_of_numbers_to_consider: int = 2,
) -> Tuple[List[Union[int, float]], List[Union[int, float]], Dict, Dict, Set, Set]:
    """Compute the number support based on combinations of input numbers.
    This function considers all possible addition/subtraction between all pairs of numbers (even self). This forms
    the support of the possible answers. The output is a sorted list of number support.

    Args:
        numbers: input numbers -- usually passage numbers
        implicit_numbers: Extra numbers not part of the passage, but added in language. E.g. 100, 0
        max_number_of_numbers_to_consider: number of numbers to consider to combine
    Returns:
        composed_numbers: List of output composed numbers (also includes implicit numbers)
        compnumber2addcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
        compnumber2subcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
            Map from number to set of number combinations that can create it using the addition/sub operator.
            For example, {2: set((1,1), (0,2))} is a valid entry for addcombinations
    """
    if max_number_of_numbers_to_consider > 2:
        raise NotImplementedError

    passagenums_w_implicitnums = [x for x in numbers]
    # Adding implicit numbers here after checking if 0 is a part of original numbers so that we don't add tons of
    #  combinations of the kind x = x + 0 / x - 0
    zero_in_passage = True if 0 in numbers else False
    # Adding implicit-numbers to the input-numbers list since they can take part in composition with input-numbers.
    if implicit_numbers:
        passagenums_w_implicitnums.extend(implicit_numbers)

    composed_num_set = set()
    # Map from composed-number to list of number-combination that lead to this number from the add/sub operation
    compnumber2subcombinations = defaultdict(set)
    compnumber2addcombinations = defaultdict(set)
    nums_from_addition = set()
    nums_from_subtraction = set()
    signs = [-1, 1]
    # all_sign_combinations = list(itertools.product(signs, repeat=2))
    # Since our modules will only perform num1-num2 / num1+num2. Computation like -num1+num2 would not be done
    all_sign_combinations = [(1.0, -1.0), (1.0, 1.0)]
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        # for number_combination in itertools.combinations(numbers, r=number_of_numbers_to_consider):
        for indexed_number_combination in itertools.product(
                enumerate(passagenums_w_implicitnums), repeat=number_of_numbers_to_consider
        ):
            ((idx1, num1), (idx2, num2)) = indexed_number_combination
            number_combination = (num1, num2)
            # if idx1 == idx2: continue     # Commented: 0 in support. Un-commented: 0 not in support
            # print(indexed_number_combination)
            for sign_combination in all_sign_combinations:
                value = sum([sign * num for (sign, num) in zip(sign_combination, number_combination)])
                if value >= 0:
                    # If 0 was originally in numbers then allow its combinations, o/w don't to avoid the
                    # combinations from getting bloated with x = x+0, 0+x, x-0
                    if (0 in number_combination and zero_in_passage) or (0 not in number_combination):
                        composed_num_set.add(value)
                        if sign_combination == (1, 1):
                            compnumber2addcombinations[value].add(number_combination)
                            nums_from_addition.add(value)
                        else:  # sign_combination == [1, -1]:
                            compnumber2subcombinations[value].add(number_combination)
                            nums_from_subtraction.add(value)

    composed_numbers = sorted(list(composed_num_set))

    return (composed_numbers, passagenums_w_implicitnums, compnumber2addcombinations, compnumber2subcombinations,
            nums_from_addition, nums_from_subtraction)


def get_year_difference_candidates(passage_date_objs: List[Date]) -> Tuple[List[int], np.array]:
    """ List of integers indicating all-possible year differences between the passage-dates
        If year difference is not defined (year = -1) or negative, we don't consider such date-combinations

        Returns the following:

        Returns:
        ---------
        year_differences:
            List[int] These are the possible year differences.
        year_difference_mat: Binary np.array of shape (D, D, y_d)
            Entry (i, j, k) == 1 denotes that D[i] - D[j] == year_differences[k]
    """
    num_date_objs = len(passage_date_objs)
    # Adding zero-first since it'll definitely be added and makes sanity-checking easy
    year_differences: List[int] = [0]

    yeardiff2combs = {0: []}

    # If any year is -1, we consider the year difference to be 0
    # If the year difference is negative, we consider the difference to be 0
    for (date1, date2) in itertools.product(passage_date_objs, repeat=2):
        year_diff = date1.year_diff(date2)
        if year_diff >= 0:
            if year_diff not in year_differences:
                year_differences.append(year_diff)
                yeardiff2combs[year_diff] = []
            yeardiff2combs[year_diff].append((date1, date2))

    return year_differences, yeardiff2combs