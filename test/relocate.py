import itertools

numbers = [1, 2, 3, 4, 5]
max_number_of_numbers_to_consider = 2

combinations = []
signs = [-1, 1]

number_support = set()
all_sign_combinations = list(itertools.product(signs, repeat=2))
for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
    # for number_combination in itertools.combinations(numbers, number_of_numbers_to_consider):
    for number_combination in itertools.product(numbers, repeat=number_of_numbers_to_consider):
        print(number_combination)
        for sign_combination in all_sign_combinations:
            value = sum([sign * num for (sign, num) in zip(sign_combination, number_combination)])
            number_support.add(value)

number_support = sorted(list(number_support))
print(number_support)
