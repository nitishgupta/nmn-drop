import math


def max_dist(probs, samples=10):
    cum_dist = [probs[0]]
    for i in range(1, len(probs)):
        cum_dist.append(probs[i] + cum_dist[i - 1])
    cum_dist_n = [math.pow(x, samples) for x in cum_dist]
    cum_dist_strict_less_n = [0] + cum_dist_n[:-1]
    max_dist = [x - y for (x, y) in zip(cum_dist_n, cum_dist_strict_less_n)]
    return max_dist


"""
def second_max_dist(probs, samples=10):
    cum_dist = [probs[0]]
    for i in range(1, len(probs)):
        cum_dist.append(probs[i] + cum_dist[i-1])

    cum_dist_n = [math.pow(x, samples) for x in cum_dist]

    cum_dist_left_shift_n = cum_dist_n[1:] + [1]
    cum_dist_left2_shift_n = cum_dist_n[2:] + [1, 1]

    print()
    print(cum_dist_n)
    print(cum_dist_left_shift_n)
    print(cum_dist_left2_shift_n)
    print()

    second_max_dist = [x - y for (x,y) in zip(cum_dist_left2_shift_n, cum_dist_left_shift_n)]

    return second_max_dist
"""


def min_dist(probs, samples=10):
    inverse_cum_dist = [1]
    for i in range(1, len(probs)):
        inverse_cum_dist.append(inverse_cum_dist[i - 1] - probs[i - 1])
    inverse_cum_dist_n = [math.pow(x, samples) for x in inverse_cum_dist]
    inverse_cum_dist_shift_n = inverse_cum_dist_n[1:] + [0]
    min_dist = [x - y for (x, y) in zip(inverse_cum_dist_n, inverse_cum_dist_shift_n)]
    return min_dist


# numbers = [1, 3, 5, 7, 9, 11, 13]
# probs = [3, 5, 9, 7, 1]
probs = [0.193, 0.232, 0.002, 0.0, 0.002, 0.002, 0.0, 0.001, 0.401, 0.055, 0.108, 0.001, 0.0, 0.001, 0.001]

# print(numbers)

samples = 5

maxdist = max_dist(probs, samples=samples)
mindist = min_dist(probs, samples=samples)
# secondmaxdist = second_max_dist(probs, samples=samples)

print(f"Original: {probs}")
print(f"Maximum : {maxdist}")
print(f"Minimum : {mindist}")
