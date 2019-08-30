import torch
from allennlp.nn import util


def max_number_distribution(num_dist):
    cum_dist = num_dist.cumsum(0)
    cum_dist_n = cum_dist ** 5
    maximum_distribution = cum_dist_n - torch.cat([cum_dist_n.new_zeros(1), cum_dist_n[:-1]])
    maximum_distribution = torch.clamp(maximum_distribution, min=1e-10, max=1 - 1e-10)
    return maximum_distribution


def find_numerindices_sorted_order(number_values, number_token_indices, passage_number_entidxs):
    # These are the values for the numbers for the number-tokens
    number_token_numbervalues = [number_values[x] for x in passage_number_entidxs]

    number_tokenidx_values = list(zip(number_token_indices, number_token_numbervalues))

    sorted_numberidx_value_tuples = sorted(number_tokenidx_values, key = lambda x: x[1])

    sorted_number_indices, _ = zip(*sorted_numberidx_value_tuples)

    return sorted_number_indices


def to_long_tensor(array):
    return torch.LongTensor(array)


passage_length = 10
number_values = [2, 5, 10]
passage_number_indices = [2, 4, 6, 8, 9]
passage_number_entidxs = [1, 2, 0, 2, 1]
passage_number_idx2entidx = [-1 for _ in range(passage_length)]

for passage_num_tokenidx, number_idx in zip(passage_number_indices, passage_number_entidxs):
    passage_number_idx2entidx[passage_num_tokenidx] = number_idx

sorted_number_indices = find_numerindices_sorted_order(number_values, passage_number_indices, passage_number_entidxs)

passage_number_idx2entidx = to_long_tensor(passage_number_idx2entidx)
token2num_mask = (passage_number_idx2entidx > -1).float()
sorted_number_indices = to_long_tensor(sorted_number_indices)

print(sorted_number_indices)

passage_attention = torch.softmax(torch.randn(passage_length), dim=-1)
passage_number_alignment = torch.zeros(passage_length, passage_length)

passage_number_alignment[2, 2] = 1
passage_number_alignment[4, 4] = 1
passage_number_alignment[6, 6] = 1
passage_number_alignment[8, 8] = 1
passage_number_alignment[9, 9] = 1
passage_number_alignment = util.masked_softmax(passage_number_alignment,
                                               mask=token2num_mask.unsqueeze(0), dim=-1)

# passage_number_alignment = util.masked_softmax(torch.randn(passage_length, passage_length),
#                                                mask=token2num_mask.unsqueeze(0), dim=-1)

pattn_times_numbertoken_distribution = passage_attention.unsqueeze(1) * passage_number_alignment

# Shape: (passage_length, num_of_number_tokens) -- These are now in sorted order
pattn_weighted_numbertoken_probs = pattn_times_numbertoken_distribution[:, sorted_number_indices]

# Shape: (num_of_number_tokens, )
expected_numbertoken_probs = pattn_weighted_numbertoken_probs.sum(0)
print(f"ExpectedNumTokenProbs: {expected_numbertoken_probs}")
numbertoken_maxprobs_sorted = max_number_distribution(expected_numbertoken_probs)
print(f"MaxedNumTokenProbs: {numbertoken_maxprobs_sorted}")

# Redistributing passage_numbertoken_max_probs for each row based on the weights in that row
maxprob_times_pattn_numbertokenprob = pattn_weighted_numbertoken_probs * numbertoken_maxprobs_sorted.unsqueeze(0)

total_weight_to_numbertoken = pattn_weighted_numbertoken_probs.sum(0, keepdim=True)

new_pattn = (maxprob_times_pattn_numbertokenprob / total_weight_to_numbertoken).sum(1)

new_pattn_times_number = new_pattn.unsqueeze(1) * passage_number_alignment
new_number = new_pattn_times_number.sum(0)


# print(passage_attention)
# print(passage_number_alignment)
# print(expected_numbertoken_probs)
# print(numbertoken_probs_sorted)
# print(numbertoken_maxprobs_sorted)
# print(passage_numbertoken_max_probs)


print(f"OrigPattn:{passage_attention}")
print(f"NewPattn: {new_pattn}")
print(new_pattn.sum())
print(new_number)




