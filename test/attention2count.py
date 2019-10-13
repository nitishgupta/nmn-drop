from typing import List
import torch
import allennlp.nn.util as util
import random

random.seed(90)


def number2count_auxloss(passage_number_values: List[List[float]], device_id=-1):
    """ Using passage numnbers, make a (batch_size, max_passage_numbers) (padded) tensor, each containing a
        noisy distribution with mass distributed over x-numbers. The corresponding count-answer will be x.
        Use the attention2count rnn to predict a count value and compute the loss.
    """
    batch_size = len(passage_number_values)
    # List of length -- batch-size
    num_of_passage_numbers = [len(nums) for nums in passage_number_values]
    max_passage_numbers = max(num_of_passage_numbers)

    # Shape: (batch_size, )
    num_pasasge_numbers = util.move_to_device(torch.LongTensor(num_of_passage_numbers), cuda_device=device_id)
    # Shape: (max_passage_numbers, )
    range_vector = util.get_range_vector(size=max_passage_numbers, device=device_id)

    mask = (range_vector.unsqueeze(0) < num_pasasge_numbers.unsqueeze(1)).float()
    print(mask)

    number_distributions = mask.new_zeros(batch_size, max_passage_numbers).normal_(0, 0.01).abs_()
    count_answers = number_distributions.new_zeros(batch_size, max_passage_numbers).long()

    for i, num_numbers in enumerate(num_of_passage_numbers):
        """ Sample a count value between [0, min(5, num_numbers)]. Sample indices in this range, and set them as 1.
            Add gaussian noise to the whole tensor and normalize. 
        """
        # Pick a count answer
        count_value = random.randint(0, min(7, num_numbers))
        count_answers[i, count_value] = 1
        # Pick the indices that will have mass
        if count_value > 0:
            indices = random.sample(range(num_numbers), count_value)
            # Add 1.0 to all sampled indices
            number_distributions[i, indices] += 1.0

    number_distributions = number_distributions * mask
    number_distributions = number_distributions / torch.sum(number_distributions, dim=1).unsqueeze(1)


if __name__ == "__main__":
    passage_number_values = [[1, 3, 5], [1, 16, 20, 24, 29], [1, 16], [1, 16, 18, 91]]

    number2count_auxloss(passage_number_values, device_id=0)
