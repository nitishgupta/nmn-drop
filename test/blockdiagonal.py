import torch
import allennlp.nn.util as util

def masking_blockdiagonal(batch_size, passage_length, window, device_id):
    """ Make a (batch_size, passage_length, passage_length) tensor M of 1 and -1 in which for each row x,
        M[:, x, y] = -1 if y < x - window or y > x + window, else it is 1.
        Basically for the x-th row, the [x-win, x+win] columns should be 1, and rest -1
    """

    lower_limit = [max(0, i - window) for i in range(passage_length)]
    upper_limit = [min(passage_length, i + window) for i in range(passage_length)]

    lower = util.move_to_device(torch.LongTensor(lower_limit), cuda_device=device_id)
    upper = util.move_to_device(torch.LongTensor(upper_limit), cuda_device=device_id)

    lower_range_vector = util.get_range_vector(passage_length, device=device_id).unsqueeze(0)
    upper_range_vector = util.get_range_vector(passage_length, device=device_id).unsqueeze(0)

    lower_un = lower.unsqueeze(1)
    upper_un = upper.unsqueeze(1)

    lower_mask = lower_range_vector >= lower_un
    upper_mask = upper_range_vector <= upper_un

    inwindow_mask = (lower_mask == upper_mask).float()
    outwindow_mask = (lower_mask != upper_mask).float()

    print(inwindow_mask)
    print(outwindow_mask)

if __name__ == '__main__':
    masking_blockdiagonal(batch_size=3, passage_length=15, window=3, device_id=0)






