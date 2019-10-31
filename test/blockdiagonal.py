import torch
import allennlp.nn.util as allenutil


def masking_blockdiagonal(passage_length, window, device_id):
    """ Make a (passage_length, passage_length) tensor M of 1 and -1 in which for each row x,
        M[x, y] = -1 if y < x - window or y > x + window, else it is 1.
        Basically for the x-th row, the [x-win, x+win] columns should be 1, and rest -1
    """

    # The lower and upper limit of token-idx that won't be masked for a given token
    lower = allenutil.get_range_vector(passage_length, device=device_id) - window
    upper = allenutil.get_range_vector(passage_length, device=device_id) + window
    lower = torch.clamp(lower, min=0, max=passage_length - 1)
    upper = torch.clamp(upper, min=0, max=passage_length - 1)
    lower_un = lower.unsqueeze(1)
    upper_un = upper.unsqueeze(1)

    # Range vector for each row
    lower_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)
    upper_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)

    # Masks for lower and upper limits of the mask
    lower_mask = lower_range_vector >= lower_un
    upper_mask = upper_range_vector <= upper_un

    # Final-mask that we require
    inwindow_mask = (lower_mask == upper_mask).float()
    outwindow_mask = (lower_mask != upper_mask).float()

    return inwindow_mask, outwindow_mask


inwin, outwin = masking_blockdiagonal(7, 2, -1)


print(inwin)

print("\n")

print(outwin)





