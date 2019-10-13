import torch
import torch.nn.functional as F
from allennlp.nn import util

x = torch.LongTensor([[[1, 2], [0, 1]], [[1, 2], [-1, -1]], [[1, 0], [0, 2]]])
x_mask = (x > -1).long()

d = torch.randn(3)

print(x)
print(d)

x_m = x * x_mask
d_uns = d.unsqueeze(0).unsqueeze(2)
x_uns = x_m.unsqueeze(0)

print(x_m)

selected_d_1 = util.batched_index_select(target=d_uns, indices=x_uns[:, :, :, 0]).squeeze(0).squeeze(-1)
selected_d_2 = util.batched_index_select(target=d_uns, indices=x_uns[:, :, :, 1]).squeeze(0).squeeze(-1)
selected_d_m1 = selected_d_1 * x_mask[:, :, 0].float()
selected_d_m2 = selected_d_2 * x_mask[:, :, 1].float()

print(selected_d_m1)
print(selected_d_m2)

expected_prob = (selected_d_m1 * selected_d_m2).sum(dim=1)
print(expected_prob)
