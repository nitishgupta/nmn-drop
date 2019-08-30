import torch
import torch.nn.functional as F

a = torch.cuda.FloatTensor([0.90, 0.0, 1.0])
b = torch.cuda.FloatTensor([1.0, 0.0, 1.0])

for i in range(0, 100):
    loss = F.binary_cross_entropy(a, b)




