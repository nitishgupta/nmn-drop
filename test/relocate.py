import torch
import allennlp
from allennlp.modules.matrix_attention import (MatrixAttention, BilinearMatrixAttention,
                                               DotProductMatrixAttention, LinearMatrixAttention)


qvec = torch.randn(5)
pvec = torch.randn(10, 5)



relocatema = LinearMatrixAttention(tensor_1_dim=5,
                                   tensor_2_dim=5,
                                   combination="x,y,x*y")

p_in = qvec.unsqueeze(0) + pvec

#torch.cat([qvec.unsqueeze(0), pvec], dim=1)

sim = relocatema(p_in.unsqueeze(0), pvec.unsqueeze(0)).squeeze(0)

print(p_in.size())
print(sim.size())
