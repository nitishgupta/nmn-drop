import torch
import torch.nn.functional as F
from allennlp.nn import util
from torch.nn import GRU
from allennlp.modules import Seq2SeqEncoder
from allennlp.common import Params
from semqa.profiler.profile import Profile

scalingvals = [1.0, 2.0, 5.0, 10.0]


def gru_func(max_length, seq_length, gru_net):
    x = torch.randn(max_length).cuda()
    mask = torch.FloatTensor([1.0]*seq_length + [0.0]*(max_length-seq_length)).cuda()
    scaled_x = [x * sf for sf in scalingvals]
    # [length, size]
    scaled_x = torch.stack(scaled_x, dim=1)
    # [1, length, direction*output]
    batched_output, _ = gru_net(scaled_x.unsqueeze(1))
    # [length, direction*output]
    output = batched_output.squeeze(1)
    masked_output = output * mask.unsqueeze(1)
    return masked_output



def s2s_func(max_length, seq_length, gru_net):
    x = torch.randn(max_length).cuda()
    mask = torch.FloatTensor([1.0]*seq_length + [0.0]*(max_length-seq_length)).cuda()

    scaled_x = [x * sf for sf in scalingvals]
    # [length, size]
    scaled_x = torch.stack(scaled_x, dim=1)
    # [1, length, direction*output]
    batched_output = gru_net(scaled_x.unsqueeze(0), mask.unsqueeze(0))
    # [length, direction*output]
    output = batched_output.squeeze(0)
    masked_output = output * mask.unsqueeze(1)

    return masked_output


def gru_test(nsteps):
    gru_net = GRU(input_size=4, hidden_size=20, num_layers=3, batch_first=False, dropout=0.0, bidirectional=True)
    gru_net.cuda(device=0)

    with Profile("gru"):
        for i in range(nsteps):
            gru_func(max_length=450, seq_length=350, gru_net=gru_net)
    print(Profile.to_string())


def s2s_test(nsteps):
    params = Params(params={"type": "gru",
                            "input_size": 4,
                            "hidden_size": 20,
                            "num_layers": 3,
                            "bidirectional": True})
    gru_net = Seq2SeqEncoder.from_params(params=params)
    gru_net.cuda(device=0)

    with Profile("s2s"):
        for i in range(nsteps):
            s2s_func(max_length=450, seq_length=350, gru_net=gru_net)
    print(Profile.to_string())


if __name__=='__main__':
    nsteps = 25

    gru_test(nsteps)
    # s2s_test(nsteps)

