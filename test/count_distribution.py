import torch
from torch.autograd import Variable

lr = 0.1

countvals = Variable(torch.FloatTensor(range(0, 10)))


def forward(mean, answer):
    l2_by_vsquared = torch.pow(countvals - mean, 2) / (2 * variance * variance)
    # print(l2_by_vsquared)

    exp_val = torch.exp(-1 * l2_by_vsquared) + 1e-30
    # print(exp_val)

    distribution = exp_val / (torch.sum(exp_val))
    print(distribution)

    loss = -1 * torch.log(distribution[answer])
    loss.backward(retain_graph=True)
    grad = mean.grad
    print(f"Grad: {grad}")
    with torch.no_grad():
        # mean.data = mean.data + -1 * grad * lr
        mean -= mean.grad * lr
        mean.grad.zero_()


# mean = Variable(torch.tensor(20.0))
mean = torch.tensor(15.0, requires_grad=True)
# mean.requires_grad = True
# variance = abs(10 - mean.detach())/10
if mean.detach() > 10:
    variance = mean.detach() - 10
else:
    variance = 0.5

for i in range(0, 100):
    print(f"Mean: {mean}")
    forward(mean, 5)
