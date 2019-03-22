import json
import torch

dates = [4,6,5]
numd = len(dates)

probs1 = torch.FloatTensor([0.333, 0.333, 0.333])
probs2 = torch.FloatTensor([0.333, 0.333, 0.333])

mat = [[0 for _ in range(numd)] for _ in range(numd)]

for i in range(numd):
    for j in range(numd):
        mat[i][j] = 1 if dates[i] > dates[j] else 0

mat = torch.FloatTensor(mat)

joint_prob = torch.matmul(probs1.unsqueeze(1), probs2.unsqueeze(0))

ut = torch.triu(torch.ones(numd, numd))

print(f"Joint Prob:\n{joint_prob}")


print(f"GT mat:\n{mat}")

print(f"Dates: {dates}")
print(f"P1: {probs1}  P2: {probs2}")

greater = (mat * joint_prob).sum()
print(f"w/o UT: {greater}")

greater = (mat * joint_prob * ut).sum()
print(f"w/ UT: {greater}")


