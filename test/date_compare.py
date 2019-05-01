from typing import List
import json
import torch
import allennlp.nn.util as allenutil

from semqa.domain_languages.drop_old.drop_language import Date, DropLanguage


def compute_date_greater_than_matrix(date_values: List[Date], device_id: int):
    date_greater_than_mat = [[0 for _ in range(len(date_values))] for _ in range(len(date_values))]
    date_lesser_than_mat = [[0 for _ in range(len(date_values))] for _ in range(len(date_values))]
    # self.encoded_passage.new_zeros(self.num_passage_dates, self.num_passage_dates)
    for i in range(len(date_values)):
        for j in range(len(date_values)):
            date_greater_than_mat[i][j] = 1.0 if date_values[i] > date_values[j] else 0.0
            date_lesser_than_mat[i][j] = 1.0 if date_values[i] < date_values[j] else 0.0
            # date_greater_than_mat[j][i] = 1.0 - date_greater_than_mat[i][j]
    date_greater_than_mat = allenutil.move_to_device(torch.FloatTensor(date_greater_than_mat), device_id)
    date_lesser_than_mat = allenutil.move_to_device(torch.FloatTensor(date_lesser_than_mat), device_id)

    return date_greater_than_mat, date_lesser_than_mat


def dt_greater(dt1, dt2, date_gt_mat):
    joint_prob = torch.matmul(dt1.unsqueeze(1), dt2.unsqueeze(0))
    bool_greater = (date_gt_mat * joint_prob).sum()
    return bool_greater

def dt_lesser(dt1, dt2, date_lt_mat):
    joint_prob = torch.matmul(dt1.unsqueeze(1), dt2.unsqueeze(0))
    bool_lesser = (date_lt_mat * joint_prob).sum()
    return bool_lesser


dates = [
            Date(year=1917, month=11, day=-1),
            Date(year=1918, month=1, day=18),
            Date(year=1918, month=1, day=-1),
            Date(year=1918, month=2, day=9)
        ]


date_distribution_1 = [0.5, 0.0, 0.5, 0.0] # [0.33333, 0.333333, 0.333333]
date_distribution_2 = [1.0, 0.0, 0.0, 0.0] # [0.33333, 0.333333, 0.333333]
date_distribution_3 = [0.5, 0.0, 0.5, 0.0] # [0.33333, 0.333333, 0.333333]

date_gt_mat, date_lt_mat = compute_date_greater_than_matrix(dates, -1)

# for i in range(len(date_gt_mat)):
#     date_gt_mat[i,i] = 0.5

probs1 = torch.FloatTensor(date_distribution_1)
probs2 = torch.FloatTensor(date_distribution_2)
probs3 = torch.FloatTensor(date_distribution_3)
probs1.requires_grad = True
probs2.requires_grad = True
probs3.requires_grad = True

gt_1_2 = dt_greater(probs1, probs2, date_gt_mat)
lt_1_2 = dt_lesser(probs1, probs2, date_lt_mat)
lt_2_1 = dt_lesser(probs2, probs1, date_lt_mat)
lt_1_3 = dt_lesser(probs1, probs3, date_lt_mat)
lt_3_1 = dt_lesser(probs3, probs1, date_lt_mat)


print(f"{[str(d) for d in dates]}\n")

print(f"Date Greater than:\n{date_gt_mat}\n")
print(f"Date Lesser than:\n{date_lt_mat}\n")

print(f"Date 1: {date_distribution_1}")
print(f"Date 2: {date_distribution_2}")

print(f"p(D1 < D2): {lt_1_2}")
print(f"p(D2 < D1): {lt_2_1}")
print(f"p(D1 < D3): {lt_1_3}")
print(f"p(D3 < D1): {lt_3_1}")

# lt_3_1.backward()
#
# print(probs1.grad)
# print(probs3.grad)
#
# probs1 = probs1 + probs1.grad
# probs3 = probs3 + probs3.grad
#
# print(probs1)
# print(probs3)

