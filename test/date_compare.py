from typing import List
import json
import torch
import allennlp.nn.util as allenutil

from semqa.domain_languages.drop.drop_language import Date, DropLanguage


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



dates = [
            Date(year=1917, month=11, day=-1),
            Date(year=1918, month=1, day=18),
            Date(year=1918, month=1, day=-1),
            Date(year=1918, month=2, day=9)
        ]


date_distribution_1 = [0.25, 0.25, 0.25, 0.25] # [0.33333, 0.333333, 0.333333]
date_distribution_2 = [0.25, 0.25, 0.25, 0.25] # [0.33333, 0.333333, 0.333333]

date_gt_mat, date_lt_mat = compute_date_greater_than_matrix(dates, -1)

# for i in range(len(date_gt_mat)):
#     date_gt_mat[i,i] = 0.5

probs1 = torch.FloatTensor(date_distribution_1)
probs2 = torch.FloatTensor(date_distribution_2)

joint_prob = torch.matmul(probs1.unsqueeze(1), probs2.unsqueeze(0))

bool_greater = (date_gt_mat * joint_prob).sum()
bool_lesser = (date_lt_mat * joint_prob).sum()

print(f"{[str(d) for d in dates]}\n")

print(f"Date Greater than:\n{date_gt_mat}\n")
print(f"Date Lesser than:\n{date_lt_mat}\n")

print(f"Date 1: {date_distribution_1}")
print(f"Date 2: {date_distribution_2}")

print(f"p(D1 > D2): {bool_greater}")
print(f"p(D1 < D2): {bool_lesser}")

