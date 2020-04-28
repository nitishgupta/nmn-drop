import numpy as np

strong_file = "interpret_drop/strong.txt"
weak_file = "interpret_drop/weak.txt"

def read_scores(scores_file):
    scores = []
    with open(scores_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            scores.append(float(line.strip()))

    print(f"Scored read from {scores_file}. Size: {len(scores)}")
    print(f"Mean: {np.mean(scores)}")
    return scores

strong_scores = read_scores(strong_file)
weak_scores = read_scores(weak_file)

both_scores = [weak_scores, strong_scores]

original_statistic = np.mean(both_scores[0])-np.mean(both_scores[1])
print(f"Original mean (statistic): {original_statistic}")

sample_size = 10000
signs = np.random.binomial(1, 0.5, size=(sample_size, len(strong_scores)))
num_exceeding = 0
for trial in range(signs.shape[0]):
    precisions = [[], []]
    for i in range(2):
        for j in range(len(both_scores[i])):
            if signs[trial][j] > 0.5:
                precisions[i].append(weak_scores[j])
            else:
                precisions[1-i].append(strong_scores[j])
    assert len(precisions[0]) == len(precisions[1])
    statistic = np.mean(precisions[0])-np.mean(precisions[1])
    if abs(statistic) >= original_statistic:
        num_exceeding += 1

print('p-value: '+str(float(num_exceeding)/signs.shape[0]))