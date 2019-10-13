filepath = "/srv/local/data/nitishg/semqa/predictions/hotpotqa_bool_wosame/hotpotqa_parser/BS_2/OPT_adam/LR_0.001/Drop_0.2/BeamSize_32/MaxDecodeStep_12_onlyAND/predictions.txt"

correct, total = 0, 0

with open(filepath, "r") as f:
    lines = f.readlines()
    for line in lines:
        ans, d = line.strip().split(" ")
        d_str = "yes" if (float(d) > 0.5) else "no"

        print(f"ans:{ans} d_str:{d_str} d:{d}")

        if ans == d_str:
            correct += 1
        else:
            correct += 0
        total += 1

acc = 100 * float(correct) / float(total)

print(f"Correct: {correct}  Total: {total}. Accuracy: {acc}")
