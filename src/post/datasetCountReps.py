out = []
opps = ["+", "-", "*", "/"]
from collections import Counter
with open("src/post/datasetEquations.txt", "r") as f:
    data = f.read().split("\n")
    for line in data:
        lineSetNoOps = [i for i in list(line.split(" ")) if i not in opps]
        lineSetCount = Counter(lineSetNoOps)
        print(lineSetCount)
        maxReps = max(lineSetCount.values())
        print('maxReps', maxReps)
        if maxReps > 1:
            out.append(1)
        else:
            out.append(0)

repeatsPercent = sum(out) / len(out)
print(repeatsPercent)