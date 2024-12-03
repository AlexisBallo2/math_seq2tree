out = []
opps = ["+", "-", "*", "/"]
vars = ['X', 'Y', 'Z']
from collections import Counter
with open("src/post/datasetEquations.txt", "r") as f:
    data = f.read().split("\n")
    for line in data:
        # lineSetNoOps = [i for i in list(line.split(" ")) if i not in opps]
        lineSetVars = [i for i in list(line.split(" ")) if i in vars]
        # lineSetCount = Counter(lineSetNoOps)
        lineVarsCount = Counter(lineSetVars)
        # print(lineSetCount)
        print('line', line)
        print('vars', lineVarsCount)
        counts = list(lineVarsCount.values())
        print('counts', counts)
        # maxReps = max(lineSetCount.values())
        # print('maxReps', maxReps)
        if len(counts) == 0:
            out.append(0)
        else:
            maxVarReps = max(counts, default=0)
            print('maxVarReps', maxVarReps)
            if maxVarReps > 1:
                out.append(1)
            else:
                out.append(0)

repeatsPercent = sum(out) / len(out)
print(repeatsPercent)