out = []
opps = ["+", "-", "*", "/"]
vars = ['X', 'Y', 'Z']
import json
from collections import Counter

# counting how many equations have repeated variables

# with open("src/post/datasetEquations.txt", "r") as f:
#     data = f.read().split("\n")
#     for line in data:
#         # lineSetNoOps = [i for i in list(line.split(" ")) if i not in opps]
#         lineSetVars = [i for i in list(line.split(" ")) if i in vars]
#         # lineSetCount = Counter(lineSetNoOps)
#         lineVarsCount = Counter(lineSetVars)
#         # print(lineSetCount)
#         print('line', line)
#         print('vars', lineVarsCount)
#         counts = list(lineVarsCount.values())
#         print('counts', counts)
#         # maxReps = max(lineSetCount.values())
#         # print('maxReps', maxReps)
#         if len(counts) == 0:
#             out.append(0)
#         else:
#             maxVarReps = max(counts, default=0)
#             print('maxVarReps', maxVarReps)
#             if maxVarReps > 1:
#                 out.append(1)
#             else:
#                 out.append(0)

# repeatsPercent = sum(out) / len(out)a
# print(repeatsPercent)

# counting number of equations 

# f = open('data/DRAW/dolphin_t2_final.json', encoding="utf-8")
# data = json.loads(f.read())
# lens = []
# for one in data:
#     lens.append(len(one['lEquations']))

# total = Counter(lens)
# print(total)

# counting baseline accuracy (just predicting + token)
# count + tokens
with open("src/post/datasetEquations.txt", "r") as f:
    data = f.read().split("\n")
    total_tokens = 0
    total_plus = 0
    for line in data:
        splitted = line.split(" ")
        for token in splitted:
            total_tokens += 1
            if token == "+":
                total_plus += 1
    print(total_plus, total_tokens)
    print(total_plus / total_tokens)
