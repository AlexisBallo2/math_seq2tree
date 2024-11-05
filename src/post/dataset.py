
from collections import Counter
import json
f = open("../../data/DRAW/dolphin_t2_final.json", encoding="utf-8")
data = json.loads(f.read())

lengths = []
for obs in data:
    equations = obs['lSolutions']
    lengths.append(len(equations))

print(Counter(lengths))

