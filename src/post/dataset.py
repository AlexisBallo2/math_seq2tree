
from collections import Counter
import json
f = open("../../data/DRAW/draw.json", encoding="utf-8")
# f = open("../../data/DRAW/dolphin_t2_final.json", encoding="utf-8")
dolphin = json.loads(f.read())

print(len(dolphin))
lengths = []
for obs in dolphin:
    equations = obs['lSolutions']
    lengths.append(len(equations))

print(Counter(lengths))

