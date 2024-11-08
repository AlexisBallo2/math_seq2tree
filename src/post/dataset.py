
from collections import Counter
import json
import matplotlib.pyplot as plt
import numpy as np

# df = open("../../data/DRAW/draw.json", encoding="utf-8")
df = open("../../data/DRAW/dolphin_t2_final.json", encoding="utf-8")
mf = open("../../data/Math_23K.json", encoding="utf-8")
js = ""
draw = []
math = []
# for each line
for i, s in enumerate(mf):
    js += s
    i += 1
    if i % 7 == 0:  # every 7 line is a json
        data_d = json.loads(js)
        if "千米/小时" in data_d["equation"]:
            data_d["equation"] = data_d["equation"][:-5]
        math.append(data_d)
        js = ""
draw = json.loads(df.read())

# print(len(draw))
# lengths = []
# for obs in draw:
#     equations = obs['lSolutions']
#     lengths.append(len(equations))

# print(Counter(lengths))

draw_lengths = []
for obs in draw:
    equations = "".join([i if i!= " " else "" for i in "".join(obs['lEquations'])])
    draw_lengths.append(round(len(equations)/len(obs['lEquations'])))

math_lengths = []
for obs in math:
    equations = "".join(obs['equation'])
    # print(equations[0])
    math_lengths.append(round(len(equations)))

# math_dict = Counter(math_lengths)
# total = sum(list(math_dict.values()))
# updated = {}
# for k, v in math_dict.items():
#     updated[k] = v/total

# print(updated)
print(draw_lengths)
bins = np.linspace(0, 50, 30)
plt.hist(x=math_lengths, bins=bins, edgecolor='black', density=True, label="MATH23K", alpha = 0.5)
plt.hist(x=draw_lengths, bins=bins,edgecolor='black', density=True, label="DRAW-1K", alpha=0.5)
plt.legend(loc='upper right') 
filename = 'dataset-equation-lengths.pdf'
plt.title('Equation Lengths')
plt.ylabel('Percent of Dataset')
plt.xlabel('Number of Tokens')
# plt.show()
plt.savefig(filename)




