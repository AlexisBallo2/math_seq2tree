out = []
opps = ["+", "-", "*", "/"]
vars = ['X', 'Y', 'Z']
import json
from collections import Counter
import matplotlib.pyplot as plt

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
# f = open('data/PEN.json', encoding="utf-8")
# data = json.loads(f.read())
# lens = {}
# for one in data:
#     if one['dataset'] not in lens:
#         lens[one['dataset']] = [len(one['equations'])]
#     else:
#         lens[one['dataset']].append(len(one['equations']))

# for k,v in lens.items():
#     print(k, Counter(v))

# # calculate percents
# total_counts = sum([(v) for k,v in total.items()])
# print(total_counts)
# for k,v in total.items():
#     print(k, v, v/total_counts)
# # total = sum(lens)
# # print(total)


# counting baseline accuracy (just predicting + token)
# count + tokens
# with open("src/post/datasetEquations.txt", "r") as f:
#     data = f.read().split("\n")
#     total_tokens = 0
#     total_plus = 0
#     for line in data:
#         splitted = line.split(" ")
#         for token in splitted:
#             total_tokens += 1
#             if token == "+":
#                 total_plus += 1
#     print(total_plus, total_tokens)
#     print(total_plus / total_tokens)

# with open("src/post/datasetEquations.txt", "r") as f:
#     data = Counter(f.read())
#     print(data)


with open("math.json", "r") as f:
    math1 = Counter(json.loads(f.read()))
with open("draw.json", "r") as f:
    draw = Counter([i for i in json.loads(f.read()) if i != -1])
#     draw1 = [i for i in json.loads(f.read()) if i != -1]
# with open("draw2.json", "r") as f:
#     draw2 = [i for i in json.loads(f.read()) if i != -1]
# with open("draw3.json", "r") as f:
#     draw3 = [i for i in json.loads(f.read()) if i != -1]

# draw = Counter(draw1 + draw2 + draw3)


labels_math1, values_math1 = zip(*math1.items())
# print(math1)
# print(values_math1)
total_value_math1 = sum(values_math1)
values_math1 = [v/total_value_math1 for v in values_math1]
print(values_math1)
indexes_math1 = [i for i in range(len(labels_math1))]

labels_draw, values_draw = zip(*draw.items())
total_value_draw = sum(values_draw) 
values_draw = [k/total_value_draw for k in values_draw]
indexes_draw = [i for i in range(len(labels_draw))]

width = 1

# Filter the data
threshold = 0.001
filtered_categories2 = [cat for cat, val in zip(labels_draw, values_draw) if val >= threshold ]
filtered_values2 = [val for val in values_draw if val >= threshold]

filtered_categories1 = [cat for cat, val in zip(labels_math1, values_math1) if val >= threshold ]
filtered_values1 = [val for val in values_math1 if val >= threshold]


fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.bar(filtered_categories2, filtered_values2, width, label = "DRAW-1K", color = "C0")
ax1.legend()

ax2.bar(filtered_categories1, filtered_values1, width, label = "MATH23K", color = "C1")
ax2.legend()

fig.supxlabel('Equation Length')
fig.suptitle('Equation Lengths By Equation Number in DRAW-1K and MATH23K')
# plt.show()
# # ax1.bar(indexes_draw1, values_draw1, width, label = "DRAW-1K", color = "orange", alpha=0.5)
# # ax1.bar(indexes_math1, values_math1, width, label = "MATH23K", color = "blue", alpha=0.5)
# # ax1.set_title("First Equation")


# # labels_draw1, values_draw1 = zip(*draw1.items())
# # total_value_draw1 = sum(values_draw1)
# # values_draw1 = [k/total_value_draw1 for k in values_draw1]
# # indexes_draw1 = [i for i in range(len(labels_draw1))]

# # labels_draw2, values_draw2 = zip(*draw2.items())
# # total_value_draw2 = sum(values_draw2)
# # values_draw2 = [k/total_value_draw2 for k in values_draw2]
# # indexes_draw2 = [i for i in range(len(labels_draw2))]


# # labels_draw3, values_draw3 = zip(*draw3.items())
# # total_value_draw3 = sum(values_draw3)
# # values_draw3 = [k/total_value_draw3 for k in values_draw3]
# # indexes_draw3 = [i for i in range(len(labels_draw3))]




# print("math counter", math1)
# # print('draw counter', draw1)

# width = 1


# # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

# # ax1.bar(indexes_draw1, values_draw1, width, label = "DRAW-1K", color = "orange", alpha=0.5)
# # ax1.bar(indexes_math1, values_math1, width, label = "MATH23K", color = "blue", alpha=0.5)
# # ax1.set_title("First Equation")
# # ax1.set_ylabel("Fraction of Observations")

# # ax2.bar(indexes_draw2, values_draw2, width, label = "DRAW-1K", color = "orange", alpha=0.5)
# # ax2.set_title("Second Equation")

# # ax3.bar(indexes_draw3, values_draw3, width, label = "DRAW-1K", color = "orange", alpha=0.5)
# # ax3.set_title("Third Equation")


# # handles, labels = ax1.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='right')


# # for ax in fig.get_axes():
# #     ax.label_outer()

# # plt.xticks(indexes 5, labels)
plt.savefig('equation_lengths.pdf')