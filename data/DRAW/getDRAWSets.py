import json

# file = "./draw-dev.txt"
# file = "./draw-test.txt"
file = "./draw-train.txt"
# all_file = "./draw.json"
all_file = "../PEN.json"
# output_file = "./draw-dev.json"
# output_file = "./draw-test.json"
output_file = "./draw-train.json"

sets = []
with open (file, "r") as f:
    for line in f:
        sets.append(int(line.strip()))

with open (all_file, "r") as f:
    all_sets = json.loads(f.read())

print(sets)
final_set = []
for row in all_sets:
    if row['dataset'] == "draw" and row['index'] in sets:
        final_set.append(row)

with open(output_file, 'w') as f:
    f.write(json.dumps(final_set, indent=4))
