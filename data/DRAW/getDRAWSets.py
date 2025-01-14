import json

file = "./draw-dev.txt"
# file = "./draw-test.txt"
all_file = "./draw.json"
output_file = "./draw-dev.json"

sets = []
with open (file, "r") as f:
    for line in f:
        sets.append(int(line.strip()))

with open (all_file, "r") as f:
    all_sets = json.loads(f.read())

print(sets)
final_set = []
for row in all_sets:
    if row['iIndex'] in sets:
        final_set.append(row)

with open(output_file, 'w') as f:
    f.write(json.dumps(final_set, indent=4))
