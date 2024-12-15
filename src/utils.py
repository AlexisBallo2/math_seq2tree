
import sympy as sp
from sympy.solvers import solve
import json 
from collections import Counter

def solve_equation(equations, solutions):
    # convert prefix to infix



    try:
        spEqs = []
        for equ in equations:
            sympy_eq = sp.simplify("Eq(" + equ.replace("=", ",") + ")")
            spEqs.append(sympy_eq)   
        solved = solve(spEqs, dict=True)
        cur_targets = [round(i) for i in list(solved[0].values())]
        act_solns = list(round(i) for i in solutions)
        same = 0
        for i, equ in enumerate(cur_targets):
            if equ in act_solns:
                same += 1
        if same != len(cur_targets):
            return False 
        return True
    except:
        return False

def replace_nums(mapping, equation):
    final_equation = []
    for token in equation:
        if mapping.get(token, "") != "":
            final_equation.append(mapping[token])
        else:
            final_equation.append(token)
    return final_equation



def write_comparison(train, eval):
    with open("src/post/train_results.json", "w") as f:
        f.write(json.dumps(train))
    with open("src/post/eval_results.json", "w") as f:
        f.write(json.dumps(eval))


def read_comparison():
    with open("src/post/train_results.json", "r") as f:
        train = json.loads(f.read())
    with open("src/post/eval_results.json", "r") as f:
        eval = json.loads(f.read())

    last_train = train[-1]
    last_eval = eval[-1]

    total = last_train + last_eval
    all_tokens = []
    pairs = []
    for item in total:
        pred_tokens = item['prediction']
        act_tokens = item['actual']
        cur_tokens = set(pred_tokens + act_tokens)
        all_tokens = all_tokens + list(cur_tokens)
        for i, j in zip(act_tokens, pred_tokens):
            pairs.append((i, j))
    all_tokens = list(set(all_tokens))

    counter_dict = {}
    for token in all_tokens:
        current_pairs = [pair for pair in pairs if pair[0] == token]
        counter_dict[token] = Counter( [pair[1] for pair in current_pairs] )
    # print(all_tokens)
    # print(pairs)
    print(counter_dict)
    # return train, eval

# read_comparison()


opperators = ['+', '-', '*', '/']
def get_baseline():
    with open("src/post/datasetEquations.txt", "r") as f:
        data = f.readlines()
        new_data = []
        all_data = []
        for line in data:
            line = line.strip("\n")
            new_data.append(line.split())
            all_data = all_data + line.split()
    print(new_data)
    occs = Counter(all_data)
    ordered_occs = [i[0] for i in occs.most_common()]
    tokens = [i for i in list(set(all_data)) if i not in opperators]
    most_common_op  = [i for i in ordered_occs if i in opperators][0]
    most_common_token = [i for i in ordered_occs if i not in opperators][0]

    print('most common opp', most_common_op)
    print('most common token', most_common_token)

    replacement_dict = {}
    for token in tokens:
        replacement_dict[token] = most_common_token
    for op in opperators:
        replacement_dict[op] = most_common_op

    # non_ops = [i for i in all_data if i not in opperators]

    lengths = 0
    same = 0
    for equation in new_data:
        print("before", equation)
        baseline_equ = equation.copy()
        for i, equ in enumerate(baseline_equ):
            if equ in replacement_dict:
                baseline_equ[i] = replacement_dict[equ]
        print("after", baseline_equ)
        for i, j in zip(equation, baseline_equ):
            lengths += 1
            if i == j:
                same += 1
        
    print(same/lengths)
    #     # print(solve_equation(equation, [0, 1, 2]))

get_baseline()