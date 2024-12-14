
import sympy as sp
from sympy.solvers import solve
import json 

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