
import sympy as sp
from sympy.solvers import solve

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
