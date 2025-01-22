
import sympy as sp
from sympy.solvers import solve
import json 
from collections import Counter
import matplotlib.pyplot as plt
import time
import re
import os
import torch
import pathlib
import shutil
from copy import deepcopy


def solve_equation(equations, solutions):
    # convert prefix to infix



    try:
        spEqs = []
        for equ in equations:
            temp = "Eq(" + equ.replace("=", ",") + ")"
            sympy_eq = sp.simplify(temp)
            spEqs.append(sympy_eq)   
        solved = solve(spEqs, dict=True)
        # cur_targets = [round(i) for i in list(solved[0].values())]
        act_solns = [round(i) for i in (list(solved[0].values()))]

        print("act", act_solns)
        print("pred", solutions)
        if Counter(act_solns) == Counter(solutions):
            return True
        else:
            return False
    except:
        return False

def replace_nums(mapping, equation, nums, num_stack):
    final_equation = []
    for token in equation:
        if mapping.get(token, "") != "":
            final_equation.append(mapping[token])
        # pop from num stack
        elif token[0] == "N":
            pos_list = num_stack.pop()
            c = nums[pos_list[0]]
            final_equation.append(c)
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

    total = last_train + [i[0] for i in last_eval]
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
    for k,v in counter_dict.items():
        print(k, v)
    # print(counter_dict)
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

# get_baseline()


def process_loss_dicts(train, eval, title = "Losses"):
    keys = list(train[0][0].keys())
    train_vals = {}
    eval_vals = {}
    for key in keys:
        train_each = []
        for epoch in train:
            vals = [item[key] for item in epoch]
            if key == "acc_solutions_lengths":
                length_values = vals
                correct_ones = [item['acc_solutions_plain'] for item in epoch]
                one_acc = list_to_counts(length_values, correct_ones, 1 )
                two_acc = list_to_counts(length_values, correct_ones, 2 )
                three_acc = list_to_counts(length_values, correct_ones, 3 )
                train_each.append([one_acc, two_acc, three_acc])
            elif key == "acc_solutions_set":
                set_values = vals
                correct_ones = [item['acc_solutions_plain'] for item in epoch]
                draw = set_to_counts(set_values, correct_ones, "draw" )
                alg514 = set_to_counts(set_values, correct_ones, 'alg514' )
                mwaps = set_to_counts(set_values, correct_ones, 'mawps' )
                train_each.append([draw, alg514, mwaps])
            elif key == "acc_solutions_plain":
                continue
            else:
                avg = sum(vals)/len(vals)
                train_each.append(avg)

        train_vals[key] = train_each 

        eval_each = []
        for epoch in eval:
            vals = [item[key] for item in epoch if item[key] != -1]
            if len(vals) == 0:
                eval_each.append(0)
                continue
            if key == "acc_solutions_lengths":
                length_values = vals
                correct_ones = [item['acc_solutions_plain'] for item in epoch]
                one_acc = list_to_counts(length_values, correct_ones, 1 )
                two_acc = list_to_counts(length_values, correct_ones, 2 )
                three_acc = list_to_counts(length_values, correct_ones, 3 )
                eval_each.append([one_acc, two_acc, three_acc])
            elif key == "acc_solutions_set":
                set_values = vals
                correct_ones = [item['acc_solutions_plain'] for item in epoch]
                draw = set_to_counts(set_values, correct_ones, 'draw' )
                alg514 = set_to_counts(set_values, correct_ones, 'alg514' )
                mwaps = set_to_counts(set_values, correct_ones, 'mawps' )
                train_each.append([draw, alg514, mwaps])
            elif key == "acc_solutions_plain":
                continue
            else:
                avg = sum(vals)/len(vals)
                eval_each.append(avg)
        eval_vals[key] = eval_each 
    
    final_dict = {}
    for key in keys:
        if key == "acc_solutions_lengths":
            final_dict['sol_acc len 1'] = ([val[0] for val in train_vals[key]], [val[0] for val in eval_vals[key]])
            final_dict['sol_acc len 2'] = ([val[1] for val in train_vals[key]], [val[1] for val in eval_vals[key]])
            final_dict['sol_acc len 3'] = ([val[2] for val in train_vals[key]], [val[2] for val in eval_vals[key]])
        if key == "acc_solutions_set":
            final_dict['sol draw'] = ([val[0] for val in train_vals[key]], [val[0] for val in eval_vals[key]])
            final_dict['sol alg'] = ([val[1] for val in train_vals[key]], [val[1] for val in eval_vals[key]])
            final_dict['sol mwaps'] = ([val[2] for val in train_vals[key]], [val[2] for val in eval_vals[key]])
        else:
            final_dict[key] = (train_vals[key], eval_vals[key])
            print(key)
            print("train", train_vals[key])
            print("eval", eval_vals[key])
            print("\n")


    print("FINAL", json.dumps(final_dict))
    make_general_graph(final_dict, title)

def set_to_counts(lengths, corrects, goal):
    flattened_lengths = [item for sublist in lengths for item in sublist]
    flattened_corrects = [item for sublist in corrects for item in sublist]
    zipped = list(zip(flattened_lengths, flattened_corrects))
    total_correct = 0
    total = 0
    for length, correct in zipped:
        if length == goal:
            total += 1
            if correct == 1:
                total_correct += 1
    if total == 0:
        return 0
    else:
        return total_correct/total


def list_to_counts(lengths, corrects, goal):
    flattened_lengths = [item for sublist in lengths for item in sublist]
    flattened_corrects = [item for sublist in corrects for item in sublist]
    zipped = list(zip(flattened_lengths, flattened_corrects))
    total_correct = 0
    total = 0
    for length, correct in zipped:
        if length == goal:
            total += 1
            if correct == 1:
                total_correct += 1
    if total == 0:
        return 0
    else:
        return total_correct/total

def make_general_graph(dict, title = "Losses"):
    keys = list(dict.keys())
    half = (len(keys) + 1) //3 
    fig, axs = plt.subplots(3, half)
    # fig.suptitle('Vertically stacked subplots')
    i = 0
    j = 0
    for _, key in enumerate(keys):
        train, eval = dict[key]
        axs[i,j].plot(train, label="Train")
        axs[i,j].plot(eval, label="Eval")
        axs[i,j].set_title(key)
        if j == half - 1:
            i += 1
            j = 0
        else:
            j += 1





        # axs[i].title(key)
        # axs[i].legend()
    # plt.legend()
    fig.suptitle(title)
    plt.figlegend(['Train', "Eval"], loc='upper left')
    # plt.title("Losses")
    plt.savefig(f"src/post/loss-{time.time()}-{0}.png")
    plt.show()
    plt.clf()


def getUniqueEquationCounts():
    with open("src/post/datasetEquations.txt", "r") as f:
        data = f.readlines()
        all_data = []
        for line in data:
            line = line.strip("\n")
            all_data.append(line)
        # print(all_data)
    occs = Counter(all_data)
    values = list(occs.values())
    total = sum(values)
    print(total)
    print(occs.values())
    # print(occs)

# getUniqueEquationCounts()


def read_draw_alignment(observation):
    templates = observation['Template']
    alignment = observation['Alignment']
    mapping = {}
    for i, align in enumerate(alignment):
        item = align['coeff']
        value = align['Value']
        mapping[item] = value
    finals = []
    for template in templates:
        final_single = ""
        for token in template:
            if token in mapping:
                final_single += str(mapping[token])
            elif token == " ":
                continue
            else:
                final_single += token
        finals.append(final_single)
    return finals
        

# def read_draw_alignment(observation):
#     vars = ['m', 'n', 'o', 'p', 'q', 'r']
#     mapVars = list(observation['answers'][0].keys())
#     mapVarDict = {}
#     for i, var in enumerate('lSolutions'):
#         mapVarDict[var] = vars[i]
#     templates = observation['Template']
#     # templates = [i.replace("%", "") for i in templates]
#     alignment = observation['numbers']
#     mapping = {}
#     for i, align in enumerate(alignment):
#         item = align['key']
#         value = align['value']
#         mapping[item] = value
#     finals = []
#     for template in templates:
#         final_single = ""
#         for token in template.split(" "):
#             if token in mapping:
#                 final_single += str(mapping[token])
#             elif token in mapVarDict:
#                 final_single += mapVarDict[token]
#             elif token == " ":
#                 continue
#             else:
#                 final_single += token
#         finals.append(final_single)
#     return finals

def read_pen_alignment(observation):
    vars = ['m', 'n', 'o', 'p', 'q', 'r']
    mapVars = list(observation['answers'][0].keys())
    mapVarDict = {}
    for i, var in enumerate(mapVars):
        mapVarDict[var] = vars[i]
    templates = observation['equations']
    # templates = [i.replace("%", "") for i in templates]
    alignment = observation['numbers']
    mapping = {}
    for i, align in enumerate(alignment):
        item = align['key']
        value = align['value']
        mapping[item] = value
    finals = []
    for template in templates:
        final_single = ""
        for token in template.split(" "):
            if token in mapping:
                final_single += str(mapping[token])
            elif token in mapVarDict:
                final_single += mapVarDict[token]
            elif token == " ":
                continue
            else:
                final_single += token
        finals.append(final_single)
    return finals


# def read_fold(fold):
#     keys = list(fold[0].keys())
#     dicts = {}
#     for obs in fold:
#         for key in keys:
#             if key not in dicts:
#                 dicts[key] = []
#             dicts[key].append(obs[key])
#     for k, v in dicts.items():
#         print(k, v)
#         print()



# def read_loss_dicts():
#     matches = r"eval_loss_dict"
#     with open("/Users/home/Downloads/hello_world-205.out", "r") as f:
#         raw = f.readlines()
#         # print(raw)
#     for line in raw:
#         matched = re.search(matches, line)
#         if matched:
#             # print(line[0:100])
#             subbed = re.sub(matches, "", line).strip()
#             subbed = subbed.replace("'", '"')
#             evaled = json.loads(subbed)
#             for one in evaled:
#                 read_fold(one)
#             return 
#         # print("\n")


# read_loss_dicts()

def save_general_state(path, state_dict ):
    pathlib.Path(f"{path}").mkdir(exist_ok=True)
    full_path = f"{path}/general"
    pathlib.Path(f"{full_path}").mkdir(exist_ok=True)
    for key, value in state_dict.items():
        with open(f"{full_path}/{key}.json", "w") as f:
            f.write(json.dumps(value))

def save_fold_state(path, state_dict):
    full_path = f"{path}/fold-{state_dict['fold']}"
    if os.path.exists(f"{path}/fold-{state_dict['fold'] - 1}"):
        shutil.rmtree(f"{path}/fold-{state_dict['fold'] - 1}")

    pathlib.Path(f"{full_path}").mkdir(exist_ok=True)
    for key, value in state_dict.items():
        if key in ['input_lang', 'output_lang']:
            with open(f"{full_path}/{key}.json", "w") as f:
                f.write(value.toJSON())
        else:
            with open(f"{full_path}/{key}", "w") as f:
                f.write(json.dumps(value))


def save_epoch_state(path, state_dict):
    full_path = f"{path}/epoch-{state_dict['epoch']}"
    if os.path.exists(f"{path}/epoch-{state_dict['epoch'] - 5}"):
        shutil.rmtree(f"{path}/epoch-{state_dict['epoch'] - 5}")

    pathlib.Path(f"{full_path}").mkdir(exist_ok=True)
    for key, value in state_dict.items():
        if key in ["models"]:
            with open(f"{path}/{key}.json", "w") as f:
                f.write(json.dumps(list(value.keys())))

            pathlib.Path(f"{full_path}/{key}/").mkdir(exist_ok=True)
            for model in value:
                torch.save(value[model], f"{full_path}/{key}/{model}.pth")
        elif key in ['schedulers']:
            torch.save(value, f"{full_path}/{key}.pth")
        elif key in ['optimizers']:
            torch.save(value, f"{full_path}/{key}.pth")
        else:
            with open(f"{full_path}/{key}.json", "w") as f:
                f.write(json.dumps(value))


# def read_general_state(path):
#     state_dict = {
#         "models": [],
#         "optimizers":  [],
#         "schedulers": [],
#         "config": {},
#         "fold_accuracies": [],
#         "fold": 0,
#         "fold_pairs": [],
#         "pairs": [],
#         "generate_nums": [],
#         "copy_nums": [],
#         "vars": [],
#         "input_lang": {},
#         "output_lang": {},
#         "train_pairs": [],
#         "test_pairs": [],
#         "generate_num_ids": [],
#         "debug": {},
#         "fold_accuracies": [],
#         "train_comparison": [],
#         "eval_comparison": [],
#         "all_train_accuracys": [],
#         "all_train_loss": [],
#         "all_eval_loss": [],
#         "all_eval_accuracys": [],
#         "all_soln_eval_accuracys": [],
#         "total_training_time": [],
#         "total_inference_time": [],
#         "train_time_array": [],
#         "test_time_array": [],
#         "full_start": [],
#     }
#     for file in os.listdir(path):
#         print('file', file)
#         if file == "models":
#             state_dict["models"] = {} 
#             models = json.loads(open(f"{path}/models.json", "r").read())
#             for model in models:
#                 model_act = torch.load(f"{path}/models/{model}.pth")
#                 state_dict["models"][model] = model_act
#         elif file == "optimizers.pth":
#             state_dict["optimizers"] = torch.load(f"{path}/optimizers.pth")
#         elif file == "schedulers.pth":
#             state_dict["schedulers"] = torch.load(f"{path}/schedulers.pth")
#         else:
#             with open(f"{path}/{file}", "r") as f:
#                 state_dict[file.replace(".json", "")] = json.loads(f.read())
#     return state_dict

def read_general_state(path):
    full_path = f"{path}/general"
    state_dict = {}
    for file in os.listdir(full_path):
        with open(f"{full_path}/{file}", "r") as f:
            state_dict[file.replace(".json", "")] = json.loads(f.read())
    return state_dict

def read_fold_state(path):
    # full_path = f"{path}"
    state_dict = {}
    for file in os.listdir(path):
        if 'fold' in file:
            for fold_file in os.listdir(f"{path}/{file}"):
                with open(f"{path}/{file}/{fold_file}", "r") as f:
                    state_dict[fold_file.replace(".json", "")] = json.loads(f.read())
    return state_dict

def read_epoch_state(path):
    # full_path = f"{path}"
    state_dict = {}
    for dir_files in os.listdir(path):
        if 'epoch' in dir_files:
            for file in os.listdir(f"{path}/{dir_files}"):
                if file == "models":
                    state_dict["models"] = {} 
                    models = json.loads(open(f"{path}/models.json", "r").read())
                    for model in models:
                        model_act = torch.load(f"{path}/{dir_files}/models/{model}.pth")
                        state_dict["models"][model] = model_act
                elif file == "optimizers.pth":
                    state_dict["optimizers"] = torch.load(f"{path}/{dir_files}/optimizers.pth")
                elif file == "schedulers.pth":
                    state_dict["schedulers"] = torch.load(f"{path}/{dir_files}/schedulers.pth")
                else:
                    with open(f"{path}/{dir_files}/{file}", "r") as f:
                        state_dict[file.replace(".json", "")] = json.loads(f.read())
    return state_dict




def get_draw_train(pairs, type):

    if type == "dev":
        file = "data/DRAW/draw-train.txt"
    elif type == "test":
        file = "data/DRAW/draw-train.txt"
    else:
        file = "data/DRAW/draw-train.txt"

    sets = []
    with open (file, "r") as f:
        for line in f:
            sets.append(int(line.strip()))


    print(sets)
    final_set = []
    for row in pairs:
        if row['id'] in sets:
            final_set.append(row)
    return final_set



    # for file in os.listdir(path):
    #     print('file', file)
    #     if file == "models":
    #         state_dict["models"] = {} 
    #         models = json.loads(open(f"{path}/models.json", "r").read())
    #         for model in models:
    #             model_act = torch.load(f"{path}/models/{model}.pth")
    #             state_dict["models"][model] = model_act
    #     elif file == "optimizers.pth":
    #         state_dict["optimizers"] = torch.load(f"{path}/optimizers.pth")
    #     elif file == "schedulers.pth":
    #         state_dict["schedulers"] = torch.load(f"{path}/schedulers.pth")
    #     else:
    #         with open(f"{path}/{file}", "r") as f:
    #             state_dict[file.replace(".json", "")] = json.loads(f.read())
    return state_dict



def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, deepcopy(num_stack))
    print('test', 'tar', test, tar)
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res



def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None
