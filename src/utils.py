
import sympy as sp
from sympy.solvers import solve
import json 
from collections import Counter
import matplotlib.pyplot as plt
import time

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

# get_baseline()


def process_loss_dicts(train, eval):
    keys = list(train[0][0].keys())
    train_vals = {}
    eval_vals = {}
    for key in keys:
        train_each = []
        for epoch in train:
            vals = [item[key] for item in epoch]
            avg = sum(vals)/len(vals)
            train_each.append(avg)
        train_vals[key] = train_each 

        eval_each = []
        for epoch in eval:
            vals = [item[key] for item in epoch]
            avg = sum(vals)/len(vals)
            eval_each.append(avg)
        eval_vals[key] = eval_each 
    
    final_dict = {}
    for key in keys:
        final_dict[key] = (train_vals[key], eval_vals[key])
        print(key)
        print("train", train_vals[key])
        print("eval", eval_vals[key])
        print("\n")

    print(json.dumps(final_dict))
    make_general_graph(final_dict)


def make_general_graph(dict):
    keys = list(dict.keys())
    half = (len(keys) + 1) // 2
    fig, axs = plt.subplots(2, half)
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
    plt.figlegend(['Train', "Eval"], loc='upper left')
    # plt.title("Losses")
    plt.savefig(f"src/post/loss-{time.time()}-{0}.png")
    plt.show()


# loaded = json.loads('{"total_loss": [[6.058565735816956, 4.905589580535889, 4.389082551002502, 4.08012318611145, 3.892840027809143, 3.6918283700942993, 3.460804581642151, 3.3396753072738647, 3.1192832589149475, 2.9994025826454163, 2.7962230443954468, 2.7709938883781433, 2.647879421710968, 2.5545849204063416, 2.677104949951172, 2.417169988155365, 2.4138357639312744, 2.4528122544288635, 2.3307154178619385, 2.3122368454933167], [4.833349760027899, 4.048999876215838, 4.177802839140961, 4.714296710663947, 4.585973639419113, 4.84451737265656, 4.888840851576432, 5.335695370383885, 5.4389452277750205, 5.333162136699842, 5.562119599701702, 5.737154190091119, 5.3902236223220825, 5.578143031700797, 5.321415061536043, 5.667680436286373, 5.863898662553317, 6.167958060900371, 5.827045941698378, 5.702759594157122]], "equation_loss": [[2.2444160878658295, 1.6352395415306091, 1.4298388957977295, 1.2432770133018494, 1.2174363732337952, 1.060388371348381, 0.9985116273164749, 0.9501462131738663, 0.8280393332242966, 0.7891131788492203, 0.6517368257045746, 0.7117727249860764, 0.641921266913414, 0.6266640350222588, 0.7395714372396469, 0.5576988831162453, 0.5295577049255371, 0.6150279864668846, 0.49877721816301346, 0.48426781594753265], [3.2981793016627217, 2.6516381070233774, 2.6932132762411367, 3.2416054234988447, 3.138806270516437, 3.3656647689100625, 3.4315911894259243, 3.821581394776054, 3.8946504765662593, 3.751215905383013, 3.946627919224725, 4.023671329885289, 3.6715563604797143, 3.881398296010667, 3.6293752020683843, 3.9388263294662256, 4.099723067836485, 4.378066199413245, 4.001895583194235, 3.860287711240243]], "classify_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "sni_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "semantic_alignment_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "equ_1_acc": [[0.17463155477788242, 0.32119065183671924, 0.4182702465328421, 0.4168487620264377, 0.4721109135516681, 0.4752061265119074, 0.556331497236134, 0.5793412829703153, 0.5994287802193198, 0.6641619184342877, 0.7005705094573819, 0.7026823971605525, 0.7446488145792047, 0.7717048396076058, 0.743019929406068, 0.798118264455503, 0.7708514362819624, 0.7742851906158358, 0.7895439118277434, 0.8014296807832044], [0.3409055855926228, 0.41575369157547115, 0.381445864426077, 0.4527246691279539, 0.4553646194270773, 0.4044985683914207, 0.4380748558511378, 0.4177586454366641, 0.3948714032063067, 0.5106489537045465, 0.477699997414629, 0.515043679243437, 0.5673057336565208, 0.5670114739410741, 0.5808333014955708, 0.5868247079742568, 0.5506410340331459, 0.6046188095965991, 0.6005825742237101, 0.622853858325658]], "equ_2_acc": [[0.20163403199117486, 0.4151154401154401, 0.399286543885847, 0.4772295635304356, 0.5427464692170575, 0.5969994058229352, 0.5873487903225807, 0.5655807777656516, 0.6375, 0.6503472741881898, 0.7143367505436471, 0.6439806960273211, 0.6890814116789417, 0.7342660375008894, 0.6673561752512688, 0.7660605697710005, 0.7586926961926961, 0.6950185843568197, 0.7327539100316245, 0.7649430802652889], [0.08571428571428577, 0.17846790890269143, 0.20248447204968947, 0.09275362318840583, 0.09689440993788825, 0.08778467908902696, 0.08571428571428576, 0.08571428571428574, 0.08571428571428577, 0.08571428571428576, 0.08571428571428577, 0.08571428571428576, 0.08571428571428577, 0.08571428571428576, 0.08281573498964807, 0.08571428571428576, 0.08571428571428577, 0.08281573498964807, 0.08281573498964809, 0.08571428571428577]], "equ_3_acc": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}')
# make_general_graph(loaded)