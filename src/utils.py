
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


def process_loss_dicts(train, eval, title = "Losses"):
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
            vals = [item[key] for item in epoch if item[key] != -1]
            if len(vals) == 0:
                eval_each.append(0)
                continue
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
    make_general_graph(final_dict, title)


def make_general_graph(dict, title = "Losses"):
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
    fig.suptitle(title)
    plt.figlegend(['Train', "Eval"], loc='upper left')
    # plt.title("Losses")
    plt.savefig(f"src/post/loss-{time.time()}-{0}.png")
    plt.show()


# string = """
# {"total_loss": [[5.482218927807278, 4.590908553865221, 4.239986578623454, 3.8153675397237143, 3.4805677466922336, 3.02931104765998, 3.132846567365858, 2.9446283976236978, 2.727083577050103, 2.5597587956322565, 2.4054687552981906, 2.448180900679694, 2.307718555132548, 2.3737647268507214, 2.076695329613156, 2.028747664557563, 1.9564105934566922, 1.8201553291744657, 1.7825875282287598, 1.5785731342103746], [2.9548415246048596, 2.208081321987679, 2.1475584516680337, 2.049432476603888, 2.0672965090933855, 1.975492755329706, 2.030097089404982, 1.9669811805573905, 1.9030184791824682, 1.931962733830863, 1.8975737669119022, 2.097460259993871, 2.028773166784426, 1.9351976042598245, 1.9737695997081153, 1.8814610708050612, 1.8936756268991688, 1.8754453190216205, 1.852167987484273, 1.865128419627019]], "equation_loss": [[2.426046782069736, 1.7295958863364325, 1.6144261360168457, 1.2885032892227173, 1.1618073185284932, 1.0414696799384222, 0.9771636128425598, 0.9923672444290585, 0.8722271157635583, 0.8045124146673415, 0.6608731531434588, 0.797020317779647, 0.593984540965822, 0.6129692412085004, 0.7149435447321998, 0.5765993379884295, 0.5534458176957237, 0.3819243725803163, 0.46694715983337826, 0.4498171897398101], [2.2239924523888566, 1.703965672632543, 1.6355887292846432, 1.5826357449215602, 1.5874941762143033, 1.5112209637475207, 1.4995693319696721, 1.4293599709626135, 1.4117250396468775, 1.3485205862822571, 1.339120736391079, 1.4265897549628241, 1.4941662698681277, 1.4017992572086613, 1.3327333106984938, 1.2606557246025016, 1.2760254910018871, 1.2619256641322034, 1.2252408517751752, 1.2861564922623518]], "classify_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "sni_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "semantic_alignment_loss": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "equ_1_acc": [[0.2676218370958859, 0.41413320841366796, 0.42280766115382434, 0.4403920389403646, 0.4368879837473052, 0.485725303332866, 0.5046754584597052, 0.5140404596419121, 0.5472625330846304, 0.5596594408913061, 0.5783401525488903, 0.6250087816110921, 0.6112004954554279, 0.6164750374515171, 0.6593686296566942, 0.6446261828862134, 0.6608238567894524, 0.6610168593254623, 0.6778195132349187, 0.6734694760022694], [0.3165892558313103, 0.444956715943345, 0.44211657705657154, 0.4281829570407938, 0.4443843767585286, 0.45685877440558265, 0.4639341942883681, 0.5120763339528935, 0.512245332882599, 0.5398598379343261, 0.5540671534791709, 0.5265280200823876, 0.5630511957843389, 0.555577758774084, 0.5466954263525092, 0.5745292590570883, 0.562002385459535, 0.5701369520098264, 0.5818473246970807, 0.5811520731367842]], "equ_2_acc": [[0.27090967661845133, 0.4811861421437828, 0.4916130428933097, 0.5178366521103261, 0.5483397897465647, 0.6018808381247762, 0.5890372467652732, 0.5990264532873374, 0.6876702479512156, 0.6508467418122968, 0.6615017176024637, 0.648683679357338, 0.6936763625922282, 0.6986568356882987, 0.6438644846396784, 0.735899827593115, 0.7237451647794303, 0.753608860280447, 0.7844312089338795, 0.8118994964800784], [0.11001320269612937, 0.20865818914599404, 0.18485859217566536, 0.2189890506963677, 0.22435748533309502, 0.22992544943764456, 0.2278289009996327, 0.22890000694878734, 0.23646227304763887, 0.2355976453537429, 0.22029542273444697, 0.2387851534192998, 0.24518895738407934, 0.22269871050358858, 0.23125465320587263, 0.24320160417721387, 0.2535543047738169, 0.256761666517764, 0.25698104966397645, 0.26334613895589504]], "equ_3_acc": [[0.07671957671957672, 0.3162180814354727, 0.2652777777777778, 0.40575396825396826, 0.3340643274853801, 0.2613417601975955, 0.4695767195767196, 0.35555555555555557, 0.35694444444444445, 0.45348516218081436, 0.45462962962962966, 0.4221560846560847, 0.4722222222222222, 0.4576545530492899, 0.4437830687830688, 0.48872785829307563, 0.5551587301587301, 0.6027777777777777, 0.4693581780538302, 0.41375661375661377], [0.009639953542392566, 0.015214866434378629, 0.011149825783972125, 0.011730545876887339, 0.013937282229965157, 0.012311265969802554, 0.016608594657375145, 0.019628339140534263, 0.01742160278745645, 0.01951219512195122, 0.020905923344947737, 0.0124274099883856, 0.017421602787456445, 0.019744483159117303, 0.019512195121951223, 0.02032520325203252, 0.02032520325203252, 0.021370499419279907, 0.021138211382113817, 0.021370499419279907]]}
# """

# loaded = json.loads(string)

# make_general_graph(loaded)

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

getUniqueEquationCounts()