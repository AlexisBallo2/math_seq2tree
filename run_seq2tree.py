# coding: utf-8
import os
from src.train_and_evaluate import *
from src.models import *
from src.post.loss_graph import *
from src.utils import *
import time
import torch.optim
from src.expressions_transfer import *
import numpy as np
import sympy as sp
from sympy.solvers import solve



import sys
args = sys.argv
if "-id" in args:
    id_index = args.index("-id")
    run_id = args[id_index + 1]
    print("ID", run_id)
else:
    run_id = "0"

sys.stdout = open('output.txt','wt')


# batch_size = 64
# torch.manual_seed(10)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(10)
# torch.cuda.manual_seed_all(2)
# np.random.seed(10)

# batch_size = 1 
# batch_size = 10
batch_size = 20
# batch_size = 30 
# batch_size = 64 
hidden_size = 512
# n_epochs = 5 
# n_epochs = 10 
n_epochs = 20 
# n_epochs = 40 
# learning_rate = 1e-2 
learning_rate = 1e-3 
# learning_rate = 1e-3 
# learning_rate = 1e-3 
weight_decay = 1e-5
beam_size = 5
n_layers = 2

# num_obs = 20
# num_obs = 50
# num_obs = 100
# num_obs = 200
# num_obs = 600 
# num_obs = 1000 
num_obs = None 

# torch.autograd.set_detect_anomaly(True)

useCustom = True
# useCustom = False 

# setName = "MATH"
setName = "DRAW"

useSubMethod = True
# useSubMethod = False

# useSemanticAlignment = True
useSemanticAlignment = False

# combine all equations into one
# useOneEquation = True
useOneEquation = False

# take vars out of the vocab (control where they go)
useSeperateVars = True
# useSeperateVars = False

# weight the choosing of op vs var vs num
# useOpScaling = True
useOpScaling = False

# decide if we must be able to solve equation
useEquSolutions = True
# useEquSolutions = False 

# use vars as numbers
# do scoring versus do in neural net seperately
useVarsAsNums = True
# useVarsAsNums = False

# useSNIMask = True
useSNIMask = False

# useTFix = True
useTFix = False

useBertEmbeddings = True 

if useBertEmbeddings:
    embedding_size = 768
else:
    embedding_size = 128

# abalations = {
#     "useSubMethod" : True,
#     "useSemanticAlignment" : False,
#     "useOneEquation" : False,
#     "useSeperateVars" : True,
#     "useOpScaling" : False,
#     "useEquSolutions": True,
#     "useVarsAsNums" : True,
#     "useSNIMask": False,
#     "useTFix" : False
# }

title = f"{num_obs} Observations, {n_epochs} Epochs, Dataset = {setName}, Custom = {useCustom} "
config = {
    "batch_size": batch_size,
    "embedding_size": embedding_size,
    "hidden_size": hidden_size,
    "n_epochs": n_epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "beam_size": beam_size,
    "n_layers": n_layers,
    "useCustom": useCustom,
    "setName" : setName,
    "title" : title,
    "useSubMethod": useSubMethod,
    "useSemanticAlignment": useSemanticAlignment,
    "useOneEquation": useOneEquation
}
print("CONFIG \n", config)
os.makedirs("models", exist_ok=True)
if setName == "DRAW":
    data = load_DRAW_data("data/DRAW/dolphin_t2_final.json")
else:
    data = load_raw_data("data/Math_23K.json")
if num_obs:
    data = data[0:num_obs]

# data format:
# {
# "id":"10431",
# "original_text":"The speed of a car is 80 kilometers per hour. It can be written as: how much. Speed ​​* how much = distance.",
# "segmented_text":"The speed of a car is 80 kilometers per hour, which can be written as: how much. speed * how much = distance. ",
# "equation":"x=80",
# "ans":"80"
# }'

pairs, generate_nums, copy_nums, vars = transfer_num(data, setName, useCustom, useEquSolutions, useSubMethod, useSeperateVars)
# pairs.shuffle()
random.shuffle(pairs)
if num_obs:
    pairs = pairs[0:num_obs]
# pairs: list of tuples:
#   input_seq: masked text
#   out_seq: equation with in text numbers replaced with "N#", and other numbers left as is
#   nums: list of numbers in the text
#   num_pos: list of positions of the numbers in the text
# generate_nums: list of common numbers not in input text (ex constants)
# copy_nums:  max length of numbers

temp_pairs = []
for p in pairs:
    # input_seq, prefixed equation, nums, num_pos
    p['equations'] = [from_infix_to_prefix(equ) for equ in p['equations']]
    if useOneEquation:
        equ_with_equals = []
        for equ in p['equations']:
            equ_with_equals += equ
        p['equations'] = [equ_with_equals]
        p['equationTargetVars'] = ["0"]
# pairs = temp_pairs


num_folds = 2
fold_size = int(len(pairs) * 1/num_folds)
fold_pairs = []
for split_fold in range(num_folds - 1):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * (num_folds-1)):])

best_acc_fold = []

all_train_accuracys = []
all_train_loss = []
all_eval_loss = []
all_eval_accuracys = []
all_soln_eval_accuracys = []

total_training_time = 0
total_inference_time = 0

train_time_array = []
test_time_array = []


train_comparison = []
eval_comparison = []

full_start = time.time()
for fold in range(num_folds):
    pairs_tested = []
    pairs_trained = []

    fold_accuracies = {
        "train_token": [],
        "train_soln": [],
        "train_num_x_mse": [],
        "train_op_right": [],
        "train_sni_acc": [],
        "train_losses" : [],
        'train_loss_dict': [],

        "eval_token": [],
        "eval_soln": [],
        "eval_num_x_mse": [],
        "eval_op_right": [],
        "eval_sni_acc": [],
        "eval_losses" : [],
        'eval_loss_dict': [],

        "loss" : []
    }

    fold_train_accuracy = []
    fold_loss = []
    fold_eval_accuracy = []
    fold_soln_eval_accuracy = []
    # train on current fold, test on other folds
    for fold_t in range(num_folds):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums, copy_nums, vars, useCustom, useSeperateVars, useBertEmbeddings, tree=True)
    # all_pairs = train_pairs + test_pairs
    # out = []
    # for pair in all_pairs:
    #     equations = pair[2]
    #     for equ in equations:
    #         english = [output_lang.index2word[i] for i in equ]
    #         out.append(english)
    # with open("src/post/datasetEquations.txt", "w") as f:
    #     for equ in out:
    #         f.write(" ".join(equ) + "\n")
    # print()
    # pair:
    #   input: sentence with all numbers masked as NUM
    #   length of input
    #   output: prefix equation. Numbers from input as N{i}, non input as constants
    #   length of output
    #   nums: numbers from the input text
    #   loc nums: where nums are in the text
    #   [[] of where each number in the equation (that is not in the output lang) is found in the nums array]
    # Initialize models
    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,n_layers=n_layers, useBertEmbeddings = useBertEmbeddings, input_lang=input_lang)
    encoder_var = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,n_layers=n_layers, useBertEmbeddings = useBertEmbeddings, input_lang=input_lang)
    if useSeperateVars:
        op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars)
    else:
        op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums)

    predict = Prediction(hidden_size=hidden_size, op_nums=op_nums, input_size=len(generate_nums), num_vars=len(vars))
    predict_output = Prediction(hidden_size=hidden_size, op_nums=op_nums, input_size=len(generate_nums), num_vars=len(vars))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=op_nums, embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    num_x_predict = PredictNumX(hidden_size=hidden_size, output_size=4, batch_size=batch_size)
    x_generate = GenerateXs(hidden_size=hidden_size, output_size=5, batch_size=batch_size)
    x_to_q = XToQ(hidden_size=hidden_size)

    sementic_alignment = Seq2TreeSemanticAlignment(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, hidden_size=hidden_size)
    num_or_opp = NumOrOpp(512)
    sni = SNI(hidden_size=hidden_size)
    fix_t = FixT(hidden_size=hidden_size)



    models = {
        "encoder": encoder,
        "encoder_var": encoder_var,
        "predict": predict,
        'predict_output': predict_output,
        "generate": generate,
        "merge": merge,
        "num_x_predict": num_x_predict,
        "q_generate": x_generate,
        "q_to_x": x_to_q,
        "semantic_alignment": sementic_alignment,
        "num_or_opp": num_or_opp,
        "sni": sni,
        "fix_t": fix_t
    }

    debug = {
        "active" : True,
        "output_lang": output_lang
    }
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    encoder_var_optimizer = torch.optim.Adam(encoder_var.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_output_optimizer = torch.optim.Adam(predict_output.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_x_predict_optimizer = torch.optim.Adam(num_x_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_generate_optimizer = torch.optim.Adam(x_generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_to_q_optimizer = torch.optim.Adam(x_to_q.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sementic_alignment_optimizer = torch.optim.Adam(sementic_alignment.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_or_opp_optimizer = torch.optim.Adam(num_or_opp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sni_optimizer = torch.optim.Adam(sni.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fix_t_optimizer = torch.optim.Adam(fix_t.parameters(), lr=learning_rate, weight_decay=weight_decay)


    optimizers = [
        encoder_optimizer,
        encoder_var_optimizer,
        predict_optimizer,
        predict_output_optimizer,
        generate_optimizer,
        merge_optimizer,
        num_x_predict_optimizer,
        x_generate_optimizer,
        x_to_q_optimizer,
        sementic_alignment_optimizer,
        num_or_opp_optimizer,
        sni_optimizer,
        fix_t_optimizer
    ]

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    encoder_var_scheduler = torch.optim.lr_scheduler.StepLR(encoder_var_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    predict_output_scheduler = torch.optim.lr_scheduler.StepLR(predict_output_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    num_x_predict_scheduler = torch.optim.lr_scheduler.StepLR(num_x_predict_optimizer, step_size=20, gamma=0.5)
    x_generate_scheduler = torch.optim.lr_scheduler.StepLR(x_generate_optimizer, step_size=20, gamma=0.5)
    x_to_q_scheduler = torch.optim.lr_scheduler.StepLR(x_to_q_optimizer, step_size=20, gamma=0.5)
    sementic_alignment_scheduler = torch.optim.lr_scheduler.StepLR(sementic_alignment_optimizer, step_size=20, gamma=0.5)
    num_or_opp_scheduler = torch.optim.lr_scheduler.StepLR(num_or_opp_optimizer, step_size=20, gamma=0.5)
    sni_scheduler = torch.optim.lr_scheduler.StepLR(sni_optimizer, step_size=20, gamma=0.5)
    fix_t_scheduler = torch.optim.lr_scheduler.StepLR(fix_t_optimizer, step_size=20, gamma=0.5)

    schedulers = [
        encoder_scheduler,
        encoder_var_scheduler,
        predict_scheduler,
        predict_output_scheduler,
        generate_scheduler,
        merge_scheduler,
        num_x_predict_scheduler,
        x_generate_scheduler,
        x_to_q_scheduler,
        sementic_alignment_scheduler,
        num_or_opp_scheduler,
        sni_scheduler,
        fix_t_scheduler
    ]

    # Move models to GPU
    if USE_CUDA:
        for k,v in models.items():
            v.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(n_epochs):
        for scheduler in schedulers:
            scheduler.step()
        # for scheduler in schedulers:
        #     scheduler.step()
        # loss_total = 0
        # input_batches: padded inputs
        # input_lengths: length of the inputs (without padding)
        # output_batches: padded outputs
        # output_length: length of the outputs (without padding)
        # num_batches: numbers from the input text 
        # num_stack_batches: the corresponding nums lists
        # num_pos_batches: positions of the numbers lists
        # num_size_batches: number of numbers from the input text
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches, output_var_solutions, equation_targets, var_pos, batches_sni = prepare_train_batch(train_pairs, batch_size, vars, output_lang, input_lang)
        # generate temp x vectors

        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        train_accuracys = []
        batch_accuricies = {
            "train_token": [],
            "train_soln": [],
            "train_num_x_mse": [],
            "train_op_right": [],
            "train_sni_acc": [],
            "train_total_loss": 0,
            "train_loss_dict": [],


            "eval_token": [],
            "eval_soln": [],
            "eval_op_right": [],
            "eval_num_x_mse": [],
            "eval_sni_acc": [],
            "eval_total_loss": 0, 
            "eval_loss_dict": []
        } 
        start = time.time()
        for idx in range(len(input_lengths)):
            # Zero gradients of both optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Make sure all are in training mode
            for k,v in models.items():
                v.train()

            input_batch_len = len(input_batches[idx])
            start = time.perf_counter()
            loss, acc, num_x_mse, comparison, op_right, sni_acc, loss_dict, acc_list = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models,
                output_lang, num_pos_batches[idx], equation_targets[idx], var_pos[idx], batches_sni[idx], useCustom, vars, debug, setName, useSemanticAlignment, useSeperateVars, useOpScaling, useVarsAsNums, useSNIMask, useTFix, True)
            end = time.perf_counter()
            train_time_array.append([input_batch_len,end - start])
            train_comparison.append(comparison)
            # loss_total += loss
            batch_accuricies['train_total_loss'] += loss
            batch_accuricies["train_token"].append(acc)
            batch_accuricies["train_op_right"].append(op_right)
            batch_accuricies["train_num_x_mse"].append(num_x_mse)
            batch_accuricies["train_sni_acc"].append(sni_acc)
            batch_accuricies['train_loss_dict'].append(loss_dict)
            # train_accuracys.append(acc)
            
            # Step the optimizers
            for optimizer in optimizers:
                optimizer.step()
        # step the schedulers


        batch_loss = batch_accuricies['train_total_loss'] / len(input_lengths)
        batch_train_acc = sum(batch_accuricies["train_token"]) / len(batch_accuricies["train_token"])
        batch_train_op_right = sum(batch_accuricies["train_op_right"]) / len(batch_accuricies["train_op_right"])
        batch_train_num_x_mse = sum(batch_accuricies["train_num_x_mse"]) / len(batch_accuricies["train_num_x_mse"])
        batch_train_sni_acc = sum(batch_accuricies["train_sni_acc"]) / len(batch_accuricies["train_sni_acc"])

        print("loss:", batch_loss)
        print("train accuracy", batch_train_acc)

        fold_accuracies["train_losses"].append(batch_loss)
        fold_accuracies["train_token"].append(batch_train_acc)
        fold_accuracies["train_op_right"].append(batch_train_op_right)
        fold_accuracies["train_num_x_mse"].append(batch_train_num_x_mse)
        fold_accuracies["train_sni_acc"].append(batch_train_sni_acc)
        fold_accuracies["train_loss_dict"].append(batch_accuricies['train_loss_dict'])


        if True:
            # eval_accuracies =  {
            #     "eval_token": [],
            #     "eval_soln": []
            # }
            batch_eval_comparison = []
            # for test_batch in test_pairs:
            input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches, output_var_solutions, equation_targets, var_pos, batches_sni = prepare_train_batch(test_pairs, 1, vars, output_lang, input_lang)
            for idx in range(len(input_lengths)):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                for k, v in models.items():
                    v.eval()
                input_batch_len = len(input_batches[idx])
                start = time.perf_counter()
                loss, acc, num_x_mse, comparison, op_right, sni_acc, loss_dict, acc_list = train_tree( input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx], num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models, output_lang, num_pos_batches[idx], equation_targets[idx], var_pos[idx], batches_sni[idx], useCustom, vars, debug, setName, useSemanticAlignment, useSeperateVars, useOpScaling, useVarsAsNums, useSNIMask, useTFix, False) 
                print()
                end = time.perf_counter()
                test_time_array.append([input_batch_len,end - start])
                # testc.append(comparison)
                batch_accuricies['eval_total_loss'] += loss
                batch_accuricies["eval_token"].append(acc)
                batch_accuricies["eval_op_right"].append(op_right)
                batch_accuricies["eval_num_x_mse"].append(num_x_mse)
                batch_accuricies["eval_sni_acc"].append(sni_acc)
                batch_accuricies['eval_loss_dict'].append(loss_dict)
                if acc == 1:
                    batch_accuricies["eval_soln"].append(1)
                else:
                    batch_accuricies["eval_soln"].append(0)

            batch_loss = batch_accuricies['eval_total_loss'] / len(input_lengths)
            batch_eval_acc = sum(batch_accuricies["eval_token"]) / len(batch_accuricies["eval_token"])
            batch_eval_op_right = sum(batch_accuricies["eval_op_right"]) / len(batch_accuricies["eval_op_right"])
            batch_eval_num_x_mse = sum(batch_accuricies["eval_num_x_mse"]) / len(batch_accuricies["eval_num_x_mse"])
            batch_eval_sni_acc = sum(batch_accuricies["eval_sni_acc"]) / len(batch_accuricies["eval_sni_acc"])
            batch_eval_soln_acc = sum(batch_accuricies["eval_soln"]) / len(batch_accuricies["eval_soln"])

            print("loss:", batch_loss)
            print("eval accuracy", batch_eval_acc)

            fold_accuracies["eval_losses"].append(batch_loss)
            fold_accuracies["eval_token"].append(batch_eval_acc)
            fold_accuracies["eval_op_right"].append(batch_eval_op_right)
            fold_accuracies["eval_num_x_mse"].append(batch_eval_num_x_mse)
            fold_accuracies["eval_sni_acc"].append(batch_eval_sni_acc)
            fold_accuracies["eval_soln"].append(batch_eval_soln_acc)
            fold_accuracies["eval_loss_dict"].append(batch_accuricies['eval_loss_dict'])
            # eval_acc = sum(batch_accuricies["eval_token"]) / len(batch_accuricies["eval_token"])
            # # eval_soln_acc = sum(batch_accuricies["eval_soln"]) / len(batch_accuricies["eval_soln"])
            # eval_loss = batch_accuricies['eval_total_loss'] / len(input_lengths)
            # eval_num_x_mse = sum(batch_accuricies["eval_num_x_mse"]) / len(batch_accuricies["eval_num_x_mse"])
            # print('eval accuracy', eval_acc)
            # fold_accuracies["eval_token"].append(eval_acc)
            # fold_accuracies['eval_total_loss'] = eval_loss
            # # fold_accuracies["eval_soln"].append(eval_soln_acc)
            # fold_accuracies["eval_num_x_mse"].append(eval_num_x_mse)
            # # fold_eval_accuracy.append(eval_acc)
            # # fold_soln_eval_accuracy.append(eval_soln_acc)

            print("------------------------------------------------------")
            # torch.save(encoder.state_dict(), "models/encoder")
            # torch.save(predict.state_dict(), "models/predict")
            # torch.save(generate.state_dict(), "models/generate")
            # torch.save(merge.state_dict(), "models/merge")
    all_train_accuracys.append(fold_accuracies["train_token"])
    all_eval_accuracys.append(fold_accuracies["eval_token"])

    all_train_loss.append(fold_accuracies["train_losses"])
    all_eval_loss.append(fold_accuracies["eval_losses"])
    all_soln_eval_accuracys.append(fold_accuracies["eval_soln"])

    # all_soln_eval_accuracys.append(fold_accuracies["eval_soln"])

    for k, v in fold_accuracies.items():
        print(k, v)
        print("\n")
    # print('COMPARISONS', train_comparison, eval_comparison)
    # write_comparison(train_comparison, eval_comparison)
    # print('fold accuracies', fold_accuracies)
    # make_loss_graph(
    #     fold_accuracies['loss'], 
    #     f"src/post/loss-{time.time()}-{run_id}.png", title,
    #     "Epoch", "Loss By Epoch"
    #     )
    make_eval_graph(
        [fold_accuracies["train_losses"], fold_accuracies["eval_losses"]], 
        ['Train', "Eval"],
        f"src/post/loss-{time.time()}-{run_id}.png", title,
        "Epoch", "Loss By Epoch", None 
        )
    make_eval_graph(
        [fold_accuracies["train_token"], fold_accuracies["eval_token"]], 
        ['Train', "Eval"],
        f"src/post/accuracy-{time.time()}-{run_id}.png", title,
        "Epoch", "Accuracy By Epoch", [0, 1]
        )
    print('fold train accuracy', fold_accuracies["train_token"])
    print('fold eval accuracy', fold_accuracies['eval_token'])
    print('All TRAIN ACC', all_train_accuracys)
    print('ALL EVAL ACC', all_eval_accuracys)
    print('ALL EVAL SOLN ACC', all_soln_eval_accuracys)
    process_loss_dicts(fold_accuracies['train_loss_dict'], fold_accuracies['eval_loss_dict'], f"src/post/loss-dict-{time.time()}-{run_id}.png")
    break 

# a, b, c = 0, 0, 0
# for bl in range(len(best_acc_fold)):
#     a += best_acc_fold[bl][0]
#     b += best_acc_fold[bl][1]
#     c += best_acc_fold[bl][2]
#     print(best_acc_fold[bl])
# print(a / float(c), b / float(c))


train_time_per_all = []
test_time_per_all = []
for length, runtime in train_time_array:
    time_per = runtime / length
    train_time_per_all.append(time_per)

for length, runtime in test_time_array:
    time_per = runtime / length
    test_time_per_all.append(time_per)

print('train time per token', sum(train_time_per_all) / len(train_time_per_all))
print('infrence time per token', sum(test_time_per_all) / len(test_time_per_all))

full_end = time.time()
total_run_time = full_end - full_start
print("total run time", total_run_time)
