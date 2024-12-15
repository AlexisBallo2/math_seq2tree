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



# batch_size = 64
# torch.manual_seed(10)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(10)
# torch.cuda.manual_seed_all(2)
# np.random.seed(10)

# batch_size = 2 
# batch_size = 10
# batch_size = 30 
batch_size = 64 
embedding_size = 128
hidden_size = 512
# n_epochs = 3 
# n_epochs = 10 
n_epochs = 20 
learning_rate = 1e-3 
# learning_rate = 1e-3 
# learning_rate = 1e-3 
weight_decay = 1e-5
beam_size = 5
n_layers = 2

# num_obs = 20
# num_obs = 100
# num_obs = 600 
# num_obs = 1000 
num_obs = None 

# torch.autograd.set_detect_anomaly(True)

useCustom = True
# useCustom = False 

# setName = "MATH"
setName = "DRAW"

# decide if we must be able to solve equation
useEquSolutions = True
# useEquSolutions = False 
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
    "title" : title
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

pairs, generate_nums, copy_nums, vars = transfer_num(data, setName, useCustom, useEquSolutions)
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
    equ_with_equals = []
    for equ in p['equations']:
        equ_with_equals += equ
    p['equations'] = [equ_with_equals]
    p['equationTargetVars'] = ["0"]
    # temp_pairs.append((p[0], equations, p[2], p[3], p[4], p[5], p[6], p[7]))
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

        "eval_token": [],
        "eval_soln": [],
        "eval_num_x_mse": [],

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

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums, copy_nums, vars, useCustom, tree=True)
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
    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,n_layers=n_layers)
    encoder_var = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars), input_size=len(generate_nums))
    predict_output = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars), input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars), embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    num_x_predict = PredictNumX(hidden_size=hidden_size, output_size=4, batch_size=batch_size)
    x_generate = GenerateXs(hidden_size=hidden_size, output_size=5, batch_size=batch_size)
    x_to_q = XToQ(hidden_size=hidden_size)

    sementic_alignment = Seq2TreeSemanticAlignment(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, hidden_size=hidden_size)


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
        "sementic_alignment": sementic_alignment
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
        sementic_alignment_optimizer
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
        sementic_alignment_scheduler
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
        loss_total = 0
        # input_batches: padded inputs
        # input_lengths: length of the inputs (without padding)
        # output_batches: padded outputs
        # output_length: length of the outputs (without padding)
        # num_batches: numbers from the input text 
        # num_stack_batches: the corresponding nums lists
        # num_pos_batches: positions of the numbers lists
        # num_size_batches: number of numbers from the input text
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches, output_var_solutions, equation_targets, var_pos = prepare_train_batch(train_pairs, batch_size, vars, output_lang, input_lang)
        # generate temp x vectors

        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        train_accuracys = []
        batch_accuricies = {
            "train_token": [],
            "train_soln": [],
            "train_num_x_mse": [],


            "eval_token": [],
            "eval_soln": [],
            "eval_num_x_mse": []
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
            loss, acc, num_x_mse, comparison = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models,
                output_lang, num_pos_batches[idx], equation_targets[idx], var_pos[idx], useCustom, vars, debug, setName)
            end = time.perf_counter()
            train_time_array.append([input_batch_len,end - start])
            train_comparison.append(comparison)
            loss_total += loss
            batch_accuricies["train_token"].append(acc)
            batch_accuricies["train_num_x_mse"].append(num_x_mse)
            # train_accuracys.append(acc)
            
            # Step the optimizers
            for optimizer in optimizers:
                optimizer.step()
        # step the schedulers


        print("loss:", loss_total / len(input_lengths))
        batch_train_acc = sum(batch_accuricies["train_token"]) / len(batch_accuricies["train_token"])
        print("train accuracy", batch_train_acc)
        fold_accuracies["loss"].append(loss_total / len(input_lengths))
        fold_accuracies["train_token"].append(batch_train_acc)
        # fold_train_accuracy.append(train_acc)
        fold_loss.append(loss_total / len(input_lengths))
        fold_accuracies["train_num_x_mse"].append(sum(batch_accuricies["train_num_x_mse"]) / len(batch_accuricies["train_num_x_mse"]))
        # fold_train_accuracy.append(train_acc)
        # print("training time", time_since(time.time() - start))
        # print("--------------------------------")
        # if epoch % 10 == 0 or epoch > n_epochs - 5:
        if True:
            # eval_accuracies =  {
            #     "eval_token": [],
            #     "eval_soln": []
            # }
            for k, v in models.items():
                v.eval()
            start = time.time()
            batch_eval_comparison = []
            for test_batch in test_pairs:
                start = time.perf_counter()
                test_res, pred_num_x = evaluate_tree(test_batch['input_cell'], test_batch['input_len'], generate_num_ids, models, input_lang, output_lang, test_batch['num_pos'], vars, useCustom, debug, beam_size=beam_size)
                end = time.perf_counter()
                test_time_array.append([1, end - start])
                lengths = 0
                same = 0
                num_pred_equations = len(test_res)
                equation_strings = []
                print('test res')

                num_x_mse = (pred_num_x - len(test_batch['equations']))**2
                print(f'predicted num x mse: {pred_num_x}, actual: {len(test_batch["equations"])}')
                batch_accuricies["eval_num_x_mse"].append(num_x_mse)
                for equ_count in range(len(test_batch['equations'])):
                    actual_length = test_batch['equation_lens'][equ_count]
                    actual = [output_lang.index2word[i] for i in test_batch['equations'][equ_count][0:actual_length + 1]]
                    if equ_count > len(test_res) - 1:
                        predicted = [None for i in range(len(actual))]
                    else:
                        equn, token = test_res[equ_count]
                        # print('temp predicted', [output_lang.index2word[i] for i in equn])
                        pred_token = output_lang.index2word[token]
                        predicted = [output_lang.index2word[i] if i < len(output_lang.index2word) else " " for i in equn ]
                        replaced_nums = replace_nums(test_batch['pairNumMapping'], predicted)
                        predicted_infix = from_prefix_to_infix(replaced_nums)
                        
                        # predicted = [output_lang.index2word[i] for i in equn[0:min(len(test_res[equ_count]) + 1, actual_length + 1)]]
                        equation_strings.append(pred_token + "=" + predicted_infix)
                    print(f"    equation {equ_count}")
                    print("         actual", actual)
                    print("         predicted", predicted)
                    batch_eval_comparison.append({
                        "prediction": predicted,
                        "actual": actual
                    })

                    for i in range(len(actual)):
                        lengths += 1
                        if i < len(predicted):
                            if actual[i] == predicted[i]:
                                same += 1
                    # if setName == "DRAW":
                    #     lengths += 1
                    #     if pred_token == test_batch['equationTargetVars'][equ_count]:
                    #         same += 1

                print('equation strings', equation_strings)
                if setName == "DRAW":
                    same_equation = solve_equation(equation_strings, test_batch['solution'])
                    if same_equation:
                        print('solution success')
                        batch_accuricies["eval_soln"].append(1)
                    else: 
                        print('solution failed')
                        batch_accuricies["eval_soln"].append(0)
                else:
                    if lengths == same:
                        batch_accuricies["eval_soln"].append(1)
                    else:
                        batch_accuricies["eval_soln"].append(0)

                accuracy = same / lengths
                batch_accuricies["eval_token"].append(accuracy)
                eval_comparison.append(batch_eval_comparison)

            eval_acc = sum(batch_accuricies["eval_token"]) / len(batch_accuricies["eval_token"])
            eval_soln_acc = sum(batch_accuricies["eval_soln"]) / len(batch_accuricies["eval_soln"])
            eval_num_x_mse = sum(batch_accuricies["eval_num_x_mse"]) / len(batch_accuricies["eval_num_x_mse"])
            print('eval accuracy', eval_acc)
            fold_accuracies["eval_token"].append(eval_acc)
            fold_accuracies["eval_soln"].append(eval_soln_acc)
            fold_accuracies["eval_num_x_mse"].append(eval_num_x_mse)
            # fold_eval_accuracy.append(eval_acc)
            # fold_soln_eval_accuracy.append(eval_soln_acc)

            print("------------------------------------------------------")
            # torch.save(encoder.state_dict(), "models/encoder")
            # torch.save(predict.state_dict(), "models/predict")
            # torch.save(generate.state_dict(), "models/generate")
            # torch.save(merge.state_dict(), "models/merge")
    all_train_accuracys.append(fold_accuracies["loss"])
    all_train_loss.append(fold_accuracies["loss"])
    all_eval_accuracys.append(fold_accuracies["eval_token"])
    all_soln_eval_accuracys.append(fold_accuracies["eval_soln"])

    for k, v in fold_accuracies.items():
        print(k, v)
        print("\n")
    # print('COMPARISONS', train_comparison, eval_comparison)
    write_comparison(train_comparison, eval_comparison)
    # print('fold accuracies', fold_accuracies)
    make_loss_graph(
        fold_accuracies['loss'], 
        f"src/post/loss-{time.time()}-{run_id}.png", title,
        "Epoch", "Loss By Epoch"
        )
    make_eval_graph(
        [fold_accuracies["train_token"], fold_accuracies["eval_token"]], 
        ['Train', "Eval"],
        f"src/post/accuracy-{time.time()}-{run_id}.png", title,
        "Epoch", "Accuracy By Epoch"
        )
    print('fold train accuracy', fold_accuracies["train_token"])
    print('fold eval accuracy', fold_accuracies['eval_token'])
    print('All TRAIN ACC', all_train_accuracys)
    print('ALL EVAL ACC', all_eval_accuracys)
    print('ALL EVAL SOLN ACC', all_soln_eval_accuracys)
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
