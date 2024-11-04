# coding: utf-8
import os
from re import I

from torch import eq
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *

batch_size = 64
# batch_size = 1 
# batch_size = 20
embedding_size = 128
hidden_size = 512
# n_epochs = 2 
# n_epochs = 20 
n_epochs = 80
# learning_rate = 1e-1 
learning_rate = 1e-5 
weight_decay = 1e-5
beam_size = 5
n_layers = 2

os.makedirs("models", exist_ok=True)
setName = "DRAW"
# setName = "MATH"
if setName == "MATH":
    data = load_MATH23k_data("data/Math_23K.json")
else:
    data = load_DRAW_data("data/DRAW/dolphin_t2_final.json")
    # data = load_DRAW_data("data/DRAW/single.json")
# MATH data format:
# {
# "id":"10431",
# "original_text":"The speed of a car is 80 kilometers per hour. It can be written as: how much. Speed ​​* how much = distance.",
# "segmented_text":"The speed of a car is 80 kilometers per hour, which can be written as: how much. speed * how much = distance. ",
# "equation":"x=80",
# "ans":"80"
# }'
# DRAW format:
# {
# "sQuestion",
# "lEquations",
# }

pairs, generate_nums, copy_nums, vars = transfer_num(data, setName)
pairs = pairs[0:20]
# pairs: list of tuples:
#   input_seq: masked text
#   [out_seq]: equation with in text numbers replaced with "N#", and other numbers left as is
#   target equation answer
#   nums: list of numbers in the text
#   num_pos: list of positions of the numbers in the text
# generate_nums: list of common numbers not in input text (ex constants)
# copy_nums:  max length of numbers
# vars in equations

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[3], p[4], p[2], p[5]))


pairs = temp_pairs
# pairs:
#   input_seq: masked text
#   [out_seq]: equation with in text numbers replaced with "N#", and other numbers left as is
#   list of numbers in the text
#   list of positions of the numbers in the text
#   target equation answer

# split data into groups of 20%
fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []


total_training_time = 0
total_inference_time = 0

train_time_array = []
test_time_array = []

all_losses = []
for fold in range(5):
    fold_loss = []
    pairs_tested = []
    pairs_trained = []
    # train on current fold, test on other folds
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    if setName == "MATH":
        input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    else:
        input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 0, generate_nums,
                                                                    copy_nums, tree=True)
    # pair:
    #   input: sentence with all numbers masked as NUM
    #   length of input
    #   [outputs]: prefix equation. Numbers from input as N{i}, non input as constants
    #   [length of outputs]
    #   nums: numbers from the input text
    #   loc nums: where nums are in the text
    #   [[] of where each token in the equation is found in the nums array]
    # use this
    # for input_seq, input_length, equations, equation_lengths, input_nums, input_nums_pos, num_stack, eqn_vars in train_pairs:
        # print("     problem", input_lang.ids_to_tokens(input_seq))
        # print("     equations", [output_lang.ids_to_tokens(equations[i]) for i in range(len(equations))])
        # print("     vars", eqn_vars)

    # # confirm:
    # print("for input:", temp_pairs[0])
    # print("     problem", input_lang.ids_to_tokens(train_pairs[0][0]))
    # print("     equations", [output_lang.ids_to_tokens(train_pairs[0][2][i]) for i in range(len(train_pairs[0][2]))])
    # pair:
    #   input: sentence with all numbers masked as NUM
    #   length of input
    #   output: prefix equation. Numbers from input as N{i}, non input as constants
    #   length of output
    #   nums: numbers from the input text
    #   loc nums: where nums are in the text
    #   [[] of where each number in the equation (that is not in the output lang) is found in the nums array]
    # Initialize models
    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
    # max of 5 possible trees generated 
    num_x_predict = PredictNumX(hidden_size=hidden_size, output_size=5, batch_size=batch_size)
    x_generate = GenerateXs(hidden_size=hidden_size, output_size=5, batch_size=batch_size)
    x_to_q = XToQ(hidden_size=hidden_size)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_x_predict_optimizer = torch.optim.Adam(num_x_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_generate_optimizer = torch.optim.Adam(x_generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_to_q_optimizer = torch.optim.Adam(x_to_q.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    num_x_predict_scheduler = torch.optim.lr_scheduler.StepLR(num_x_predict_optimizer, step_size=20, gamma=0.5)
    x_generate_scheduler = torch.optim.lr_scheduler.StepLR(x_generate_optimizer, step_size=20, gamma=0.5)
    x_to_q_scheduler = torch.optim.lr_scheduler.StepLR(x_to_q_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(n_epochs):
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        num_x_predict_scheduler.step()
        x_generate_scheduler.step()
        x_to_q_scheduler.step()

        loss_total = 0
        # input_batches: padded inputs
        # input_lengths: length of the inputs (without padding)
        # output_batches: padded outputs
        # output_length: length of the outputs (without padding)
        # num_batches: numbers from the input text 
        # num_stack_batches: the corresponding nums lists
        # num_pos_batches: positions of the numbers lists
        # num_size_batches: number of numbers from the input text
        input_batches, input_lengths, output_batches, output_batch_mask, output_lengths, output_tokens, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, solution_batches, var_tokens_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        # start = time.time()
        for idx in range(len(input_lengths)):
            start = time.perf_counter()
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_batch_mask[idx], output_lengths[idx], output_tokens[idx],
                num_stack_batches[idx], num_size_batches[idx], var_tokens_batches[idx], solution_batches[idx], generate_num_ids, encoder, num_x_predict, x_generate, x_to_q, predict, generate, merge,
                encoder_optimizer, num_x_predict_optimizer, x_generate_optimizer, x_to_q_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], output_lang.variables)
            end = time.perf_counter()
            train_time_array.append([1, end - start])
            loss_total += loss

        print(f"epoch {epoch} fold {fold} loss:", loss_total / len(input_lengths))
        fold_loss.append(loss_total / len(input_lengths))
        # print("training time", time_since(time.time() - start))
        print("--------------------------------")
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()
        x_to_q_optimizer.step()
        x_generate_optimizer.step()
        num_x_predict_optimizer.step()
        # if epoch % 10 == 0 or epoch > n_epochs - 5:
        if True:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            # start = time.time()
            # print('test pairs', test_pairs)
            for test_batch in test_pairs:
                start = time.perf_counter()
                test_res, pred_token = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate, x_generate, x_to_q, num_x_predict, merge, output_lang, test_batch[5], beam_size=beam_size)
                end = time.perf_counter()
                test_time_array.append([1, end - start])
                # print('test res', test_res)
                # print('test token', pred_token)
                # for i in test_res:
                #     print('i', i)
                #     # for j in i:
                #     #     print('j', j)
                #     print(output_lang.index2word(i))
                # print('test result', [output_lang.index2word[i] for i in test_res[0] ])
                val_ac, equ_ac, test, tar = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                # print('test', test)
                # print('tar', tar)
                # print('actual', test_batch[2])
                actuals = []
                for equation in test_batch[2]:
                    actual = [output_lang.index2word[i] for i in equation]
                    actuals.append(actual)
                print('actuals', actuals)
                print('actual_tokens', [output_lang.index2word[i] for i in test_batch[7]])
                preds = []
                for equation in test_res:
                    predicted = [output_lang.index2word[i] for i in equation]
                    preds.append(predicted)
                print('preds', preds)
                print('pred_tokens', pred_token)
                same = 0
                print(len(actual), len(predicted))
                for i in range(min(*test_batch[3], len(predicted), len(actual) )):
                    # print('checking', actual[i], predicted[i])
                    if actual[i] == predicted[i]:
                        same += 1
                for i in range(min(len(pred_token), len(test_batch[7]))):
                    if pred_token[i] == test_batch[7][i]:
                        same += 1
                print("actual     " , actual)
                print("predicted  ", predicted)
                print('same:', same, same/(max(test_batch[3]) + max(len(pred_token), len(test_batch[7]))))

                print("\n")
                # for i in test_res:
                #     print(output_lang.index2word[i])
                # for i in test_batch[2]:
                #     print(output_lang.index2word[i])
                # print('value accuracy', val_ac)
                # print('equation accuracy', equation_ac)
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            # print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(encoder.state_dict(), "models/encoder")
            torch.save(predict.state_dict(), "models/predict")
            torch.save(generate.state_dict(), "models/generate")
            torch.save(merge.state_dict(), "models/merge")
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))
    all_losses.append(fold_loss)
    print('FOLD OUTPUT', fold_loss)
    train_time_per_all = []
    test_time_per_all = []
    for length, runtime in train_time_array:
        time_per = runtime / length
        train_time_per_all.append(time_per)

    for length, runtime in test_time_array:
        time_per = runtime / length
        test_time_per_all.append(time_per)

    print('epoch' , epoch, 'fold', fold, 'train time per token', sum(train_time_per_all) / len(train_time_per_all))
    print('epoch', epoch, 'fold', fold, 'infrence time per token', sum(test_time_per_all) / len(test_time_per_all))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))


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

print('all losses', all_losses)
