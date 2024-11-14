# coding: utf-8
import os
from src.train_and_evaluate import *
from src.models import *
from src.post.loss_graph import *
import time
import torch.optim
from src.expressions_transfer import *

# batch_size = 64
torch.manual_seed(1)
random.seed(1)
batch_size = 5 
embedding_size = 128
hidden_size = 512
n_epochs = 10 
learning_rate = 1e-3 
weight_decay = 1e-5
beam_size = 5
n_layers = 2

# torch.autograd.set_detect_anomaly(True)

useCustom = True
# useCustom = False 
# setName = "MATH"
setName = "DRAW"
os.makedirs("models", exist_ok=True)
if setName == "DRAW":
    data = load_DRAW_data("data/DRAW/dolphin_t2_final.json")
else:
    data = load_raw_data("data/Math_23K.json")
data = data[0:10]
# data = load_raw_data("data/DRAW/draw.json")
# data = None
# with open("data/DRAW/draw.json", "r") as f:
#     data = json.loads(f.read())
# data format:
# {
# "id":"10431",
# "original_text":"The speed of a car is 80 kilometers per hour. It can be written as: how much. Speed ​​* how much = distance.",
# "segmented_text":"The speed of a car is 80 kilometers per hour, which can be written as: how much. speed * how much = distance. ",
# "equation":"x=80",
# "ans":"80"
# }'

pairs, generate_nums, copy_nums, vars = transfer_num(data, setName, useCustom)
pairs = pairs[0:10]
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
    equations = [from_infix_to_prefix(equ) for equ in p[1]]
    temp_pairs.append((p[0], equations, p[2], p[3], p[4]))
pairs = temp_pairs


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
all_eval_accuracys = []

total_training_time = 0
total_inference_time = 0

train_time_array = []
test_time_array = []

for fold in range(num_folds):
    pairs_tested = []
    pairs_trained = []
    fold_train_accuracy = []
    fold_eval_accuracy = []
    # train on current fold, test on other folds
    for fold_t in range(num_folds):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, vars, useCustom, tree=True)
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
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars), input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars), embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    num_x_predict = PredictNumX(hidden_size=hidden_size, output_size=3, batch_size=batch_size)
    x_generate = GenerateXs(hidden_size=hidden_size, output_size=5, batch_size=batch_size)
    x_to_q = XToQ(hidden_size=hidden_size)


    models = {
        "encoder": encoder,
        "predict": predict,
        "generate": generate,
        "merge": merge,
        "num_x_predict": num_x_predict,
        "x_generate": x_generate,
        "x_to_q": x_to_q
    }
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_x_predict_optimizer = torch.optim.Adam(num_x_predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_generate_optimizer = torch.optim.Adam(x_generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_to_q_optimizer = torch.optim.Adam(x_to_q.parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizers = [
        encoder_optimizer,
        predict_optimizer,
        generate_optimizer,
        merge_optimizer,
        num_x_predict_optimizer,
        x_generate_optimizer,
        x_to_q_optimizer
    ]

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    num_x_predict_scheduler = torch.optim.lr_scheduler.StepLR(num_x_predict_optimizer, step_size=20, gamma=0.5)
    x_generate_scheduler = torch.optim.lr_scheduler.StepLR(x_generate_optimizer, step_size=20, gamma=0.5)
    x_to_q_scheduler = torch.optim.lr_scheduler.StepLR(x_to_q_optimizer, step_size=20, gamma=0.5)

    schedulers = [
        encoder_scheduler,
        predict_scheduler,
        generate_scheduler,
        merge_scheduler,
        num_x_predict_scheduler,
        x_generate_scheduler,
        x_to_q_scheduler
    ]

    # Move models to GPU
    if USE_CUDA:
        for k,v in models():
            v.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(n_epochs):
        for scheduler in schedulers:
            scheduler.step()
        loss_total = 0
        # input_batches: padded inputs
        # input_lengths: length of the inputs (without padding)
        # output_batches: padded outputs
        # output_length: length of the outputs (without padding)
        # num_batches: numbers from the input text 
        # num_stack_batches: the corresponding nums lists
        # num_pos_batches: positions of the numbers lists
        # num_size_batches: number of numbers from the input text
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches = prepare_train_batch(train_pairs, batch_size, vars)
        # generate temp x vectors

        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        train_accuracys = []
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
            loss, acc = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models,
                output_lang, num_pos_batches[idx], useCustom)
            end = time.perf_counter()
            train_time_array.append([input_batch_len,end - start])
            loss_total += loss
            train_accuracys.append(acc)
            
            # Step the optimizers
            for optimizer in optimizers:
                optimizer.step()

        print("loss:", loss_total / len(input_lengths))
        # print("training time", time_since(time.time() - start))
        print("--------------------------------")
        fold_train_accuracy.append(sum(train_accuracys) / len(train_accuracys))
        # if epoch % 10 == 0 or epoch > n_epochs - 5:
        if True:
            for k, v in models.items():
                v.eval()
            eval_accuracys = []
            start = time.time()
            for test_batch in test_pairs:
                start = time.perf_counter()
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, models, output_lang, test_batch[5], vars, useCustom, beam_size=beam_size)
                end = time.perf_counter()
                test_time_array.append([1, end - start])
                lengths = 0
                same = 0
                for equ_count in range(len(test_batch[2])):
                    equ_length = test_batch[3][equ_count]
                    actual = [output_lang.index2word[i] for i in test_batch[2][equ_count]]
                    print('actual', actual[0:equ_length])
                    try:
                        print('preds', [output_lang.index2word[i] for i in test_res[equ_count][0:min(equ_length, len(test_res[equ_count]))]])
                    except:
                        print('preds', "None")
                    for token in range(equ_length):
                        lengths += 1
                        token_actual = actual[token]
                        try:
                            token_pred = output_lang.index2word[test_res[equ_count][token]]
                        except:
                            token_pred = None
                        if token_actual == token_pred:
                            same += 1

                accuracy = same / lengths
                eval_accuracys.append(accuracy)

            fold_eval_accuracy.append(sum(eval_accuracys) / len(eval_accuracys))

            print("------------------------------------------------------")
            # torch.save(encoder.state_dict(), "models/encoder")
            # torch.save(predict.state_dict(), "models/predict")
            # torch.save(generate.state_dict(), "models/generate")
            # torch.save(merge.state_dict(), "models/merge")
    all_train_accuracys.append(fold_train_accuracy)
    all_eval_accuracys.append(fold_eval_accuracy)
    make_loss_graph(
        [fold_train_accuracy, fold_eval_accuracy], 
        ['Train', "Eval"],
        f"src/post/accuracy-{time.time()}-{fold}.png", "Accuracy",
        "Epoch", "Accuracy By Epoch"
        )
    print('All TRAIN ACC', all_train_accuracys)
    print('ALL EVAL ACC', all_eval_accuracys)
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

