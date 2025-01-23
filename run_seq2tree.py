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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("DEVICE", device)


# do_saves = True
do_saves = False 
# use_save = True 
use_save = False 

do_folds = True
# do_folds = False
saved_epoch_completed = False
fold_save_completed = False


import sys
args = sys.argv
if "-id" in args:
    id_index = args.index("-id")
    run_id = args[id_index + 1]
    print("ID", run_id)
else:
    run_id = "0"


save_id = 0
if use_save:
    read_save_folder = f"saves/{save_id}"
    os.makedirs(read_save_folder, exist_ok=True)
if do_saves:
    save_folder = f"saves/{run_id}"
    os.makedirs(save_folder, exist_ok=True)

# sys.stdout = open('output.txt','wt')


# batch_size = 64
# torch.manual_seed(10)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(10)
# torch.cuda.manual_seed_all(2)
# np.random.seed(10)


if use_save:
    # config = json.load(open(f"{save_folder}/config.json"))
    load = read_general_state(read_save_folder)
    config = load['config']
    print("CONFIG \n", config)
    data = load['pairs']
    pairs = data
    # generate_nums = load['generate_nums']
    # copy_nums = load['copy_nums']
    # vars = load['vars']
    # input_lang = load['input_lang']
    # output_lang = load['output_lang']
    # train_pairs = load['train_pairs']
    # test_pairs = load['test_pairs']
    # generate_num_ids = load['generate_num_ids']
    # fold_accuracies = load['fold_accuracies']
    # fold = load['fold']
    # fold_pairs = load['fold_pairs']
    # models = load['models']
    # optimizers = load['optimizers']
    # schedulers = load['schedulers']
    train_comparison = load['train_comparison']
    eval_comparison = load['eval_comparison']
    all_train_accuracys = load['all_train_accuracys']
    all_train_loss = load['all_train_loss']
    all_eval_loss = load['all_eval_loss']
    all_eval_accuracys = load['all_eval_accuracys']
    all_soln_eval_accuracys = load['all_soln_eval_accuracys']
    total_training_time = load['total_training_time']
    total_inference_time = load['total_inference_time']
    train_time_array = load['train_time_array']
    test_time_array = load['test_time_array']
    # full_start = load['full_start']
    existing_fold = load['existing_fold']
    generate_nums = load['generate_nums']
    copy_nums = load['copy_nums']
    vars = load['vars']
else:
    config = {
        # "batch_size": 1,
        # "batch_size": 2,
        # "batch_size": 5,
        # "batch_size": 5,
        "batch_size": 64,
        # "embedding_size": 768,
        "embedding_size": 128,
        "hidden_size": 512,
        # "n_epochs": 15,
        # "n_epochs": 20,
        # "n_epochs": 10,
        # "n_epochs" : 20,
        "n_epochs" : 80,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "beam_size": 5,
        "n_layers": 2,
        "useCustom": True,
        # "useCustom": False,
        # "setName" : "PEN",
        "setName" : "MATH",
        # "setName" : "DRAW",
        # "setName" : "MAWPS",
        # "setName" : "ALG",
        "useSubMethod": True,
        "useEquSolutions": True,
        # "useSeperateVars": False,
        "useSeperateVars": True,
        "useSemanticAlignment": True,
        # "useSemanticAlignment": False,
        "opsInNN" : True,
        # "opsInNN" : False,
        "useOpScaling" : False,
        # "useOpScaling" : True,
        'useSNIMask' : False,
        "useOneEquation": False,
        # 'useBertEmbeddings': True,
        'useBertEmbeddings': False,
        'useTFix' : False,
        # "num_folds" : 2,
        "num_folds" : 5,
        # "num_obs": 50,   
        # "num_obs": 100,   
        "num_obs": None,   
    }
    config['title'] = f"{config['num_obs']} Observations, {config['n_epochs']} Epochs, Dataset = {config['setName']}, Custom = {config['useCustom']} ",
    if config['useBertEmbeddings']:
        config['embedding_size ']= 768


    print("CONFIG \n", config)
    if config['setName']== "DRAW":
        data = load_DRAW_data("data/PEN.json", "draw")
    elif config['setName']== "PEN":
        data = load_DRAW_data("data/PEN.json")
    elif config['setName']== "MAWPS":
        data = load_DRAW_data("data/PEN.json", 'mawps')
    elif config['setName']== "ALG":
        data = load_DRAW_data("data/PEN.json", 'alg514')
    else:
        data = load_raw_data("data/Math_23K.json")
    if config['num_obs']:
        data = data[0:config['num_obs']]


    print("len data", len(data))
    # print()
    # data format:
    # {
    # "id":"10431",
    # "original_text":"The speed of a car is 80 kilometers per hour. It can be written as: how much. Speed ​​* how much = distance.",
    # "segmented_text":"The speed of a car is 80 kilometers per hour, which can be written as: how much. speed * how much = distance. ",
    # "equation":"x=80",
    # "ans":"80"
    # }'

    if config['setName'] == "MATH":
        pairs, generate_nums, copy_nums, vars = transfer_num_math(data)
    else:
        pairs, generate_nums, copy_nums, vars = transfer_num(data, config['setName'], config['useCustom'], config['useEquSolutions'], config['useSubMethod'], config['useSeperateVars'])
    # pairs.shuffle()
    random.shuffle(pairs)
    if config['num_obs']:
        pairs = pairs[0:config['num_obs']]
    # pairs: list of tuples:
    #   input_seq: masked text
    #   out_seq: equation with in text numbers replaced with "N#", and other numbers left as is
    #   nums: list of numbers in the text
    #   num_pos: list of positions of the numbers in the text
    # generate_nums: list of common numbers not in input text (ex constants)
    # copy_nums:  max length of numbers

    temp_pairs = []
    # pairs_len = []
    for p in pairs:
        # input_seq, prefixed equation, nums, num_pos
        p['equations'] = [from_infix_to_prefix(equ) for equ in p['equations']]
        # lenof = len(p['equations'])
        # pairs_len.append(lenof)
        if config['useOneEquation']:
            equ_with_equals = []
            for equ in p['equations']:
                equ_with_equals += equ
            p['equations'] = [equ_with_equals]
            p['equationTargetVars'] = ["0"]
        if len(p['equations']) < 4:
            temp_pairs.append(p)
    pairs = temp_pairs
    # pairs = temp_pairs
    # print(Counter(pairs_len))

    if do_folds:
        fold_size = int(len(pairs) * 1/config['num_folds'])
        fold_pairs = []
        for split_fold in range(config['num_folds'] - 1):
            fold_start = fold_size * split_fold
            fold_end = fold_size * (split_fold + 1)
            fold_pairs.append(pairs[fold_start:fold_end])
        fold_pairs.append(pairs[(fold_size * (config['num_folds']-1)):])

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
    existing_fold = 0

    if do_saves:
        save_general_state(save_folder, {
            'config' : config,
            "pairs": pairs,
            "all_train_accuracys": all_train_accuracys,
            "all_train_loss": all_train_loss,
            "all_eval_loss": all_eval_loss,
            "all_eval_accuracys": all_eval_accuracys,
            "all_soln_eval_accuracys": all_soln_eval_accuracys,
            "train_comparison": train_comparison,
            "eval_comparison": eval_comparison,
            "total_training_time": total_training_time,
            "total_inference_time": total_inference_time,
            "train_time_array": train_time_array,
            "test_time_array": test_time_array,
            "existing_fold": existing_fold,
            'generate_nums': generate_nums,
            'copy_nums': copy_nums,
            'vars': vars,
            # "full_start": full_start,
        })


# full_start = time.time()

folds_to_do = config['num_folds']



for fold in range(existing_fold, folds_to_do):
    if use_save and fold_save_completed == False:
        fold_save = read_fold_state(read_save_folder)
        pairs_tested = fold_save['pairs_tested']
        pairs_trained = fold_save['pairs_trained']
        fold_accuracies = fold_save['fold_accuracies']
        fold_pairs = fold_save['fold_pairs']

        output_lang = fold_save['output_lang']
        output_lang = Lang().fromJSON(output_lang)

        input_lang = fold_save['input_lang']
        input_lang = Lang().fromJSON(input_lang)

    else:
        pairs_tested = []
        # pairs_tested = 
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
        if do_folds:
            # train on current fold, test on other folds
            for fold_t in range(config["num_folds"]):
                if fold_t == fold:
                    pairs_tested += fold_pairs[fold_t]
                else:
                    pairs_trained += fold_pairs[fold_t]
        else:
            pairs_tested = get_draw_train(pairs, 'test')
            pairs_trained = get_draw_train(pairs, 'train')

        input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums, copy_nums, vars, config['useCustom'], config['useSeperateVars'], config['useBertEmbeddings'], tree=True)
        if do_saves:
            save_fold_state(save_folder, {
            "config": config,
            "generate_nums": generate_nums,
            "copy_nums": copy_nums,
            "vars": vars,
            "input_lang": input_lang,
            "output_lang": output_lang,
            'pairs_tested': pairs_tested,
            'pairs_trained': pairs_trained,
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "fold": fold,
            "fold_pairs": fold_pairs,
            "fold_accuracies": fold_accuracies,
            })
        

    if use_save and saved_epoch_completed == False:
        epoch_load = read_epoch_state(read_save_folder)

        start_epoch = epoch_load['epoch']

        models = epoch_load['models']
        optimizers = epoch_load['optimizers']
        schedulers = epoch_load['schedulers']
    else:
        # define models
        encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config['embedding_size'], hidden_size=config['hidden_size'],n_layers=config['n_layers'], useBertEmbeddings = config['useBertEmbeddings'], input_lang=input_lang)
        encoder_var = EncoderSeq(input_size=input_lang.n_words, embedding_size=config["embedding_size"], hidden_size=config['hidden_size'],n_layers=config['n_layers'], useBertEmbeddings = config['useBertEmbeddings'], input_lang=input_lang)
        if config['useSeperateVars']:
            op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(vars)
        else:
            op_nums = output_lang.n_words - copy_nums - 1 - len(generate_nums)

        predict = Prediction(hidden_size=config['hidden_size'], op_nums=op_nums, input_size=len(generate_nums), num_vars=len(vars), opsInNN=config['opsInNN'])
        predict_output = Prediction(hidden_size=config['hidden_size'], op_nums=op_nums, input_size=len(generate_nums), num_vars=len(vars))
        generate = GenerateNode(hidden_size=config['hidden_size'], op_nums=op_nums, embedding_size=config['embedding_size'])
        merge = Merge(hidden_size=config['hidden_size'], embedding_size=config['embedding_size'])

        num_x_predict = PredictNumX(hidden_size=config['hidden_size'], output_size=4, batch_size=config['batch_size'])
        x_generate = GenerateXs(hidden_size=config['hidden_size'], output_size=4, batch_size=config['batch_size'])
        x_to_q = XToQ(hidden_size=config['hidden_size'])

        sementic_alignment = Seq2TreeSemanticAlignment(encoder_hidden_size=config['hidden_size'], decoder_hidden_size=config['hidden_size'], hidden_size=config['hidden_size'])
        num_or_opp = NumOrOpp(512)
        sni = SNI(hidden_size=config['hidden_size'])
        fix_t = FixT(hidden_size=config['hidden_size'])

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


        # define optimizers
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        encoder_var_optimizer = torch.optim.Adam(encoder_var.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        predict_optimizer = torch.optim.Adam(predict.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        predict_output_optimizer = torch.optim.Adam(predict_output.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        generate_optimizer = torch.optim.Adam(generate.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        merge_optimizer = torch.optim.Adam(merge.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        num_x_predict_optimizer = torch.optim.Adam(num_x_predict.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        x_generate_optimizer = torch.optim.Adam(x_generate.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        x_to_q_optimizer = torch.optim.Adam(x_to_q.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        sementic_alignment_optimizer = torch.optim.Adam(sementic_alignment.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        num_or_opp_optimizer = torch.optim.Adam(num_or_opp.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        sni_optimizer = torch.optim.Adam(sni.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        fix_t_optimizer = torch.optim.Adam(fix_t.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

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

        # defiine schedulers
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
        start_epoch = 0


    # Move models to GPU
    for k,v in models.items():
        v.to(device)

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(start_epoch, config['n_epochs']):
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
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches, output_var_solutions, equation_targets, var_pos, batches_sni, pair_mapping, datasets = prepare_train_batch(train_pairs, config['batch_size'], vars, output_lang, input_lang)
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
            loss, acc, num_x_mse, comparison, op_right, sni_acc, loss_dict, acc_list, acc_soln = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models,
                output_lang, num_pos_batches[idx], equation_targets[idx], var_pos[idx], batches_sni[idx], pair_mapping[idx], output_var_solutions[idx], config['useCustom'], vars, config['setName'], config['useSemanticAlignment'], config['useSeperateVars'], config['useOpScaling'], config['useSNIMask'], config['useTFix'], datasets[idx], config['opsInNN'], True)
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

            batch_eval_comparison = []
            # for test_batch in test_pairs:
            input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_var_batches, output_var_solutions, equation_targets, var_pos, batches_sni, pair_mapping, datasets = prepare_train_batch(test_pairs, 1, vars, output_lang, input_lang)
            for idx in range(len(input_lengths)):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                for k, v in models.items():
                    v.eval()
                input_batch_len = len(input_batches[idx])
                start = time.perf_counter()
                loss, acc, num_x_mse, comparison, op_right, sni_acc, loss_dict, acc_list, acc_soln = train_tree( input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx], num_stack_batches[idx], num_size_batches[idx], output_var_batches[idx], generate_num_ids, models, output_lang, num_pos_batches[idx], equation_targets[idx], var_pos[idx], batches_sni[idx], pair_mapping[idx],output_var_solutions[idx], config['useCustom'], vars, config['setName'], config['useSemanticAlignment'], config['useSeperateVars'], config['useOpScaling'], config['useSNIMask'], config['useTFix'], datasets[idx], config['opsInNN'], False) 
                end = time.perf_counter()
                test_time_array.append([input_batch_len,end - start])
                # testc.append(comparison)
                batch_accuricies['eval_total_loss'] += loss
                batch_eval_comparison.append(comparison)
                batch_accuricies["eval_token"].append(acc)
                batch_accuricies["eval_op_right"].append(op_right)
                batch_accuricies["eval_num_x_mse"].append(num_x_mse)
                batch_accuricies["eval_sni_acc"].append(sni_acc)
                batch_accuricies['eval_loss_dict'].append(loss_dict)
                if acc_soln == 1:
                    batch_accuricies["eval_soln"].append(1)
                else:
                    batch_accuricies["eval_soln"].append(0)

            batch_loss = batch_accuricies['eval_total_loss'] / len(input_lengths)
            batch_eval_acc = sum(batch_accuricies["eval_token"]) / len(batch_accuricies["eval_token"])
            batch_eval_op_right = sum(batch_accuricies["eval_op_right"]) / len(batch_accuricies["eval_op_right"])
            batch_eval_num_x_mse = sum(batch_accuricies["eval_num_x_mse"]) / len(batch_accuricies["eval_num_x_mse"])
            batch_eval_sni_acc = sum(batch_accuricies["eval_sni_acc"]) / len(batch_accuricies["eval_sni_acc"])
            batch_eval_soln_acc = sum(batch_accuricies["eval_soln"]) / len(batch_accuricies["eval_soln"])
            print(epoch, 'batch eval soln', batch_eval_soln_acc)
            eval_comparison.append(batch_eval_comparison)

            print("loss:", batch_loss)
            print("eval accuracy", batch_eval_acc)

            fold_accuracies["eval_losses"].append(batch_loss)
            fold_accuracies["eval_token"].append(batch_eval_acc)
            fold_accuracies["eval_op_right"].append(batch_eval_op_right)
            fold_accuracies["eval_num_x_mse"].append(batch_eval_num_x_mse)
            fold_accuracies["eval_sni_acc"].append(batch_eval_sni_acc)
            fold_accuracies["eval_soln"].append(batch_eval_soln_acc)
            fold_accuracies["eval_loss_dict"].append(batch_accuricies['eval_loss_dict'])

            print("------------------------------------------------------")

            if (epoch + 1) % 5 == 0 and do_saves:
                save_epoch_state(save_folder, {
                    "models": models,
                    "optimizers": optimizers,
                    "schedulers": schedulers,
                    "fold_accuracies": fold_accuracies,
                    'epoch': epoch,
                })
                save_general_state(save_folder, {
                    'config' : config,
                    "pairs": pairs,
                    "all_train_accuracys": all_train_accuracys,
                    "all_train_loss": all_train_loss,
                    "all_eval_loss": all_eval_loss,
                    "all_eval_accuracys": all_eval_accuracys,
                    "all_soln_eval_accuracys": all_soln_eval_accuracys,
                    "train_comparison": train_comparison,
                    "eval_comparison": eval_comparison,
                    "total_training_time": total_training_time,
                    "total_inference_time": total_inference_time,
                    "train_time_array": train_time_array,
                    "test_time_array": test_time_array,
                    "existing_fold": existing_fold,
                    # "full_start": full_start,
                })
            saved_epoch_completed = True
    saved_fold_completed = True 
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
    write_comparison(train_comparison, eval_comparison)
    # print('fold accuracies', fold_accuracies)
    # make_loss_graph(
    #     fold_accuracies['loss'], 
    #     f"src/post/loss-{time.time()}-{run_id}.png", config['title'],
    #     "Epoch", "Loss By Epoch"
        # )
    make_eval_graph(
        [fold_accuracies["train_losses"], fold_accuracies["eval_losses"]], 
        ['Train', "Eval"],
        f"src/post/loss-{time.time()}-{run_id}-fold_{fold}.png", config['title'],
        "Epoch", "Loss By Epoch", None 
        )
    make_eval_graph(
        [fold_accuracies["train_token"], fold_accuracies["eval_token"]], 
        ['Train', "Eval"],
        f"src/post/accuracy-{time.time()}-{run_id}-fold_{fold}.png", config['title'],
        "Epoch", "Accuracy By Epoch", [0, 1]
        )
    print('fold train accuracy', fold_accuracies["train_token"])
    print('fold eval accuracy', fold_accuracies['eval_token'])
    print('All TRAIN ACC', all_train_accuracys)
    print('ALL EVAL ACC', all_eval_accuracys)
    print('ALL EVAL SOLN ACC', all_soln_eval_accuracys)
    process_loss_dicts(fold_accuracies['train_loss_dict'], fold_accuracies['eval_loss_dict'], f"src/post/loss-dict-{time.time()}-{run_id}-fold_{fold}.png")
    if config["num_folds"] == 2:
        break

    if do_saves:
        save_epoch_state(save_folder, {
            "models": models,
            "optimizers": optimizers,
            "schedulers": schedulers,
            "fold_accuracies": fold_accuracies,
            'epoch': epoch,
        })
        save_general_state(save_folder, {
            'config' : config,
            "pairs": pairs,
            "all_train_accuracys": all_train_accuracys,
            "all_train_loss": all_train_loss,
            "all_eval_loss": all_eval_loss,
            "all_eval_accuracys": all_eval_accuracys,
            "all_soln_eval_accuracys": all_soln_eval_accuracys,
            "train_comparison": train_comparison,
            "eval_comparison": eval_comparison,
            "total_training_time": total_training_time,
            "total_inference_time": total_inference_time,
            "train_time_array": train_time_array,
            "test_time_array": test_time_array,
            "existing_fold": existing_fold,
            # "full_start": full_start,
        })
        save_fold_state(save_folder, {
            "config": config,
            "generate_nums": generate_nums,
            "copy_nums": copy_nums,
            "vars": vars,
            "input_lang": input_lang,
            "output_lang": output_lang,
            'pairs_tested': pairs_tested,
            'pairs_trained': pairs_trained,
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "generate_num_ids": generate_num_ids,
            "fold": fold,
            "fold_pairs": fold_pairs,
            "fold_accuracies": fold_accuracies,
        })
    # if not do_folds:
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
# total_run_time = full_end - full_start
# print("total run time", total_run_time)
