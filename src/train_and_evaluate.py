# coding: utf-8
# import line_profiler

from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import matplotlib.pyplot as plt

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

# this equation keeps only operators 
def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # target[t] is the ACTUAL equation character at index t for each batch
    #    target[t] = 1 x num_batches
    # outputs is the strength of operators or a number token
    # num_stack_batch is the cooresponding num lists
    # num_start is where non-operators begin
    # unk is unknown token


    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    # for the token at that position in each batch
    for i in range(len(target)):
        # if not sure what token it is
        if target[i] == unk:
            # get the numbers that coorespond to the token
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            # for each cooresponding number
            for num in num_stack:
                # if the score of the number is higher than the max score
                if decoder_output[i, num_start + num] > max_score:
                    # set the target to the number
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        # if the token is NOT an operator, hide it 
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


# def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
#     # when the decoder input is copied num but the num has two pos, chose the max
#     if USE_CUDA:
#         decoder_output = decoder_output.cpu()
#     for i in range(target.size(0)):
#         if target[i] == unk:
#             num_stack = nums_stack_batch[i].pop()
#             max_score = -float("1e12")
#             for num in num_stack:
#                 if decoder_output[i, num_start + num] > max_score:
#                     target[i] = num + num_start
#                     max_score = decoder_output[i, num_start + num]
#     return target


# def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
#     # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
#     up_num_start = decoder_input >= nums_start
#     down_num_end = decoder_input < (nums_start + copy_nums)
#     num_mask = up_num_start == down_num_end
#     num_mask_encoder = num_mask < 1
#     num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
#     repeat_dims = [1] * num_mask_encoder.dim()
#     repeat_dims[1] = embedding_size
#     num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

#     all_embedding = encoder_outputs.transpose(0, 1).contiguous()
#     all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
#     indices = decoder_input - nums_start
#     indices = indices * num_mask.long()  # 0 or the num pos in sentence
#     indices = indices.tolist()
#     for k in range(len(indices)):
#         indices[k] = num_pos[k][indices[k]]
#     indices = torch.LongTensor(indices)
#     if USE_CUDA:
#         indices = indices.cuda()
#     batch_size = decoder_input.size(0)
#     sen_len = encoder_outputs.size(0)
#     batch_num = torch.LongTensor(range(batch_size))
#     batch_num = batch_num * sen_len
#     if USE_CUDA:
#         batch_num = batch_num.cuda()
#     indices = batch_num + indices
#     num_encoder = all_embedding.index_select(0, indices)
#     return num_mask, num_encoder, num_mask_encoder


# def out_equation(test, output_lang, num_list, num_stack=None):
#     test = test[:-1]
#     max_index = len(output_lang.index2word) - 1
#     test_str = ""
#     for i in test:
#         if i < max_index:
#             c = output_lang.index2word[i]
#             if c == "^":
#                 test_str += "**"
#             elif c == "[":
#                 test_str += "("
#             elif c == "]":
#                 test_str += ")"
#             elif c[0] == "N":
#                 if int(c[1:]) >= len(num_list):
#                     return None
#                 x = num_list[int(c[1:])]
#                 if x[-1] == "%":
#                     test_str += "(" + x[:-1] + "/100" + ")"
#                 else:
#                     test_str += x
#             else:
#                 test_str += c
#         else:
#             if len(num_stack) == 0:
#                 print(test_str, num_list)
#                 return ""
#             n_pos = num_stack.pop()
#             test_str += num_list[n_pos[0]]
#     return test_str


# def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
#     # print(test_res, test_tar)

#     if len(num_stack) == 0 and test_res == test_tar:
#         return True, True, test_res, test_tar
#     test = out_expression_list(test_res, output_lang, num_list)
#     tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
#     # print(test, tar)
#     if test is None:
#         return False, False, test, tar
#     if test == tar:
#         return True, True, test, tar
#     try:
#         if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
#             return True, False, test, tar
#         else:
#             return False, False, test, tar
#     except:
#         return False, False, test, tar



# def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
#     if len(num_stack) == 0 and test_res == test_tar:
#         return True, True
#     test = out_equation(test_res, output_lang, num_list)
#     tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
#     if test is None:
#         return False, False
#     if test == tar:
#         return True, True
#     try:
#         if abs(eval(test) - eval(tar)) < 1e-4:
#             return True, False
#         else:
#             return False, False
#     except:
#         return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    # encoder_outputs: max_len x num_batches x hidden_size 
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    # for each in batch
    for b in range(batch_size):
        # i = position of a num in input text
        for i in num_pos[b]:
            # prob going to flatten b into 1 d later.
            # append the index of the number in the input text after we do a flatten
            indices.append(i + b * sen_len)
            # mark this num pos as processed?
            masked_index.append(temp_0)
        # fill rest with 0s (0 = not a number from input text)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        # full rest with 1s ([1s] signify that these locations are not from the input text)
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    # size: b * num_size
    indices = torch.LongTensor(indices)
    # size: b * (num_size x hidden_size)
    masked_index = torch.ByteTensor(masked_index)
    # flatten and convert to mask:
    #   masked_index = b x num_size x hidden_size
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    masked_index = masked_index.bool()
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    # convert encoder format to be batch first to match mask
    # all_outputs: num_batches x max_length x hidden_size 
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    # flatten batches:
    # all_embedding (num_batches * max_length) x hidden_size
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H

    # grab the tokens where we have determined a number is
    all_num = all_embedding.index_select(0, indices)
    # break back into batches
    all_num = all_num.view(batch_size, num_size, hidden_size)
    # since num_size is the max num of numbers, and indicies length is from it, have indicies that
    # dont actually correspond to any numbers in the input text. so mask them
    # return num_batches x num_size x hidden_size that corresponds to the embeddings of the numbers
    return all_num.masked_fill_(masked_index, 0.0)

def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

# @line_profiler.profile
def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, output_var_batches, generate_nums, models, output_lang, num_pos, equation_targets,var_pos, useCustom, all_vars,  debug, setName, english=False):
    # input_batch: padded inputs
    # input_length: length of the inputs (without padding)
    # target_batch: padded outputs
    # target_length: length of the outputs (without padding)
    # num_stack_batch: the corresponding nums lists
    # num_size_batch: number of numbers from the input text
    # generate_nums: numbers to generate
    # num_pos: positions of the numbers lists



    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    problem_vars = torch.LongTensor(output_var_batches)
    target = torch.LongTensor(target_batch)#.transpose(0, 1)
    target_length = torch.LongTensor(target_length)
    if useCustom:
        equation_targets_tensor = torch.LongTensor(equation_targets)
    else:
        equation_targets_tensor = None

    # num vars total in the output lang. will need to mask ones not in the current equation
   # num_total_vars = len(problem_vars[0])
    non_expanded_lengths = []
    for i in target_length.tolist():
        non = [j for j in i if j != 0]
        non_expanded_lengths.append(non)
    #    non_expanded_lengths.append(target_length[i])
    num_equations_per_obs = torch.LongTensor([len(equ_set) for equ_set in non_expanded_lengths])
    num_total_vars = max(num_equations_per_obs)

    # sequence mask for attention
    # 0s where in input, 1s where not in input
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    # number mask 
    # 0s where the numbers are from input, 1s where not in input
    num_mask = []
    if useCustom:
        max_num_size = max(num_size_batch) + len(generate_nums) + len(all_vars) 
    else:
        max_num_size = max(num_size_batch) + len(generate_nums) 
    # in language its 
    # operators + gen numbers + vars + copy numbers
    for i, num_size in enumerate(num_size_batch):
        if useCustom:
            d = num_size + len(problem_vars[i].tolist()) + len(generate_nums)
            num_mask.append([0] * len(generate_nums) + problem_vars[i].tolist() + [0] * num_size + [1] * (max_num_size - d))
        else:
            d = num_size + len(generate_nums)
            num_mask.append([0] * len(generate_nums) + [0] * num_size + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(models['predict'].hidden_size)]).unsqueeze(0)


    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    batch_size = len(input_length)

    total_loss = None
    total_acc = [] 

    # Run words through encoder
    # embedding + dropout layer
    # encoder_outputs: num_batches x 512 q_0 vector
    # problem_output: max_length x num_batches x hidden_size
    encoder_outputs, problem_output = models['encoder'](input_var, input_length) 
    encoder_var_outputs, var_output = models['encoder_var'](input_var, input_length) 
    # Prepare input and output variables

    
    # array of len of numbers that must be copied from the input text
    copy_num_len = [len(_) for _ in num_pos]
    # max nums to copy
    num_size = max(copy_num_len)

    var_num_len = [len(_) for _ in var_pos]
    # max nums to copy
    var_size = max(var_num_len)

    # get the embeddings of the numbers
    # num_batches x num_size x hidden_size that correspond to the embeddings of the numbers
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, models['encoder'].hidden_size)



    # index in the language where the special (operators) tokens end and input/output text begins
    num_start = output_lang.num_start

    if useCustom:
        pred_num_equations = models['num_x_predict'](encoder_outputs)
        print()
    else:
        pred_num_equations = 0

    if useCustom:
        # node that max(num_equations_per_obs) should be the same as the lenth of vars
        # xs: batch_size x num_vars x hidden_size
        qs = models['q_generate'](len(all_vars), encoder_outputs, problem_output)
        # xs = get_all_number_encoder_outputs(encoder_outputs, var_pos, batch_size, var_size, models['encoder'].hidden_size)
        # xs = torch.zeros(batch_size, len(all_vars), 512)
    else: 
        qs = problem_output
    if useCustom:
        # qs: batch_size x num_vars x hidden_size
        xs = models['q_to_x'](encoder_var_outputs, qs, var_output)
    else:
        xs = None 

    # do equations one at a time
    for cur_equation in range(max(num_equations_per_obs)):
        # select the ith equation in each obs
        ith_equation_target = deepcopy(target[:, cur_equation, :].transpose(0,1))
        if useCustom:
            ith_equation_solution = deepcopy(equation_targets_tensor[:, cur_equation])
        else:
            ith_equation_solution = None
        ith_equation_target_lengths = deepcopy(target_length[:, cur_equation])
        # it_equation_solution 
        if useCustom:
            ith_equation_goal = qs[:, cur_equation, :]
            node_stacks = [[TreeNode(_)] for _ in ith_equation_goal.split(1, dim=0)]
        else:
            ith_equation_goal = problem_output
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        ith_equation_target_lengths = torch.Tensor(target_length)[:, cur_equation]
        ith_equation_num_stacks = []
        for stack in nums_stack_batch:
            ith_equation_num_stacks.append(stack[cur_equation])

        # max_target_length = int(max(ith_equation_target_lengths.tolist()))
        max_target_length = len(ith_equation_target)

        all_node_outputs = []
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        pred_equ_solutions = [None for _ in range(batch_size)]

        for t in range(max_target_length):

            # predict gets the encodings and embeddings for the current node 
            #   num_score: batch_size x num_length
            #       likliehood prediction of each number
            #   op: batch_size x num_ops
            #       likliehood of the operator tokens
            #   current_embeddings: batch_size x 1 x hidden_size
            #       goal vector (q) for the current node 
            #   current_context: batch_size x 1 x hidden_size
            #       context vector (c) for the subtree
            #   embedding_weight: batch_size x num_length x hidden_size
            #       embeddings of the generate and copy numbers

            num_score, op, current_embeddings, current_context, current_nums_embeddings = models['predict'](node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, xs, seq_mask, num_mask, useCustom, debug)


            # this is mainly what we want to train
            outputs = torch.cat((op, num_score), 1)

            # plt.plot(outputs[0].tolist())
            # plt.clf()
            # plt.imshow(outputs.detach().numpy())
            all_node_outputs.append(outputs)

            # target[t] is the equation character at index t for each batch
            #    target[t] = 1 x num_batches
            # outputs is the strength of operators or a number token
            # num_stack_batch is the cooresponding num lists
            # num_start is where non-operators begin
            # unk is unknown token
            # returns
            #   for position t in each equation
            #       target_t: actual equation value
            #       generate_input: equation value if its an operator
            target_t, generate_input = generate_tree_input(ith_equation_target[t].tolist(), outputs, ith_equation_num_stacks, num_start, unk)
            ith_equation_target[t] = target_t
            if USE_CUDA:
                generate_input = generate_input.cuda()

            # takes:
            #     generate a left and right child node with a label
            #     current_embeddings: q : batch_size x 1 x hidden_dim
            #     generate_input: [operator tokens at position t]
            #     current_context: c : batch_size x 1 x hidden_dim
            # returns
            #     l_child: batch_size x hidden_dim
            #          hidden state h_l:
            #     r_child: batch_size x hidden_dim
            #          hidden state h_r:
            #     node_label_ : batch_size x embedding_size 
            #          basically the context vector (c)
            # the node generation takes the first half of equations (10) and (11) 
            left_child, right_child, node_label = models['generate'](current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                node_stacks, ith_equation_target[t].tolist(), embeddings_stacks):
                # current_token = output_lang.ids_to_tokens([i])
                # current_equation = output_lang.ids_to_tokens(target.transpose(0,1)[idx])
                #print("at token", current_token, "in", current_equation)
                #print("current node_stack length", len(node_stack))
                # for 
                #   batch_num
                #   the left child: h_l 
                #   the right child: h_r
                if len(node_stack) != 0:
                    node = node_stack.pop()
                    #print("removed last from node_stack, now", len(node_stack), "elems")
                else:
                    left_childs.append(None)
                    continue

                # i is the num in language of where that specific language token is
                # if i is an operator
                if i < num_start:
                    #print(current_token, "is an operator, making a left and right node")
                    # make a left and right tree node
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    # save the embedding of the operator 
                    # terminal means a leaf node
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    #print("saving node embedding to o (non terminal node), and r, and l to node_stack. o now of size", len(o), "node_stack of size", len(node_stack))
                else:
                    #print(current_token, "is not an operator")
                    # otherwise its either a number from the input equation or a copy number
                    # we have a list (o) of the current nodes in the tree
                    # if we have a leaf node at the top of the stack, get it.
                    # next element in the stack must be an operator, so get it 
                    # and combine the new node, operator, and other element

                    # current_nums_embedding: batch_size x num_length x hidden_size
                    # current_num = num_embedding of the number selected
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    # while there are tokens in the embedding stack and the last element IS a leaf node
                    while len(o) > 0 and o[-1].terminal:
                        #print("terminal element in o, getting terminal element and operator, and merging")
                        # get the two elements from it
                        sub_stree = o.pop()
                        op = o.pop()
                        # contains equation (13)
                        # this combines a left and right tree along with a node
                        current_num = models['merge'](op.embedding, sub_stree.embedding, current_num)
                        #print('merged. o now of size', len(o))
                    # then re-add the node back to the stack
                    #print("adding current_num to o (terminal node)")
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    #print("terminal element in o, adding to left child")
                    # left_childs is a running vector of the sub tree embeddings "t" 
                    # need this for generation of the right q
                    left_childs.append(o[-1].embedding)
                    pred_equ_solutions[idx] = [o[-1]]
                else:
                    left_childs.append(None)


        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        
        # all_node_outputs:  for each token in the equation:
        #   the current scoring of nums for each batch
        # 
        # transform to 
        # all_node_outputs2: for each batch:
        #   the current scoring of nums for each token in equation
        # = batch_size x max_len x num_nums
        all_node_outputs2 = torch.stack(all_node_outputs, dim=1)  # B x S x N

        # batch_size x max_len x num_nums
        ith_equation_target = ith_equation_target.transpose(0, 1).contiguous()
        if USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs2 = all_node_outputs2.cuda()
            ith_equation_target = ith_equation_target.cuda()

        # for batch in target:
        #     print([output_lang.index2word[_] for _ in batch])
        #print('done equation')
        current_equation_loss = masked_cross_entropy(all_node_outputs2, ith_equation_target, ith_equation_target_lengths )
        # current_equation_loss = torch.nn.CrossEntropyLoss(reduction="none")(all_node_outputs2.view(-1, all_node_outputs2.size(2)), ith_equation_target.view(-1).to(device)).mean()

        
        # # batch_size x max_len 
        # as_equations = current_equation_loss_before.view(all_node_outputs2.size(0), -1)


        # # generate a mask for tokens that WE FILLED 
        # n = len(ith_equation_target_lengths)
        # max_length_t =len(as_equations[0])
        # index = torch.arange(max_length_t).unsqueeze(0).repeat(n, 1)  # Create a 2D index tensor with shape (n, max_length)
        # mask_t = (index < ith_equation_target_lengths.unsqueeze(1)).float()  # Expand lengths tensor & compare with index tensor, then cast to float

        # as_equations = as_equations * mask_t.to(device)
        # current_equation_loss = as_equations.view(-1).mean()
        
        # print(mask)

        # remove the lengths we dont care about

        same = 0
        lengths = 0
        # print(f'Equation {cur_equation}')
        for i, batch in enumerate(all_node_outputs2):
            # cur_len = 0
            # cur_same = 0
            equ_length = int(ith_equation_target_lengths[i].item())
            # vals = []
            for j, probs in enumerate(batch):
                if j < equ_length:
                    lengths += 1
                    # cur_len += 1
                    max_val = torch.argmax(probs)
                    # vals.append(max_val)
                    if max_val == ith_equation_target[i][j]:
                        same += 1
                        # cur_same += 1
            # print(f"        prediction: {[output_lang.index2word[_] for _ in vals[0:equ_length]]}")
            # print(f"        actual:     {[output_lang.index2word[_] for _ in ith_equation_target[i][0:equ_length]]}")
            # print('same', cur_same, 'length', cur_len)

        # for i, batch in enumerate(all_node_outputs2):
        #     vals = []
        #     # print('coming equ length', equ_length)
        #     for j, probs in enumerate(batch):
        #         if j == equ_length:
        #             break
        #         max_val = torch.argmax(probs)
        #         # lengths += 1
        #         # if max_val == ith_equation_target[i][j]:
        #         #     same += 1

        if useCustom and setName == "DRAW":
        # if False:
            # we masked some of the equations (they are 0s) so the model predicted all left nodes.
            # so in pred_equ_solutions they are all None
            # fill these with 0s. 
            for i in range(len(pred_equ_solutions)):
                if pred_equ_solutions[i] == None:
                    pred_equ_solutions[i] = [TreeEmbedding(torch.zeros(1, models['predict'].hidden_size).to(device), True)]
            num_score, op, current_embeddings, current_context, current_nums_embeddings = models['predict_output'](pred_equ_solutions, [None for i in range(len(pred_equ_solutions))], encoder_outputs, all_nums_encoder_outputs, padding_hidden, xs, seq_mask, num_mask, useCustom, debug)

            prediction = torch.cat((op, num_score), 1)

            tokenPredictions = prediction.argmax(dim = 1)
            for token, target_t in zip(tokenPredictions, ith_equation_solution):
                print(f'predicted: {output_lang.index2word[token.item()]} actual: {output_lang.index2word[target_t.item()]}')
                lengths += 1
                if token == target_t:
                    same += 1
            equation_prediction_loss = torch.nn.CrossEntropyLoss(reduction="none")(prediction, ith_equation_solution.to(device)).mean()
            if total_loss != None:
                total_loss += equation_prediction_loss
            else:
                total_loss = equation_prediction_loss

            # print('acc', cur_same/cur_len)
        if total_loss != None:
            total_loss += current_equation_loss
            # total_loss += current_equation_loss_before.mean()
        else:
            total_loss = current_equation_loss
            # total_loss = current_equation_loss_before.mean()
        total_acc += [same/lengths]
    
    # add the loss of number equations
    if useCustom:
    # if False:
        num_x_loss = torch.nn.CrossEntropyLoss()(pred_num_equations, num_equations_per_obs.to(device))
        total_loss += num_x_loss
        total_loss += equation_prediction_loss

        # predict a solution token for the tree 
        # actual_target = 
    total_loss.backward()

    # Update parameters with optimizers
    return total_loss.item(), sum(total_acc)/len(total_acc)

# @line_profiler.profile
def evaluate_tree(input_batch, input_length, generate_nums, models, input_lang, output_lang, num_pos, vars, useCustom, debug, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(1, input_length + len(vars)).fill_(0)
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch + [input_lang.word2index[i] for i in vars]).unsqueeze(1)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)


    # Set to not-training mode to disable dropout

    padding_hidden = torch.FloatTensor([0.0 for _ in range(models['predict'].hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        # num_mask = num_mask.cuda()

    # Run words through encoder
    # encoder_outputs, problem_output = models['encoder'](input_var, [input_length + len(vars)])
    encoder_outputs, problem_output = models['encoder'](input_var, [input_length])
    encoder_var_outputs, var_output = models['encoder_var'](input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size, models['encoder'].hidden_size)
    num_start = output_lang.num_start
    # B x P x N

    # predict number of xs
    if useCustom:
        num_x = models['num_x_predict'](encoder_outputs, eval=True).argmax().item()
    else:
        num_x = 1

    # if useCustom:
    #     # make random x vectors
    #     x_list = []
    #     x_list.append(torch.rand(2, 512))
    #     # # fill other as zero
    #     # x_list.append(torch.zeros(512))
    #     x_list = torch.stack(x_list)
    # else:
    #     x_list = None

    # get xs
    if useCustom:
        qs = models['q_generate'](num_x, encoder_outputs, problem_output)
        # var_pos = [[input_length + i for i in range(len(vars))]]
        # var_size = len(var_pos[0])
        # # xs = get_all_number_encoder_outputs(encoder_outputs, var_pos, batch_size, var_size, models['encoder'].hidden_size)
        # # padd the xs in the first dim to match the length of variables
        if (num_x) < len(vars):
            padding = torch.zeros(1, len(vars) - num_x, 512)
            qs = torch.cat((qs, padding.to(device)), dim=1).to(device)
        if num_x > len(vars):
            qs = qs[:, :len(vars), :]
    else: 
        qs = problem_output

    # get qs
    if useCustom:
        xs = models['q_to_x'](encoder_var_outputs, qs, var_output)
        # xs = torch.zeros(1, len(vars), 512)
    else:
        xs = None


    # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)
    num_mask = []
    # max_num_size = num_pos + len(generate_nums) + len(vars) 
    # in language its 
    # operators + gen numbers + vars + copy numbers
    if useCustom:
        num_mask.append([0] * len(generate_nums) + [0] * min(num_x, len(vars)) + [1] * (max(len(vars) - num_x, 0)) + [0] * num_size)
    else:
        num_mask.append([0] * len(generate_nums)  +  [0] * num_size )
    num_mask = torch.ByteTensor(num_mask).to(device)


    final_beams = []

    for i in range(num_x):
        # get the node stacks
        if useCustom:
            ith_equation_goal = qs[:, i, :]
            node_stacks = [[TreeNode(_)] for _ in ith_equation_goal.split(1, dim=0)]
        else:
            ith_equation_goal = problem_output
            node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        # evaulation uses beam search
        # key is how the beams are compared
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [] )]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = models['predict']( b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, xs, seq_mask, num_mask, useCustom, debug)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                # topv:
                #   largest elements in the out_score
                # topi:
                #   indexes of the largest elements 
                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start

                # for the largest element, and its index
                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    # the predicted token is that of the highest score relation
                    out_token = int(ti)
                    # save token 
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    # if the predicted token is an operator
                    if out_token < num_start:
                        # this is the token to generate l and r from
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        # get the left and right children and current label
                        left_child, right_child, node_label = models['generate'](current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        # predicted token is a number
                        # get the token embedding - embedding of either the generate num or copy num
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                        
                        # if we are a right node (there is a left node and operator)
                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = models['merge'](op.embedding, sub_stree.embedding, current_num)
                        # save node (or subtree) to the embeddings list
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)

                    # the beam "score" is the sum of the associations 
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out ))
            # order beam by highest to lowest
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        num_score_temp, op_temp, _, _, _ = models['predict_output']([beams[0].node_stack[-1]], [None for i in range(len([beams[0].node_stack[-1]]))], encoder_outputs, all_nums_encoder_outputs, padding_hidden, xs, seq_mask, num_mask, useCustom, debug)
        possible_tokens = torch.cat((op_temp, num_score_temp), 1)
        final_beams.append([beams[0].out, possible_tokens.argmax(dim=1)])
    return final_beams
