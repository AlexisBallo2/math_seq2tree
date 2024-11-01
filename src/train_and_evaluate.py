# coding: utf-8

from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


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
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        # if the token is NOT an operator, hide it 
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = [out_expression_list(equ, output_lang, num_list) for equ in test_res ]
    tar = [out_expression_list(equ, output_lang, num_list, copy.deepcopy(num_stack)) for equ in test_tar]
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



def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


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


def train_tree(input_batch, input_length, target_batch, target_mask, target_length, target_equation_sonls, nums_stack_batch, num_size_batch, var_tokens_batch, generate_nums,
               encoder, num_x_predict, x_generate, x_to_q, predict, generate, merge, encoder_optimizer, num_x_predict_optimizer, x_generate_optimizer, x_to_q_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, all_vars, english=False):
    # input_batch: padded inputs
    # input_length: length of the inputs (without padding)
    # target_batch: padded outputs
    # target_length: length of the outputs (without padding)
    # num_stack_batch: the corresponding nums lists
    # num_size_batch: number of numbers from the input text
    # generate_nums: numbers to generate

    # num_pos: positions of the numbers lists


    # # get max lengths of each equation and number of equations in the batch
    # max_equations_per_problem = 0
    # max_tokens_per_equation = 0
    # for batch in input_batch:
    #     for input_seq, input_length_temp, equations, equation_lengths, input_nums, input_nums_pos, num_stack in batch:
    #     # for equation in output_length:
    #         max_equations_per_problem = max(max_equations_per_problem, len(equations))
    #         for equation in equations:
    #             max_tokens_per_equation = max(max_tokens_per_equation, len(equation))
    
    # print("max equations per problem: ", max_equations_per_problem)
    # print('max tokens per equation: ', max_tokens_per_equation)


    # sequence mask for attention
    # 0s where in input, 1s where not in input
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)


    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (equation x max_len x batch_size ) tensors, 
    print("input batch", input_batch)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    print("input var", input_var)
    print('target batch', target_batch)
    target = torch.stack([torch.LongTensor(equation_set) for equation_set in target_batch], dim=-1)
    print('target', target)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    num_x_predict.train()
    x_generate.train()
    x_to_q.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        # num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    num_x_predict_optimizer.zero_grad()
    x_generate_optimizer.zero_grad()
    x_to_q_optimizer.zero_grad()
    # Run words through encoder

    # embedding + dropout layer
    # encoder_outputs: num_batches x 512 q_0 vector
    # problem_output: max_length x num_batches x hidden_size
    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    

    max_target_length = 0
    max_num_equations = 0
    for pair_equations_lengths in target_batch:
        max_target_length = max(max_target_length, *[len(equation) for equation in pair_equations_lengths])
        max_num_equations = max(max_num_equations, len(pair_equations_lengths))
    # for pair_equations_lengths in target_length:
    #     max_target_length = max(max_target_length, *pair_equations_lengths)
    #     max_num_equations = max(max_num_equations, len(pair_equations_lengths))
    # max_target_length = max(target_length)
    # for equation in input_batch
    # max_num_equations = max()

    # all_leafs = []

    # array of len of numbers that must be copied from the input text
    copy_num_len = [len(_) for _ in num_pos] 
    # max nums to copy
    num_size = max(copy_num_len)

    # num_batches x num_size x hidden_size that correspond to the embeddings of the numbers
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    # get the number of x's to generate
    num_x = num_x_predict(encoder_outputs)
    # generate the x vectors
    # target_length is batch_size x num_equations x num_tokens
    # actual_x = torch.Tensor([max_num_equations for _ in target_length])

    gen_xs = []

    print()
    # number of variables to generate
    target_ordering_of_vars = all_vars
    temp_encoder = encoder_outputs.transpose(0, 1)

    batch_all_vars = []
    empty = torch.zeros(encoder.hidden_size)
    for batch_num, batch_vars in enumerate(var_tokens_batch):
        cur_bach_vars = []
        xs = x_generate(len(batch_vars), temp_encoder[batch_num], all_nums_encoder_outputs[batch_num], problem_output[batch_num])
        # get the generated variables
        for variable in target_ordering_of_vars:
            match = False
            for var, x in zip(batch_vars, xs):
                if var == variable:
                    cur_bach_vars.append(x)
                    match = True
            if not match:
                cur_bach_vars.append(empty)
        batch_all_vars.append(torch.stack(cur_bach_vars))
    all_vars_embs = torch.stack(batch_all_vars)

    qs = x_to_q(temp_encoder, all_vars_embs)
            # for x_emb, x in zip(xs, vars):

            # gen_xs.append(xs)

    # these are the xs generated:
    # num variables x hidden_size
    # gen_xs = torch.cat(gen_xs)
    # qs, updated_encoder_outputs, updated_nums_encoder_outputs = xq_generate(actual_x, encoder_outputs, all_nums_encoder_outputs, problem_output)

    # number mask 
    # 0s where the numbers are from input, 1s where not in input
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums) + len(gen_xs)
    for i in num_size_batch:
        d = i + len(generate_nums) + len(gen_xs)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)
    
    # for each batch, if there are variables in output lang [x,y,z] but group only has [x,y], mask would be [0,0,1]
    var_mask = []
    # first get variables in output lang 
    # var_list = [output_lang.variables for _ in target_batch]
    # for each equation group
    for i, equ_group in enumerate(target_batch):
        group_var_mask = []
        unique_in_equation = [output_lang.index2word[i] for i in list(set([el for arr in equ_group for el in arr]))]
        for var in output_lang.variables:
            # decoded = output_lang.index2word[var]
            if var in unique_in_equation:
                group_var_mask.append(0)
            else:
                group_var_mask.append(1)
        var_mask.append(group_var_mask)
    var_mask = torch.ByteTensor(var_mask)
    addedQs = len(qs)
    # make sure to note that the updated encoder has the x vectors


    # index in the language where the special (operators) tokens end and input/output text begins
    num_start = output_lang.num_start
    # 
    all_outputs = []
    for equation_count in range(max_num_equations):
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        current_equation_outputs = []
        current_equation = target[equation_count]

        # make a TreeNode for each token
        # node just has embedding and a left flag? 
        tempSplit = problem_output.split(1, dim=0)
        # problem_output is q_0 for each token in equation, so use last one
        node_stacks = [[TreeNode(_.unsqueeze(0))] for _ in qs.transpose(0,1)[equation_count]]
        
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

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask, var_mask, all_vars_embs)


            # this is mainly what we want to train
            outputs = torch.cat((op, num_score), 1)
            current_equation_outputs.append(outputs)

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
            # target_t, generate_input = generate_tree_input(target[equation_count][t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target_t, generate_input = generate_tree_input(current_equation[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            # target[t] = target_t
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
            left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                node_stacks, target_t, embeddings_stacks):
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
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        #print('merged. o now of size', len(o))
                    # then re-add the node back to the stack
                    #print("adding current_num to o (terminal node)")
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    #print("terminal element in o, adding to left child")
                    # left_childs is a running vector of the sub tree embeddings "t" 
                    # need this for generation of the right q
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        # tree has been built. now predict the = " " token
        # num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
        #     node_stacks, left_childs, encoder_outputs, updated_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
        # final_token = torch.cat((op, num_score), 1)
        # current_equation_outputs.append(final_token)

        all_outputs.append(torch.stack(current_equation_outputs, dim = 1))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    
    # all_node_outputs:  
    #  for each equation 
    #    for each token in equation: 
    #      the current scoring of nums for each batch
    # 
    # transform to 
    # all_node_outputs2: for each batch:
    #   the current scoring of nums for each token in equation
    # = batch_size x max_len x num_nums
    print('all', all_outputs)
    all_outputs_stacked = torch.stack(all_outputs, dim=-1)
    
    # all_outputs = torch.stack(all_outputs, dim=1)  # B x S x N
    # all_outputs = torch.cat(all_outputs, dim=1)  # B x S x N

    # target analysis
    # target: num_equations x num_tokens x batch_size
    temp_target = target.transpose(1,2).transpose(0,1)
    # print("target")
    # for i in range(temp_target.size(0)):
    #     print('batch', i)
    #     for j in range(temp_target.size(1)):
    #         # print(temp_target[i][j])
    #         print('equation', j, ":", [output_lang.index2word[_] for _ in temp_target[i][j]])
    # # predictions
    # print("predictions")
    # all_outputs2 = [_.unsqueeze(0) for _ in all_outputs]
    # all_outputs_combined = torch.cat(all_outputs2, dim=0)
    # all_outputs_combined2 = all_outputs_combined.transpose(0,1)
    # for i in range(all_outputs_combined2.size(0)):
    #     print('batch', i)
    #     for j in range(all_outputs_combined2[i].size(0)):
    #         print(all_outputs_combined2[i][j])
    #         # print('equation', j, ":", [output_lang.index2word[_] for _ in all_outputs[i][j].max(1)[1]])

    # for i in range(len(all_outputs)):
        # print('batch', i)
        # for j in range(all_outputs[i].size(0)):
        #     print(all_outputs[i][j].max(1))
            # print('equation', j, ":", [output_lang.index2word[_] for _ in all_outputs[i][j].max(1)[1]])
    # for i in range(all_outputs.size(0)):
    #     print('batch', i)
    #     for j in range(all_outputs[i].size(0)):
    #         print(all_outputs[i][j])
    #         # print('equation', j, ":", [output_lang.index2word[_] for _ in all_outputs[i][j].max(1)[1]])


    target2 = target.transpose(0, 1).contiguous()
    target3 = target2.transpose(1, 2).contiguous()
    target4 = target3.transpose(0,1).contiguous()

    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs2 = all_outputs.cuda()
        target = target.cuda()

    #print('done equation')
    target_length_filled = []
    for problem in target_length:
        current_num_equations = len(problem)
        if current_num_equations < max_num_equations:
            problem += [0 for _ in range(max_num_equations - current_num_equations)]
        target_length_filled.append(problem)
    target_length_filled = torch.Tensor(target_length_filled)
    # loss = masked_cross_entropy(all_node_outputs2, target, target_length)
    for i in range(max_num_equations):
        equation_target = target4[..., i]
        # for actuals in equation_target:
            # print(actuals)
            # print([output_lang.index2word[_] for _ in actuals])
        predictions = all_outputs_stacked[..., i]
        length = target_length_filled[..., i]
        loss = masked_cross_entropy(predictions, equation_target, length.tolist())
        print()
    # loss = masked_cross_entropy(all_outputs_stacked, target, target_length)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item() 


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, x_generate, x_to_q, num_x_predict, merge, output_lang, num_pos, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    seq_mask = []
    max_len = input_length
    # for i in input_length:
    seq_mask = [0 for _ in range(max_len)]
    seq_mask = torch.ByteTensor(seq_mask)


    input_var = torch.LongTensor(input_batch).transpose(-1, 0)
    # target = torch.stack([torch.LongTensor(equation_set) for equation_set in target_batch], dim=-1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = 1

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    num_x_predict.eval()
    x_generate.eval()
    x_to_q.eval()
    

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    # batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    # encoder_outputs, problem_output = encoder(input_var, [input_length])
    # TODO: make sure input var moves correctly
    encoder_outputs, problem_output = encoder(input_var.unsqueeze(0).transpose(0,1), [input_length])

    max_target_length = 0
    max_num_equations = 3

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    copy_num_len = [len(num_pos)] 
    num_size = len(num_pos)
    # num_size = 1
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    # all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
    num_x = num_x_predict(encoder_outputs, eval = True)
    num_to_gen = math.ceil(num_x.item())

    temp_encoder = encoder_outputs.transpose(0, 1)
    #                                                           encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]



    batch_all_vars = []
    empty = torch.zeros(encoder.hidden_size)
    cur_bach_vars = []
    xs = x_generate(num_to_gen, temp_encoder[0], all_nums_encoder_outputs[0], problem_output[0])

    # get the generated variables
    # for variable in target_ordering_of_vars:
    #     match = False
    #     for var, x in zip(batch_vars, xs):
    #         if var == variable:
    #             cur_bach_vars.append(x)
    #             match = True
    #     if not match:
    #         cur_bach_vars.append(empty)
    # batch_all_vars.append(torch.stack(cur_bach_vars))
    all_vars_embs = torch.stack(xs).unsqueeze(0)

    qs = x_to_q(temp_encoder, all_vars_embs)

    # empty = torch.zeros(encoder.hidden_size)

    # evaulation uses beam search
    # key is how the beams are compared
    output = []

    for num_x in range(num_to_gen):
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, None, None, all_vars_embs)

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
                        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

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
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        # save node (or subtree) to the embeddings list
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    # the beam "score" is the sum of the associations 
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,current_left_childs, current_out))
            # order beam by highest to lowest
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        output.append(beams[0].out)

    return output