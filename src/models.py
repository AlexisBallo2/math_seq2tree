# coding: utf-8
# import line_profiler

import torch
import torch.nn as nn
from torch_kmeans import KMeans
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            # attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
            attn_energies = attn_energies.masked_fill_(seq_mask, 0)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden1, num_embeddings, num_mask=None):
        # hidden1 is the concatenation [q c]
        # num_embeddings is the embedding weights and number encodings
        #   batch_size x num_length x hidden_dim
        # max_len is max of num_lengths
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden1.dim()
        repeat_dims[1] = max_len
        # repeat the leaf embeddings across 
        # hidden: batch_size x 1 x 2*hidden_dim
        # ->
        # batch_size x num_length x 2*hidden_dim
        hidden = hidden1.repeat(*repeat_dims)  # B x O x H


        # For each position of encoder outputs
        # batch_size
        this_batch_size = num_embeddings.size(0)

        # batch_size x num_length x (2*hidden_dim + hidden_dim)
        # energy_in1 = [q c e(y|P)] but e(y|P) for number tokens is just h^p for each num
        energy_in1 = torch.cat((hidden, num_embeddings), 2)
        # input size is the length of numbers that must be generated
        # hidden size is a constant
        energy_in = energy_in1.view(-1, self.input_size + self.hidden_size)

        # this is equation (7)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            # score = score.masked_fill_(num_mask.bool(), -1e12)
            score = score.masked_fill_(num_mask.bool(), 0)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden2 = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden2, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            # attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), 0)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, useBertEmbeddings = False, input_lang = None, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        model_name = 'bert-base-uncased'
        self.input_lang = input_lang
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # Text to embed
        # text = "This is a sample sentence."

        # Tokenize input text
        # encoded_input = tokenizer(text, return_tensors='pt')
        self.useBertEmbeddings = useBertEmbeddings

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def getEmbeddings(self, input_seqs):
            output = []
            input_per_batch = input_seqs.transpose(0, 1)
            for each_batch in input_per_batch:
                string = [self.input_lang.index2word[i] for i in each_batch.tolist()]
                input_ids = [self.tokenizer.encode(word, add_special_tokens=True) for word in string]
                input_ids_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True)
                with torch.no_grad():
                    outputs = self.model(input_ids_padded)
                last_hidden_states = outputs.last_hidden_state

                # To get a single embedding per word, typically the first token's (CLS) vector of each word's encoding is used.
                embeddings = [state[0] for state in last_hidden_states]
                embs = torch.stack(embeddings)
                output.append(embs)
            stacked_output = torch.stack(output)
            # make original shape: batch_size x max_len x hidden_size
            final = stacked_output.transpose(0, 1)
            return final



    def forward(self, input_seqs, input_lengths, hidden=None):
        # input_seqs: max_len x batch_size
        # max_len comes from longest in the batch
        # embedded = max_len x num_batches x embedding_dim
        # 32 x 20 x 128
        # embedded1 = self.embedding(input_seqs)  # S x B x E
        if self.useBertEmbeddings:
            embedded1 = self.getEmbeddings(input_seqs)
        else:
            embedded1 = self.embedding(input_seqs)
        # embedded1 = self.tokenizer(input_seqs, return_tensors='pt')
        embedded = self.em_dropout(embedded1)
        # packed = lengths x embedding
        # with multiple batches it seems to concatenate the padded 
        # sequences of variable length
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # convert to RNN form
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)


        # initial hidden state for gru
        pade_hidden = hidden
        # pass equation through GRU
        # this is equation (1) AND (2) (bidirectional GRU)
        # pade_outputs: max_len x num_batches x (2 (bidirectional) * hidden_size)
        #   these are the values from the last layer in the GRU
        # pade_hidden: (2 (bidirectional) * num_layers) x num_batches x hidden_size
        pade_outputs1, pade_hidden = self.gru_pade(packed, pade_hidden)


        # convert the GRU packed format back into dense tensor form
        # pade_outputs: max_len x num_batches x (2 (bidirectional) * hidden_size) 
        pade_outputs2, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs1)
        # pade_outputs2 = torch.ones(input_seqs.size(0), input_seqs.size(1), self.hidden_size * 2)#.to(pade_outputs2.device)
        

        # problem_output = 
        #   embedding (512) of the last GRU unit (from forward gru) 
        #   +
        #   embedding (512) of the first GRU unit (from backward gru)
        # this is equation (4)
        # num_batches x hidden_size
        problem_output = pade_outputs2[-1, :, :self.hidden_size] + pade_outputs2[0, :, self.hidden_size:]

        # pade_outputs = 
        #  forward GRU embeddings + backward GRU embeddings
        # max_length x num_batches x hidden_size
        # token embedding of each 
        pade_outputs = pade_outputs2[:, :, :self.hidden_size] + pade_outputs2[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


class TokenIrrevalant(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.5):
        super(TokenIrrevalant, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.em_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 2 , 1)
    def forward(self, hidden, goal_vect):
        # hidden = batch_size x num embeddings x hidden_size 
        # hidden = self.em_dropout(hidden)

        # g_expanded = goal_vect.unsqueeze(0).unsqueeze(0).expand(hidden.shape[0], hidden.shape[1], hidden.shape[2])
        g_expanded = goal_vect.unsqueeze(1).repeat(1, hidden.shape[1], 1)

        # Concatenate q to hidden along the hidden_size dimension
        hidden_with_g = torch.cat((hidden, g_expanded), dim=2)
        # concat goal_vect to every hidden
        concatted = torch.sigmoid(self.out(hidden_with_g))
        return concatted

class NumOrOpp(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(NumOrOpp, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.em_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 2)
        self.out2 = nn.Linear(hidden_size * 2, 3)
        self.padding_hidden = torch.FloatTensor([0.0 for _ in range(hidden_size)])

        self.k = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.upper = nn.Parameter(torch.randn(1, hidden_size))
    def forward(self, encoder_outputs, goal_vect, equation_goal):
        # batch size x hidden_size, 
        squeezed_goals = goal_vect.squeeze(1)
        concatted = torch.concat((squeezed_goals, equation_goal), 1)
        out = torch.relu(self.out2(concatted))
        return out
        # return self.out(squeezed_goals)
        # encoder_outputs: batch_size x hidden_size
        # to batch_size x tokens x hidden_size
        enc_outputs = encoder_outputs.transpose(0, 1)
        # for each batch
        outs = []
        for i in range(enc_outputs.size(0)):
            # for each token
            # q,k,v: num_tokens x hidden_size
            # q = self.q(enc_outputs[i])
            q = squeezed_goals.repeat(enc_outputs.size(1), 1)
            k = self.k(enc_outputs[i])
            v = self.v(enc_outputs[i])
            qk = torch.matmul(q, k.transpose(0, 1))
            smkq = nn.functional.softmax(qk)
            smkqv = torch.matmul(smkq, v)
            subbed = smkqv - self.upper
            # out = torch.sigmoid(torch.matmul(smkq, v))
            # add across the tokens
            # batch_kqvs = torch.stack(batch_kqvs)
            # summed = torch.sum(out, dim=0)
            summed = torch.sum(subbed, dim=0)
            fc = torch.sigmoid(self.out(summed))
            outs.append(fc)
            # outs.append(out)
        stacked = torch.stack(outs)
        return stacked



        # current_embeddings1 = []
        # # for each stack of tokens 
        # # 2 batches = 2 stacks
        # for st in node_stacks:
        #     # not sure why would be zero. it's initialized w/ single num each 
        #     # if it is zero, let that token's embedding be the zero embedding
        #     if len(st) == 0:
        #         current_embeddings1.append(self.padding_hidden)
        #     else:
        #         # use embedding from the last node in the stack
        #         current_node = st[-1]
        #         current_embeddings1.append(current_node.embedding.squeeze())

        # print()
        # current_embeddings = torch.stack(current_embeddings1)
        out = self.out(squeezed_goals)

        return out 




class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, num_vars, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.var = nn.Linear(hidden_size * 2, num_vars)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

        self.irr = TokenIrrevalant(hidden_size, 2, dropout)

    # @line_profiler.profile  
    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, xs, seq_mask, mask_nums, useCustom, debug, useSeperateVars,all_q, useVarsAsNums):
        # node_stacks: [TreeNodes] for each node containing the hidden state for the node
        # left_childs: [] of 
        # encoder_outputs: token embeddings: max_len x num_batches x hidden state 
        # num_pads:  embeddings of the numbers: num_batches x num_size x hidden_size 
        # padding_hidden: 0s of hidden_size
        # seq_mask: num_batches x seq_len 
        # mask_nums: 0s where the num in the nums emb come from input text. 
        # this aligns with the num_pades. num_batches x num_size

        # let num_length = 2 + len(numbers)

        current_embeddings1 = []

        # for each stack of tokens 
        # 2 batches = 2 stacks
        for st in node_stacks:
            # not sure why would be zero. it's initialized w/ single num each 
            # if it is zero, let that token's embedding be the zero embedding
            if len(st) == 0:
                current_embeddings1.append(padding_hidden)
            else:
                # use embedding from the last node in the stack
                current_node = st[-1]
                current_embeddings1.append(current_node.embedding)
        
        # current_embeddings1 = the embedding of the last node in the stack

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings1):
            # l = left child
            # c = embedding of parent node
            # second half of equation (10)
            if l is None:
                # if not left child (this is a leaf)
                # nodes context vector 1 x hidden_dim
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                # second half of equation (11)
                ld = self.dropout(l)
                # ld = l
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)


        # batch_size x 1 x hidden_dim
        current_node = torch.stack(current_node_temp)
        # this is the q of the current subtree
        current_embeddings = current_node # self.dropout(current_node)
        current_embeddings = self.dropout(current_node)

        # this gets the score calculation and relation between each 
        # encoded token
        # this is the a in equation (6)
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        # the encoder_outputs are the h

        # this is the context vector 
        # equation (6)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)


        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        # self.embedding_weight:
        #   1 x 2 x hidden_dim
        #   this is the amount to weight the each hidden dim for 2 things
        #       maybe the current context, and the leaf context? 


        # repeat the embedding weight for each batch
        # batch_size x 2 x hidden_dim
        embedding_weight1 = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N

        # batch_size x (2 + number of numbers we have encodings for) x hidden_dim
        # batch_size is the embeddings of the numbers
        #   batch_size x nums_count x hidden_dim
        if useCustom and useSeperateVars and useVarsAsNums:
            # embedding_weight = torch.cat((embedding_weight1, num_pades), dim=1)  # B x O x N
            # embedding_weight = torch.cat((embedding_weight1, xs, num_pades), dim=1)  # B x O x N
             embedding_weight = torch.cat((xs, embedding_weight1, num_pades), dim=1)  # B x O x N
            #  embedding_weight = torch.cat((xs, embedding_weight1, num_pades), dim=1)  # B x O x N
        else:
            embedding_weight = torch.cat((embedding_weight1, num_pades), dim=1)  # B x O x N



        # # get if the tokens are relevant to the problem
        # relavelant = self.irr(embedding_weight, all_q)
        # repeated = relavelant.repeat(1, 1, self.hidden_size)
        # embedding_weight = embedding_weight * repeated 
        # # print()



        # get the embedding of a leaf
        # embedding of a leaf is the concatenation of 
        #   the node embedding and the embedding after attention
        # this is the [q c]
        # batch_size x 1 x 2*hidden_dim 
        leaf_input1 = torch.cat((current_node, current_context), 2)
        # batch_size x 2*hidden_dim 
        leaf_input = leaf_input1.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        # embedding_weight_ = embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)


        # get the scores between the leaves,  
        # leaf_input is the embedding of a leaf
        # TODO
        # embedding_weight is the weight matrix of scaling the input hidden dims? (need to conform this is true)
        # mask_nums is batch_size x max_nums_length and is 0 where the num comes from text
        # equation (7) done here
        # this is the log likliehood of generating token y from the specified vocab
        #   doesnt seem like the full vocab is being used, only

        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        # plt.imshow(num_score.detach().cpu().numpy())
        # plt.clf()

        # get the predicted operation (classification)
        # batch_size x num_ops 
        op = self.ops(leaf_input)
        if useVarsAsNums:
            var = None
        else:
            var = self.var(leaf_input)
        

        # return p_leaf, num_score, op, current_embeddings, current_attn

        # this returns
        #   num_score: batch_size x num_length
        #       score values of each number
        #   op: operation list
        #       batch_size x num_ops
        #   current_node: q : batch_size x 1 x hidden_size
        #       goal vector for the subtree
        #   current_context: c : batch_size x 1 x hidden_size
        #       context vector for the subtree
        #   embedding_weight : batch_size x num_length x hidden_size
        #       number embeddings
        return num_score, op, var, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        # node_embedding: q : batch_size x 1 x hidden_dim
        # node_label: [operator tokens at position t]
        # current_context: c : batch_size x 1 x hidden_dim

        # the operators will all be embedded the same way
        node_label_ = self.embeddings(node_label)
        # node_label = node_label_
        node_label = self.em_dropout(node_label_)

        # squeeze embedding and context for each
        # node_embedding: batch_size x hidden_dim
        # current_context: batch_size x hidden_dim
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        # current_context = self.em_dropout(current_context)


        # first half of equation (10)

        # C_l = tanh( W_cl [q c e(\hat y | P)] ) 
        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        # o_l = sigmoid( W_ol [q c e(\hat y | P)] ) 
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        # h_l = o_1 * C_l
        l_child = l_child * l_child_g


        # first half of equation (11)
        # C_r = tanh( W_cr [q c e(\hat y | P)] ) 
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        # o_r = sigmoid( W_or [q c e(\hat y | P)] ) 
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        # h_r = o_r * C_r
        r_child = r_child * r_child_g

        # l_child: hidden state h_l:
        #   batch_size x hidden_dim
        # r_child: hidden state h_r:
        #   batch_size x hidden_dim
        # node_label_ : embedding of the operator
        #   batch_size x embedding_size 
        return l_child, r_child, node_label_

        # equation (11)

class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        # this is equation (13)
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class PredictNumX(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.5):
        super(PredictNumX, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.em_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 10, 1)

        self.lstm = nn.LSTM(hidden_size, hidden_size, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 4)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, goal_vect, eval = False):
        # goal_vect = self.em_dropout(goal_vect)
        temp = self.fc1(goal_vect)
        temp2 = self.relu(temp)
        temp3 = self.fc2(temp2)

        mask = torch.tensor([0, 1, 1, 1]).to(device)

        temp4 = self.relu(temp3) * mask
        out = self.softmax(temp4) * mask
        return out 

        # hidden will be a list of unknown length with embed dimension of 512. 

        # if eval == False:
        #     zeroList = [[[0.0] * 512] for _ in range(hidden.size(1))]
        # else:
        #     zeroList = [[[0.0] * 512] for _ in range(1)]
        # padding_tensor = torch.tensor(zeroList)  # or any other values you want to pad with
        # padding_tensor = padding_tensor.squeeze(1)
        # num_padding_needed = 100 - hidden.size(0)

        # # Ensure num_padding_needed is positive
        # if num_padding_needed > 0:
        #     padding = padding_tensor.unsqueeze(0).expand(num_padding_needed, -1, -1).to(device)
        #     hidden2 = torch.cat((hidden, padding), dim=0)
        # else:
        #     # If padding is not needed, use the original tensor
        #     hidden2 = hidden

        # hidden2T = hidden2.transpose(0, 1)



        # # pad this list to be 100 long
        # # Initialize hidden and cell states with zeros
        # h0 = torch.zeros(self.lstm.num_layers * 2, hidden2T.size(0), self.lstm.hidden_size).to(hidden2T.device)
        # c0 = torch.zeros(self.lstm.num_layers * 2, hidden2T.size(0), self.lstm.hidden_size).to(hidden2T.device)

        # # Forward propagate LSTM
        # # out: batch_size x max_tokens x hidden size
        # out, _ = self.lstm(hidden2T, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # # pass last token through feedforward nn
        # final_token_emb = out[:, -1, 512:] + out[:, -1, :512]
        # first_token_emb = out[:, 1, :512] + out[:, 1, 512:]
        # emb = torch.cat((final_token_emb.to(device), first_token_emb.to(device)), dim = -1)
        # out = self.fc(emb).squeeze(-1)  # out: tensor of shape (batch_size, output_size)
        # # mask the first token (dont want to predict 0 xs)
        # # out[:, 0] = -1e12
        # out[:, 0] = 0
        # softmax = torch.nn.Softmax(dim=-1)
        # return softmax(out)
        # return abs(out) 

class GenerateXs(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.5):
        super(GenerateXs, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.em_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 10, 1)

        self.new_old_final = nn.Linear(hidden_size * 2, hidden_size)

        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        # self.oneK = KMeans(n_clusters=1)
        self.twoK = KMeans(n_clusters=2)
        self.threeK = KMeans(n_clusters=3)



    def forward(self, num_xs, hidden, problem_q):
        
        # hidden2: batch_size x tokens x hidden_size

        hidden2 = hidden.transpose(0,1)

        # for batch in hidden2:
        #     # if num_xs < 2:
        #         # kmeans = self.oneK
        #     elif num_xs < 3:
        #         kmeans = self.twoK
        #     else:
        #         kmeans = self.threeK
        #     result = kmeans(batch)
            # print(result)


        # for each in batch
        out = []
        for i in range(hidden2.shape[0]):
            xs = []
            nums_to_gen = num_xs
            # nums_to_gen = max(int(num_xs.tolist()), 1)
            # goal_vect = self.em_dropout(problem_q[i])
            goal_vect = problem_q[i]
            kt = self.K(hidden2[i]).transpose(0,1)
            v = self.V(hidden2[i])
            # for each number to gen
            for j in range(nums_to_gen):
                # leave the first vector
                if len(xs) == 0:
                    xs.append(goal_vect)
                else:
                    # generate the next one from the attention of previous
                    qkt = torch.matmul(xs[j-1], kt)
                    smqkt = nn.functional.softmax(qkt)
                    # output: hidden_size
                    # outAttention = torch.sigmoid(torch.matmul(smqkt, v))
                    outAttention = torch.matmul(smqkt, v)
                    xs.append(outAttention)
            out.append(torch.stack(xs))
        final = torch.stack(out)
        return final
        # return xs


class XToQ(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(XToQ, self).__init__()
        self.em_dropout = nn.Dropout(dropout)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, hidden, x, problem_q):
        # x = batch_size x num_xs x hidden_size
        # hidden = batch_size x tokens x hidden_size

        finals = []

        # hidden2 = batch_size x tokens x hidden_size
        hidden2 = hidden.transpose(0,1)

        # for each in batch
        for i in range(hidden2.shape[0]):
            # get the x's for this batch
            xs = x[i]
            qs = []
            # for each x
            for j in range(len(xs)):
                # # for each x for the batch
                # # xk^T
                qkt = torch.matmul(xs[j], self.K(hidden2[i]).transpose(0,1))
                # qkt = self.em_dropout(qkt)
                smqkt = nn.functional.softmax(qkt)
                output = torch.sigmoid(torch.matmul(smqkt, self.V(hidden2[i])))
                # qs.append(output)
                # output_zeros = torch.zeros(output.size()).to(output.device)
                qs.append(output)
                # qs.append(output_zeros)
                # qs.append(problem_q[i])
            # need to change later
            qs = torch.stack(qs)#.transpose(0,1)
            finals.append(qs)
            # having more than 1 q means we need a q that covers all q/xs
            # if len(qs) > 1:
            #     h0 = torch.zeros(1, self.lstm.num_layers, self.lstm.hidden_size).to(qs.device)
            #     c0 = torch.zeros(1, self.lstm.num_layers, self.lstm.hidden_size).to(qs.device)

            #     # Forward propagate LSTM
            #     out1, _ = self.lstm(qs, (h0, c0))  # out: tensor of shape (seq_length, hidden_size)
            #     qs = torch.cat((qs, out1[0].unsqueeze(0)), dim=0)
            # finals.append(qs)
                
        output = torch.stack(finals)
        return output



# Attention:
class Attn2(nn.Module):
    def __init__(self, hidden_size, batch_first=False, bidirectional_encoder=True):
        super(Attn2, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional_encoder = bidirectional_encoder
        if self.bidirectional_encoder:
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.attn = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        if self.batch_first:  # B x S x H
            max_len = encoder_outputs.size(1)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[1] = max_len
        else:  # S x B x H
            max_len = encoder_outputs.size(0)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[0] = max_len
        # batch_first: False S x B x H
        # batch_first: True B x S x H
        hidden = hidden.repeat(*repeat_dims)  # Repeats this tensor along the specified dimensions

        # For each position of encoder outputs
        if self.batch_first:
            batch_size = encoder_outputs.size(0)
        else:
            batch_size = encoder_outputs.size(1)
        # (B x S) x (2 x H) or (S x B) x (2 x H)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1 or (B x S) x 1
        attn_energies = attn_energies.squeeze(1)  # (S x B) or (B x S)
        if self.batch_first:
            attn_energies = attn_energies.view(batch_size, max_len)  # B x S
        else:
            attn_energies = attn_energies.view(max_len, batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        attn_energies = self.softmax(attn_energies)
        return attn_energies.unsqueeze(1)


class Seq2TreeSemanticAlignment(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, hidden_size, batch_first=False, bidirectional_encoder=True):
        super(Seq2TreeSemanticAlignment, self).__init__()
        self.batch_first = batch_first
        self.attn = Attn2(encoder_hidden_size,batch_first=batch_first,bidirectional_encoder=bidirectional_encoder)
        self.encoder_linear1 = nn.Linear(encoder_hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)

        self.decoder_linear1 = nn.Linear(decoder_hidden_size, hidden_size)
        self.decoder_linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self,  decoder_hidden, encoder_outputs):
        # print(decoder_hidden.size())
        # print(encoder_outputs.size())
        if self.batch_first:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(0)
        else:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(1)
        attn_weights = self.attn(decoder_hidden, encoder_outputs, None)
        if self.batch_first:
            align_context = attn_weights.bmm(encoder_outputs) # B x 1 x H
        else:
            align_context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H
            align_context = align_context.transpose(0,1)

        encoder_linear1 = torch.tanh(self.encoder_linear1(align_context))
        encoder_linear2 = self.encoder_linear2(encoder_linear1)

        decoder_linear1 = torch.tanh(self.decoder_linear1(decoder_hidden))
        decoder_linear2 = self.decoder_linear2(decoder_linear1)

        return encoder_linear2, decoder_linear2



class SNI(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(SNI, self).__init__()

        self.lstm = nn.LSTM(input_size=512, hidden_size = 512, num_layers=1, batch_first=True, bidirectional=False)
        self.h0 = torch.randn(1, 512, hidden_size )
        self.c0 = torch.randn(1, 512, hidden_size)

        self.classifyer = torch.nn.Linear(512, 2)


    def forward(self, encoder_states):
        # encoder_states: 7 x 512
        encoder_states = encoder_states.unsqueeze(0)
        # out = self.lstm(encoder_states, (self.h0, self.c0))
        out, _ = self.lstm(encoder_states)
        final_token_emb = torch.relu(out[:, -1])
        classified = self.classifyer(final_token_emb)

        return classified

        print()



class FixT(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(FixT, self).__init__()

        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.goal_encoder = nn.Linear(hidden_size, hidden_size)
        self.t_encoder = nn.Linear(hidden_size, hidden_size)

        self.goal_forward = nn.Linear(hidden_size, hidden_size)
        self.emb_fowarard = nn.Linear(hidden_size, hidden_size)
        self.new_gen = nn.Linear(hidden_size * 2, hidden_size)

        self.encoder_linear1 = nn.Linear(hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)

        self.attn = Attn2(hidden_size,batch_first=True,bidirectional_encoder=False)
        self.dropout = nn.Dropout(dropout)
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.generate_l = nn.Linear(hidden_size * 2, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self,  ith_goal, t_embs, encoder_outputs, goal_vect_global):
        stacked = torch.stack(t_embs)
        l_child = torch.tanh(self.generate_l(torch.cat((ith_goal, stacked), 1)))
        # o_l = sigmoid( W_ol [q c e(\hat y | P)] ) 
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((ith_goal, stacked), 1)))
        # h_l = o_1 * C_l
        l_child = l_child * l_child_g
        c = self.dropout(l_child)
        g = torch.tanh(self.concat_l(c))
        t = torch.sigmoid(self.concat_lg(c))
        out = g * t
        return out





        # attn_weights = self.attn(ith_goal, encoder_outputs, None)
        # # if self.batch_first:
        # align_context = attn_weights.bmm(encoder_outputs) # B x 1 x H
        # encoder_linear1 = torch.tanh(self.encoder_linear1(align_context))
        # encoder_linear2 = self.encoder_linear2(encoder_linear1)
        # return encoder_linear2



        # t_vects = []
        # for item in t_embs:
        #     if item is None:
        #         t_vects.append(torch.zeros(512))
        #     else:
        #         t_vects.append(item[0].embedding.squeeze(0))
        #     # item[0].embedding = self.linear(item[0].embedding.squeeze(0))
        # # t_vects = [item[0].embedding.squeeze(0) for item in t_embs]
        # stacked = torch.stack(t_vects)
        # stack_forward = torch.relu(self.goal_forward(ith_goal))
        # t_forward = torch.relu(self.t_encoder(stacked))
        
        # # concatted = torch.cat((ith_goal, stacked), 1)
        # concatt = torch.cat((t_forward, stack_forward), 1)
        # out = self.new_gen(concatt)
        # encoder_states: 7 x 512
        # goal_enc = self.goal_encoder(goal_vect)
        # t_enc = self.t_encoder(t)
        # concatted = torch.cat((goal_vect, t), 1)
        # out = self.linear(concatted)

        # return goal_vect, t
        # return goal_enc, t_enc
        # return out 
        return ith_goal 