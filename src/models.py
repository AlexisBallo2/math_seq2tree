# coding: utf-8

from numpy import append
import torch
import torch.nn as nn


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
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
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
            score = score.masked_fill_(num_mask.bool(), -1e12)
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
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # input_seqs: max_len x batch_size
        # max_len comes from longest in the batch
        # embedded = max_len x num_batches x embedding_dim
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
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


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
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

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
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
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)


        # batch_size x 1 x hidden_dim
        current_node = torch.stack(current_node_temp)
        # this is the q of the current subtree
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
        embedding_weight = torch.cat((embedding_weight1, num_pades), dim=1)  # B x O x N


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

        # get the predicted operation (classification)
        # batch_size x num_ops 
        op = self.ops(leaf_input)

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
        return num_score, op, current_node, current_context, embedding_weight


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
        node_label = self.em_dropout(node_label_)

        # squeeze embedding and context for each
        # node_embedding: batch_size x hidden_dim
        # current_context: batch_size x hidden_dim
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)


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

        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, hidden):
        # hidden will be a list of unknown length with embed dimension of 512. 

        zeroList = [[[0.0] * 512] for _ in range(self.batch_size)]
        padding_tensor = torch.tensor(zeroList)  # or any other values you want to pad with
        padding_tensor = padding_tensor.squeeze(1)
        num_padding_needed = 100 - hidden.size(0)
        padding = padding_tensor.unsqueeze(0).expand(num_padding_needed, -1, -1)
        hidden2 = torch.cat((hidden, padding), dim=0)

        hidden2T = hidden2.transpose(0, 1)



        # pad this list to be 100 long
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.lstm.num_layers, hidden2T.size(0), self.lstm.hidden_size).to(hidden2T.device)
        c0 = torch.zeros(self.lstm.num_layers, hidden2T.size(0), self.lstm.hidden_size).to(hidden2T.device)

        # Forward propagate LSTM
        out, _ = self.lstm(hidden2T, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :]).squeeze(-1)  # out: tensor of shape (batch_size, output_size)
        return out


class XToQ(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.5):
        super(XToQ, self).__init__()
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, hidden, x):

        finals = []

        hidden2 = hidden.transpose(0,1)
        for i in range(hidden.shape[1]):
            xs = x[i]
            qs = []
            for j in range(len(xs)):
                qkt = torch.matmul(xs[j], self.K(hidden2[i]).transpose(0,1))
                smqkt = nn.functional.softmax(qkt)
                output = torch.matmul(smqkt, self.V(hidden2[i]))
                qs.append(output)
            qs = torch.stack(qs)
            # having more than 1 q means we need a q that covers all q/xs
            if len(qs) > 1:
                h0 = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(qs.device)
                c0 = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(qs.device)

                # Forward propagate LSTM
                out1, _ = self.lstm(qs, (h0, c0))  # out: tensor of shape (seq_length, hidden_size)
                qs = torch.cat((qs, out1[0].unsqueeze(0)), dim=0)
            finals.append(qs)
            
        output = torch.stack(finals)
        return output


class GenerateXQs(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout=0.5):
        super(GenerateXQs, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.em_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size * 10, 1)

        self.new_old_final = nn.Linear(hidden_size * 2, hidden_size)

        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        self.x_to_q = XToQ(hidden_size, output_size, batch_size, dropout=0.5)

    def forward(self, num_xs, hidden, num_encoder_outputs, problem_q):
        
        # hidden2: batch_size x tokens x hidden_size

        hidden2 = hidden.transpose(0,1)
        output_xs = []

        # for each in batch
        for i in range(hidden.shape[1]):
            xs = []
            nums_to_gen = max(int(num_xs.tolist()[i]), 1)
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
                    outAttention = torch.matmul(smqkt, v)
                    xs.append(outAttention)
            output_xs.append(torch.stack(xs))

            print('here')




        final_xs = torch.stack(output_xs)
        final_qs = self.x_to_q(hidden, final_xs)

        # add the xs to the hidden
        # xs: batch_size x num_xs x hidden_size
        # hidden2: batch_size x tokens x hidden_size
        updated_hidden = torch.cat((hidden2, final_xs), dim=1)
        updated_nums = torch.cat((num_encoder_outputs, final_xs), dim=1)
        return final_qs, updated_hidden.transpose(0,1), updated_nums
        # hidden will be a list of unknown length with embed dimension of 512. 



