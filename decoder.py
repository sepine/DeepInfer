import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import to_one_hot, DecoderBase


class CopyNetDecoder(DecoderBase):
    def __init__(self, word2idx, args):
        super(CopyNetDecoder, self).__init__()

        self.args = args
        self.hidden_size = self.args.hidden_size
        self.embedding_dim = self.args.embedding_dim
        self.word2idx = word2idx
        self.reserved_vocab_size = args.default_vocab_len
        self.vocab_size = len(self.word2idx.keys())  # a total of 15 vocabs (0 - 14)

        self.embed = nn.Embedding(len(self.word2idx.keys()),
                                      self.embedding_dim,
                                      padding_idx=self.word2idx['<PAD>'])
        self.embed.weight.data.normal_(0, 1 / self.embedding_dim ** 0.05)
        self.embed.weight.data[self.word2idx['<PAD>'], :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        # input = (context + selective read size + embedding)
        self.gru = nn.GRU(2 * self.hidden_size + self.embed.embedding_dim,
                          self.hidden_size, batch_first=True)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # self.out = nn.Linear(self.hidden_size, self.reserved_vocab_size)

    def forward(self, encoder_outputs,    # B x L x dim
                inputs,    # B x L
                final_encoder_hidden,     # 2 x B x dim
                targets=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]   # B
        seq_length = encoder_outputs.data.shape[1]   # L

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))   # 1 x B x dim
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.vocab_size + seq_length)))  # B x (L + seq<64>)
        # sos_output = Variable(torch.zeros((batch_size, self.vocab_size)))
        sampled_idx = Variable(torch.ones((batch_size, 1)).long())    # B x 1
        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()

        sos_output[:, self.word2idx['<SOS>']] = 1.0  # index of the <SOS> token, one-hot encoding

        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]

        if keep_prob < 1.0:
            dropout_mask = (Variable(torch.rand(
                batch_size, 1, 2 * self.hidden_size + self.embed.embedding_dim)) < keep_prob).float() / keep_prob
        else:
            dropout_mask = None

        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))       # B x 1 x dim
        one_hot_input_seq = to_one_hot(inputs, self.vocab_size + seq_length)   # B x (L + seq)
        # one_hot_input_seq = to_one_hot(inputs, self.vocab_size)   # B x (L + seq)
        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_input_seq = one_hot_input_seq.cuda()

        for step_idx in range(1, self.args.max_length_output):
            if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
                # replace some inputs with the targets (i.e. teacher forcing)
                # B x 1
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()

                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1: step_idx])

            sampled_idx, output, hidden, selective_read = self.step(
                sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs,
             prev_selective_read, one_hot_input_seq, dropout_mask=None):

        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        # vocab_size = len(self.word2idx.keys())

        # ## Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden)
        transformed_hidden = transformed_hidden.view(batch_size, self.hidden_size, 1)  # B x dim x 1
        # reduce encoder outputs and hidden to get scores.
        # remove singleton dimension from multiplication.

        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)   # B x L x dim * B x dim x 1 => B x L x 1
        attn_weights = F.softmax(attn_scores, dim=1)      # B x L x 1
        # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)

        # B x 1 x L * B x L x dim => B x 1 x dim     <attn among all encoder inputs>
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)

        # ## Call the RNN
        # [b, 1] bools indicating which seqs copied on the previous step
        out_of_vocab_mask = prev_idx >= self.reserved_vocab_size  # > self.vocab_size
        unks = torch.ones_like(prev_idx).long() * self.word2idx['<UNK>']
        # replace copied tokens with <UNK> token before embedding
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)
        # embed input (i.e. previous output token)
        embedded = self.embed(prev_idx)

        # B x 1 x dim | B x 1 x dim | B x 1 x dim
        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()

            rnn_input *= dropout_mask

        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, prev_hidden)  # B x 1 x dim

        # ## Copy mechanism
        transformed_hidden_2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)  # B x dim x 1
        # this is linear. add activation function before multiplying.
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden_2)  # B x L x 1
        # [b, 1, vocab_size + seq_length] * B x L x 1
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)
        # tokens not present in the input sequence
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)
        # <PAD> tokens are not part of any sequence
        missing_token_mask[:, self.word2idx['<PAD>']] = 1
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # ## Generate mechanism
        gen_scores = self.out(output.squeeze(1))  # [b. vocab_size]
        gen_scores[:, self.word2idx['<PAD>']] = -1000000.0  # penalize <PAD> tokens in generate mode too

        # ## Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        # gen_probs = probs[:, :self.reserved_vocab_size]
        gen_probs = probs[:, :self.vocab_size]

        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        # gen_padding = Variable(torch.zeros(batch_size, self.vocab_size - self.reserved_vocab_size))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()

        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]

        # copy_probs = probs[:, self.reserved_vocab_size:]
        copy_probs = probs[:, self.vocab_size:]

        final_probs = gen_probs + copy_probs

        log_probs = torch.log(final_probs + 10 ** -10)

        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # ## Create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_socres_norm = F.normalize(selected_scores, p=1)

        selective_read = (selected_socres_norm * encoder_outputs).sum(dim=1).unsqueeze(1)

        return sampled_idx, log_probs, hidden, selective_read


class SimpleCopyNetDecoder(DecoderBase):
    def __init__(self, word2idx, args):
        super(SimpleCopyNetDecoder, self).__init__()

        self.args = args
        self.hidden_size = self.args.hidden_size
        self.embedding_dim = self.args.embedding_dim
        self.word2idx = word2idx
        self.reserved_vocab_size = 15
        self.vocab_size = len(self.word2idx.keys())  # a total of 15 vocabs (0 - 14)

        self.embed = nn.Embedding(len(self.word2idx.keys()),
                                  self.embedding_dim,
                                  padding_idx=self.word2idx['<PAD>'])
        self.embed.weight.data.normal_(0, 1 / self.embedding_dim ** 0.05)
        self.embed.weight.data[self.word2idx['<PAD>'], :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.embedding_dim, self.hidden_size, batch_first=True)

        self.out = nn.Linear(self.hidden_size, len(self.word2idx.keys()))

    def forward(self, encoder_outputs,
                inputs, final_encoder_hidden,
                targets=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))  # overwrite the encoder hidden state with zeros
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size, self.embed.num_embeddings)))
        sos_output[:, self.word2idx['<SOS>']] = 1.0  # index 1 is the <SOS> token, one-hot encoding
        sos_idx = Variable(torch.ones((batch_size, 1)).long())

        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sos_idx = sos_idx.cuda()

        decoder_outputs = [sos_output]
        sampled_idxs = [sos_idx]

        iput = sos_idx

        dropout_mask = torch.rand(batch_size, 1, self.hidden_size + self.embed.embedding_dim)
        dropout_mask = dropout_mask <= keep_prob
        dropout_mask = Variable(dropout_mask).float() / keep_prob

        for step_idx in range(1, self.args.max_length_output):

            if targets is not None and teacher_forcing > 0.0:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) <= teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                iput = iput.masked_scatter(teacher_forcing_mask, targets[:, step_idx-1:step_idx])

            output, hidden = self.step(iput, hidden, encoder_outputs, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            _, topi = decoder_outputs[-1].topk(1)
            iput = topi.view(batch_size, 1)
            sampled_idxs.append(iput)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, dropout_mask=None):

        batch_size = prev_idx.shape[0]
        vocab_size = self.vocab_size

        # encoder_output * W * decoder_hidden for each encoder_output
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        scores = torch.bmm(encoder_outputs, transformed_hidden).squeeze(2)  # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(attn_weights, encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)

        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * self.word2idx['<UNK>']
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)  # replace copied tokens with <UNK> token before embedding

        embedded = self.embed(prev_idx)  # embed input (i.e. previous output token)

        rnn_input = torch.cat((context, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)

        output = self.out(output.squeeze(1))  # linear transformation to output size
        output = F.log_softmax(output, dim=1)  # log softmax non-linearity to convert to log probabilities

        return output, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result
