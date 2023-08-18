import torch
import operator
from queue import PriorityQueue
from torch import nn
from torch.autograd import Variable
from encoder import TypeEncoder
from decoder import CopyNetDecoder, SimpleCopyNetDecoder
from utils import timeit
from utils import to_one_hot, to_np


class TypeEncoderDecoder(nn.Module):
    def __init__(self, opid2vec, opcode2idx, word2idx, args):
        super(TypeEncoderDecoder, self).__init__()

        self.args = args

        self.opid2vec = opid2vec
        self.opcode2idx = opcode2idx
        self.word2idx = word2idx
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim
        self.decoder_type = args.decoder_type
        self.vocab_size = len(self.word2idx.keys())

        self.encoder = TypeEncoder(opid2vec, opcode2idx, word2idx, self.hidden_size, self.embedding_dim, self.args)

        if self.decoder_type == 'simple':
            self.decoder = SimpleCopyNetDecoder(self.word2idx, self.args)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(self.word2idx, self.args)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

    def forward(self, dfs, values, targets=None, teacher_forcing=0.0):
        batch_size = dfs.shape[0]

        encoder_outputs = self.encoder(dfs, values)

        s = torch.zeros((batch_size, 1)).to(self.args.device)
        s[:, 0] = self.word2idx['<SEM>']
        values = torch.cat([s, values], dim=1)

        hidden = self.encoder.init_hidden(batch_size)
        decoder_outputs, sample_idxs = self.decoder(
            encoder_outputs, values, hidden, targets=targets, teacher_forcing=teacher_forcing)

        return decoder_outputs, sample_idxs

    def decode(self, dfs, values, trg, method='beam'):
        enc_output = self.encoder(dfs, values)
        batch_size = dfs.shape[0]
        seq_length = enc_output.data.shape[1]

        s = torch.zeros((batch_size, 1)).to(self.args.device)
        s[:, 0] = self.word2idx['<SEM>']
        inputs = torch.cat([s, values], dim=1)

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.args.device)

        if method == 'beam':
            return self.beam_decode(hidden, enc_output, inputs, seq_length)
        else:
            return self.greedy_decode(trg, hidden, enc_output)

    @timeit
    def beam_decode(self, decoder_hiddens, encoder_outputs, inputs, seq_length):

        beam_width = self.args.beam_width
        topk = self.args.topk  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(encoder_outputs.size(0)):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)

            selective_read = Variable(torch.zeros(1, 1, self.hidden_size))
            one_hot_input_seq = to_one_hot(inputs[idx].unsqueeze(0), self.vocab_size + seq_length)
            if next(self.parameters()).is_cuda:
                selective_read = selective_read.to(self.args.device)
                one_hot_input_seq = one_hot_input_seq.to(self.args.device)

            # Start with the start of the sentence token
            decoder_input_sos = torch.tensor([self.word2idx['<SOS>']], dtype=torch.long).to(self.args.device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input_sos, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            while True:
                if qsize > 1000:
                    break

                score, n = nodes.get()

                decoder_input = n.token_id
                decoder_input = decoder_input.view(1, decoder_input.size(0))
                decoder_hidden = n.h

                if n.token_id.item() == self.word2idx['<EOS>'] and n.prev_node != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                _, decoder_output, decoder_hidden, selective_read = self.decoder.step(decoder_input,
                                                                                      decoder_hidden,
                                                                                      encoder_output,
                                                                                      selective_read,
                                                                                      one_hot_input_seq)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):

                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_prob + log_p, n.length + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(to_np(n.token_id)[0])
                while n.prev_node != None:
                    n = n.prev_node
                    utterance.append(to_np(n.token_id)[0])

                utterance = utterance[::-1][:self.args.max_length_output]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hidden, prev_node, token_id, log_prob, length):
        self.h = hidden
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


