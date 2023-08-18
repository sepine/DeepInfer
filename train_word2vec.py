import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
import logging

logger = logging.getLogger(__name__)

# BOS_TOKEN = "<BOS>"
# EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
# BOW_TOKEN = "<BOW>"
# EOW_TOKEN = "<EOW>"
UNK_TOKEN = '<UNK>'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batch size per GPU/CPU for training and test.")

    parser.add_argument("--lr", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Dimension of word2vec embeddings')

    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='Dimension of hidden layers')

    parser.add_argument('--model_save_path', type=str, default='dataset_return_vyper/',
                        help='Path to models.')

    parser.add_argument('--context_size', type=int, default=2,
                        help='Total number of training epochs to perform')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs to perform')

    return parser.parse_args()


def check_args(args):
    logger.info(vars(args))


WEIGHT_INIT_RANGE = 0.1

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# Parse args
args = parse_args()
check_args(args)


def load_cbow_data():
    sents = []

    with open('dataset_return_vyper/cbow_data.txt', 'r') as fr:
        for line in fr.readlines():
            sents.append(line.strip().split(' '))

    # vocab = Vocab.build(sents, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    vocab = Vocab.build(sents, reserved_tokens=[PAD_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(s) for s in sents]

    return corpus, vocab


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<UNK>" not in tokens:
                tokens = tokens + ["<UNK>"]

            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<UNK>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)  # 存储标记及其出现次数的映射字典
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        uniq_tokens = ["<UNK>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<UNK>"]

        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split("\n")

    return Vocab(tokens)


class CbowDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        # self.bos = vocab[BOS_TOKEN]
        # self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="CBOW Dataset Construction"):
            # sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 模型输入：左右分别取context_size长度的上下文
                context = sentence[i - context_size: i] + sentence[i + 1: i + context_size + 1]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)


def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]}{embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")

        print(f"Pretrained embeddings saved to :{save_path}")


def init_weights(model):
    for name, param in model.named_parameters():
        if "embedding" not in name:
            torch.nn.init.uniform_(
                param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE
            )


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = embeds.mean(dim=1)
        output = self.output(hidden)
        log_prob = F.log_softmax(output, dim=1)
        return log_prob


embedding_dim = args.embedding_dim
context_size = args.context_size
hidden_dim = args.hidden_dim
batch_size = args.batch_size
num_epoch = args.epochs
lr = args.lr

corpus, vocab = load_cbow_data()
dataset = CbowDataset(corpus, vocab, context_size=context_size)
data_loader = get_loader(dataset, batch_size)

nll_loss = nn.NLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CBOW(len(vocab), embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()

global_steps = 0

for epoch in range(num_epoch):
    total_loss = 0.
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        global_steps += 1

        if global_steps % 100 == 0:
            cur_loss = loss.item() / len(batch)
            print(f"Loss:{cur_loss:.2f}")

    print(f"Loss:{total_loss:.2f}")

save_path = 'dataset_return_vyper/word2vec_return_vyper.vec'

save_pretrained(vocab, model.embeddings.weight.data, save_path)
