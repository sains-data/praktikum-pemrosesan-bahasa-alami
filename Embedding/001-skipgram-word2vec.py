"""
Hands-on Word2Vec Skip-gram — embedding kata statis (PyTorch).
Mata kuliah NLP Berbasis Transformer, Itera.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # Dua matriks: pusat-kata (W) dan konteks (W^T)
        self.W = nn.Parameter(-2 * torch.rand(vocab_size, embedding_size) + 1, requires_grad=True).type(dtype)
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, vocab_size) + 1, requires_grad=True).type(dtype)

    def forward(self, X):
        # X: [batch_size, vocab_size] one-hot
        hidden_layer = torch.matmul(X, self.W)
        output_layer = torch.matmul(hidden_layer, self.WT)
        return output_layer


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(vocab_size)[data[i][0]])
        random_labels.append(data[i][1])
    return random_inputs, random_labels


if __name__ == '__main__':
    sentences = [
        "i like dog",
        "i like cat",
        "i like animal",
        "dog cat animal",
        "apple cat dog like",
        "dog fish milk like",
        "dog cat eyes like",
        "i like apple",
        "apple i hate",
        "apple i movie book music like",
        "cat dog hate",
        "cat dog like"
    ]

    word_sequence = " ".join(sentences).split()
    word_list = list(set(word_sequence))
    vocab2id = {w: i for i, w in enumerate(word_list)}

    batch_size = 20
    embedding_size = 2
    vocab_size = len(vocab2id)

    # Pasangan (kata pusat, kata konteks) dengan jendela = 1
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = vocab2id[word_sequence[i]]
        context = [vocab2id[word_sequence[i - 1]], vocab2id[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        input_batch, target_batch = random_batch(skip_grams, batch_size)
        input_batch = Variable(torch.Tensor(input_batch))
        target_batch = Variable(torch.LongTensor(target_batch))

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, _ = model.parameters()
        x, y = float(W[i][0]), float(W[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title('Word2Vec Skip-gram (2D)')
    plt.show()
