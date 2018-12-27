import numpy as np
from collections import defaultdict


class word2vec:
    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, settings, corpus):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.vocab_size = len(word_counts.keys())
        self.words_list = list(word_counts.keys)
        self.w2id = dict((word, i) for i, word in enumerate(self.words_list))
        self.id2w = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.w2onehot(sentence[i])
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j >= 0 and j <= sent_len - 1:
                        w_context.append(self.w2onehot(sentence[j]))
                training_data.append([w_target, w_context])

    def w2onehot(self, word):
        w2v = np.zeros(self.vocab_size)
        widx = self.w2id[word]
        w2v[widx] = 1
        return w2v

    def train(self, training_data):
        self.w1 = np.random.normal(-1, 1, (self.vocab_size, self.n))
        self.w2 = np.random.normal(-1, 1, (self.n, self.vocab_size))

        for i in range(self.epochs):
            self.loss = 0

            for w_t, w_c in training_data:
                y_pred, h, u = self.forward(w_t)
                l = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(l, h, w_t)
                self.loss += -np.sum([u[word.index(1)]
                                      for word in w_c]) + len(w_c) * np.log(
                                          np.sum(np.exp(u)))

    def forward(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        y = self.softmax(u)
        return y, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    def word_vec(self, word):
        return self.w1[self.w2id[word]]
