import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
def Normalization(object):
    def __init__(self, x, eps = 1e-5):
        # super(Normalization, self).__init__()

        self.mean = np.mean(x)
        self.std = np.mean(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x


def Unit_Normalization(object):
    def __init__(self, x, eps = 1e-5):
        # super(Normalization, self).__init__()

        self.mean = np.mean(x.reshape(-1))
        self.std = np.mean(x.reshape(-1))
        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.reshape(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x * (self.std + self.eps)) + self.mean
        x = x.reshape(s)
        return x
"""


class Normalization(object):
    def __init__(self, x, eps=0.00001):
        super(Normalization, self).__init__()

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()