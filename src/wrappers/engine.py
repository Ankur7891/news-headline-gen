import torch
import torch.nn as nn
import torch.optim as optim


def tensor(x, dtype=None, device=None):
    return torch.tensor(x, dtype=dtype, device=device)


def zeros(shape, dtype=None, device=None):
    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None):
    return torch.ones(shape, dtype=dtype, device=device)


def zeros_like(x):
    return torch.zeros_like(x)


def ones_like(x):
    return torch.ones_like(x)


def arange(n, device=None):
    return torch.arange(n, device=device)


def cat(xs, dim=0):
    return torch.cat(xs, dim=dim)


def bmm(a, b):
    return torch.bmm(a, b)


def matmul(a, b):
    return torch.matmul(a, b)


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def log_softmax(x, dim=-1):
    return torch.log_softmax(x, dim=dim)


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.relu(x)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x)


def sqrt(x):
    return torch.sqrt(x)


def argmax(x, dim=-1):
    return torch.argmax(x, dim=dim)


def minv(a, b):
    return torch.min(a, b)


def clip(x, a, b):
    return torch.clamp(x, a, b)


def tril(x):
    return torch.tril(x)


def transpose(x, a, b):
    return x.transpose(a, b)


def sumv(x):
    return torch.sum(x)


def mean(x):
    return torch.mean(x)


def gather(x, dim, idx):
    return torch.gather(x, dim, idx)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(n):
    torch.manual_seed(n)


Module = nn.Module
ModuleList = nn.ModuleList
Linear = nn.Linear
Embedding = nn.Embedding
RNN = nn.RNN
LSTM = nn.LSTM
Dropout = nn.Dropout
LayerNorm = nn.LayerNorm

long = torch.long
float32 = torch.float32


def adam(params, lr):
    return optim.Adam(params, lr=lr)
