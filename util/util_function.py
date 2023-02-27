import numpy as np
import torch
from torch.utils import data
from basic_parameter import batchsize, Num_grid

def sparse_degree(S):
    sparse = 0
    for i in range(S.shape[0]):
        sparse += torch.nonzero(S[i]).shape[0] / S.shape[1]
    return sparse / batchsize


def conjT(H):
    return np.conj(H.T)

class CustomDataset(data.Dataset):
    def __init__(self, label, Sinputs, channel, noise):
        self.Sinputs = Sinputs
        self.label = label
        self.channel = channel
        self.noise = noise
    def __getitem__(self, index):
        return self.Sinputs[index], self.label[index], self.channel[index], self.noise[index]
    def __len__(self):
        return self.Sinputs.shape[0]


def complex_matmul(*arg):
    if len(arg) < 2:
        raise BaseException
    else:
        real, imag = arg[0].real, arg[0].imag
        for m in arg[1:]:
            real2, imag2 = m.real, m.imag
            real_temp = torch.matmul(real, real2) - torch.matmul(imag, imag2)
            imag_temp = torch.matmul(real, imag2) + torch.matmul(imag, real2)
            real, imag = real_temp[:], imag_temp[:]
    output = torch.cat([real.unsqueeze(-1), imag.unsqueeze(-1)], dim=-1)
    return output


def test_xxxxx(S, label):
    S_now = torch.nonzero(S[0])
    label_nonzero = torch.nonzero(label)

    print("The S have nonzeros is", S_now[0].shape[0], "the label have nonzero is ", label_nonzero.shape[0])
    s_, labelllll = [], []
    for i, j in zip(S_now, label_nonzero):
        print("Splace", i, "Sdata", float(S[0][i[0], i[1]]), "\t", "Lplace", j, "Ldata", label[j[0], j[1]])

def example_label(S, label, index):
    S_now = torch.nonzero(S[0][index])
    label_nonzero = torch.nonzero(label[index].real)
    print(S_now)
    print(label_nonzero)
    print("The S have nonzeros is", S_now.shape[0], "the label have nonzero is ", label_nonzero.shape[0])
    s_, labelllll = [], []
    for i in S_now:
        print("Splace", i, "Sdata", float(S[0][index, i]))
    for j in label_nonzero:
        print(j)
        print( "Lplace", j, "Ldata", label[index, j])

    # correct = 0
    # distance = 0
    # for i in S_now:
    #     if i in label_nonzero:
    #         correct += 1
    #         distance += torch.abs(S[i[0],i[1]]-label_nonzero[i[0],i[1]])


def XXXtest_model_grad(model):
    grad = {}
    for name, value in model.named_parameters():
        grad[name] = value.grad
    return grad


def XXXtest_model(model):
    grad = {}
    for name, value in model.named_parameters():
        grad[name] = value
    return grad


def check_on_the_nonzeros(S):
    return torch.sort(torch.abs(S), dim=- 1, descending=True)