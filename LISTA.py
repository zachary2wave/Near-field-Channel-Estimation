import torch

from basic_parameter import *
from copy import deepcopy as DC
from util.util_function import sparse_degree
from data_generate import dic_tensor, sensing_matrix




class LISTA_c(nn.Module):
    "this vision does not utilize complex operation "
    "suit for the torch vision under than 1.8 "
    "the two thing is different with traditioanl method "
    '1. we set each layer share the same parameter'
    "2. we encoding the S, the We is only utilized as initial"


    def __init__(self, m, n, k, T=10, lambd=1):
        '''
        :param m:  the size of sparse signal
        :param n:  the size of obversination signal
        :param k:  the size of channel


        '''

        super(LISTA_c, self).__init__()
        self.m, self.n, self.k = m, n, k

        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.A = nn.Parameter(torch.ones((2, m, n), dtype=torch.float32), requires_grad=True)
        self.B = nn.Parameter(torch.ones((2, m, m), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.A)
        # ISTA Stepsizes eta = 1/L
        self.etas = nn.Parameter(1e-1*torch.ones(T + 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)


    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _T(self, H):
        return H.transpose(1,2)

    def forward(self, y):
        '''
        :param y: the obversation signal with the dim BN, n ,1
        :return:
               x: recoveried sparse signal with the dim BN, m ,1
        '''

        y = complex_feature_trasnsmission(y).transpose(1, 2)
        # B = C(self.We, y)    # the dim will be  BN, m, 1
        x = self._shrink(self.gammas[0, :, :] * C(self.A, y), self.etas[0, :, :])
        for i in range(1, self.T + 1):
            x = self._shrink(x - self.gammas[i, :, :] * C(self.B, x) +
                             self.gammas[i, :, :] * C(self.A, y), self.etas[i, :, :])
        return x.transpose(1, 2)
