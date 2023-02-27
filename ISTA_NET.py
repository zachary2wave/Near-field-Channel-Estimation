
import torch.nn

from basic_parameter import *
from util.util_function import sparse_degree
from data_generate import dic_tensor




class ISTA_Net(nn.Module):
    "this vision does not utilize complex operation "
    "suit for the torch vision under than 1.8 "
    "the two thing is different with traditioanl method "
    '1. we set each layer share the same parameter'
    "2. we encoding the S, the We is only utilized as initial"
    def __init__(self, m, n, k, W, T=10, lambd=1):
        '''
        :param m:  the size of sparse signal
        :param n:  the size of obversination signal
        :param k:  the size of channel
        :param P_Phi: the Wd size is obversation signal,sparse

        :param T:  total training time
        :param lambd:
        :param D:
        '''
        self.m, self.n, self.k = m,n,k
        super(ISTA_Net, self).__init__()
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        # # simple ISTA
        self.WX = nn.Parameter(torch.ones((2, k, n), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.WX)
        # estimated

        self.EncodeW1 = nn.ParameterList(
            [nn.Parameter(torch.ones((2, m, k), dtype=torch.float32), requires_grad=True) for _ in range(T + 1)])

        self.DecodeW1 = nn.ParameterList(
            [nn.Parameter(torch.ones((2, n, m), dtype=torch.float32), requires_grad=True) for _ in range(T + 1)])

        for w1, w2 in zip(self.EncodeW1, self.DecodeW1):
            nn.init.xavier_normal_(w1)
            nn.init.xavier_normal_(w2)

        # ISTA Stepsizes
        self.etas = nn.Parameter(1e-4*torch.ones(T + 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)

        self.W = W
    def _linear(self, H):
        for w,b in zip(self.Wlinaer, self.Blinaer):
            H = F.relu(C(w, H) + b)
        return H

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _function(self, X, A, B):
        temp1 = F.relu(C(A, X))
        return C(B,temp1)

    def _function_compress(self, X, W1):
        X = C(W1, X)
        return X

    def _function_recovery(self, X, W1):
        a = W1[0].mm(X[0]).unsqueeze(0)
        b = W1[1].mm(X[0]).unsqueeze(0)
        return torch.cat([a,b], dim=0)

    def _T(self, H):
        return H.transpose(1, 2)

    def forward(self, y):
        # loss_1 : the x_sparse equal map
        loss_equal_map = 0
        # loss_2 : the sparse map
        loss_sparse = 0
        y = complex_feature_trasnsmission(y).transpose(1, 2)

        x = torch.zeros((2, self.k, batchsize))
        x = x.cuda()
        # x = self._shrink(self.gammas[0, :, :] * C(self.A, y), self.etas[0, :, :])

        for i in range(1, self.T + 1):
            x = x - self.gammas[i, :, :] * C(self.WX,  C(self.W, x) - y)
            # x = x - self.gammas[i, :, :] * C(self.B, x) + self.gammas[i, :, :] * C(self.A, y)
            # x = x - self.gammas[i, :, :] * C(self.B, x) + self.gammas[i, :, :] * C(self.A, y)
            xsp = self._function_compress(x, self.EncodeW1[0])
            xsp = self._shrink(xsp, self.etas[i, :, :])
            conjmatrix = self._T(torch.ones_like(self.EncodeW1[0]))
            conjmatrix[1] = -1*conjmatrix[1]
            conjmatrix = conjmatrix.cuda()
            decoderW = conjmatrix * self._T(self.EncodeW1[0])
            x_map = self._function_recovery(xsp, decoderW)
            loss_sparse += torch.mean(torch.sum(torch.abs(xsp[0]), dim=0))
            loss_equal_map += huber_loss(x, x_map)
            x = x_map
        out_sparse = sparse_degree(self._T(xsp)[0])
        return self._T(x), loss_sparse, loss_equal_map/self.T, out_sparse





