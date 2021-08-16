import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.utils.data import Dataset


class MatrixDataset(Dataset):
    def __init__(self, X):
        self.X = X[:, :-1]
        self.Y = X[:, -1]
        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'x': self.X[idx, :], 'y': self.Y[idx]}


class MatrixDataset1D(Dataset):
    def __init__(self, X):
        self.X = X
        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx]


class OneLayer1to1Net(Module):
    def __init__(self, K=10, p=0.5, use_cuda=True):
        super(OneLayer1to1Net, self).__init__()
        self.use_cuda = use_cuda
        self.x = tc.ones(1, K)
        self.w = nn.Linear(K, 1, bias=False)
        self.d = nn.Dropout(p)
        self.p = p

    def forward(self, batch_size):
        x = to_variable(self.x.expand(batch_size, -1), self.use_cuda)
        x = x * (1 - self.p)
        # print x
        d = self.d(x)
        # print tc.sum(d)
        # print w
        return self.w(d)

    def eval(self):
        super(OneLayer1to1Net, self).eval()
        self.d.train()


class OneLayer1to1Net_ReLU(OneLayer1to1Net):
    def forward(self, batch_size):
        x = to_variable(self.x.expand(batch_size, -1), self.use_cuda)
        x = x * (1 - self.p)
        # print x
        d = self.d(x)
        # print tc.sum(d)
        # print w
        return F.relu(self.w(d))


class TwoLayers1to1Net(Module):
    def __init__(self, K=10, J=10, p=0.5, use_cuda=True):
        super(TwoLayers1to1Net, self).__init__()
        self.use_cuda = use_cuda
        self.x = tc.ones(1, K)
        self.w = nn.Linear(K, J, bias=False)
        self.w2 = nn.Linear(J, 1, bias=False)
        self.d = nn.Dropout(p)
        self.p = p

    def forward(self, batch_size):
        x = to_variable(self.x.expand(batch_size, -1), self.use_cuda)
        x = x * (1 - self.p)
        # print x
        d = self.d(x)
        # print tc.sum(d)
        w = self.w(d)
        # print w
        return self.w2(w)

    def eval(self):
        super(TwoLayers1to1Net, self).eval()
        self.d.train()


class TwoLayers1to1Net_b(TwoLayers1to1Net):
    def forward(self, batch_size):
        x = to_variable(self.x.expand(batch_size, -1), self.use_cuda)
        # print x
        w = self.w(x)
        w = self.d(w * (1 - self.p))
        # print w
        return self.w2(w)


class TwoLayers1to1Net_ReLU(TwoLayers1to1Net):
    def forward(self, batch_size):
        x = to_variable(self.x.expand(batch_size, -1), self.use_cuda)
        x = x * (1 - self.p)
        # print x
        d = self.d(x)
        # print tc.sum(d)
        w = self.d(F.relu(self.w(d)))
        # print w
        return self.w2(w)


class FCNet(Module):
    def __init__(self, dropout_p=0.5, last_layer_bias=True):
        super(FCNet, self).__init__()
        self.p = dropout_p
        self.h1 = nn.Linear(1, 10)
        self.h2 = nn.Linear(10, 10)
        self.h3 = nn.Linear(10, 100)
        self.h4 = nn.Linear(100, 100)
        self.h5 = nn.Linear(100, 1, bias=last_layer_bias)

    def forward(self, x):
        x = F.relu(self.h1(x))
        #x = F.dropout(F.relu(self.h2(x)), p=self.p, training=True)
        x = F.dropout(F.relu(self.h3(x)), p=self.p, training=True)
        x = F.dropout(F.relu(self.h4(x)), p=self.p, training=True)
        x = self.h5(x)
        return x


class MSERegularizedLoss(Module):
    def __init__(self, alpha=1):
        super(MSERegularizedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, weights, prediction, target):
        mse = F.mse_loss(prediction, target)
        reg = tc.sum(tc.pow(weights, 2))
        return mse + self.alpha * reg


class LogLikelihoodLoss(Module):
    def __init__(self, sigma=1, use_cuda=False):
        super(LogLikelihoodLoss, self).__init__()
        self.var = tc.pow(to_variable(tc.Tensor([sigma]), use_cuda), 2)

    def forward(self, prediction, target):
        mse = F.mse_loss(prediction, target)
        reg = tc.log(self.var)
        return mse / (2 * self.var) + .5 * reg


class LogLikelihoodRegularizedLoss(Module):
    def __init__(self, sigma=1, alpha=1):
        super(LogLikelihoodRegularizedLoss, self).__init__()
        self.log_loss = LogLikelihoodLoss(sigma)
        self.alpha = alpha

    def forward(self, weights, prediction, target):
        loss = self.log_loss(prediction, target)
        reg = tc.sum(tc.pow(weights, 2))
        return loss + self.alpha * reg


def to_cuda(net, use_cuda=True):
    if use_cuda:
        if tc.cuda.device_count() > 1:
            print("Let's use %d GPUs!" % tc.cuda.device_count())
            net = nn.DataParallel(net)

        if tc.cuda.is_available():
            print("loading network on CUDA")
            net.cuda()
        else:
            print("CUDA not available")
    return net


def to_variable(tensor, use_cuda=True):
    if use_cuda and tc.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
