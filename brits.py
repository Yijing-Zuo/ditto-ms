# ! pip install class-resolver==0.3.10
# ! pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install --no-index torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install torch-geometric==2.0.4
# ! pip install ndlib==5.1.1

from inc.diffus import *
from inc.test import *

import argparse
import torch


def get_args(argv=None):
    """Parse command-line arguments.

    We keep the original notebook defaults so running without extra flags
    behaves the same as before.
    """
    parser = argparse.ArgumentParser()

    # ---- experiment I/O ----
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed')
    parser.add_argument('--data_dir', type=str, default='input', help='dataset folder')
    parser.add_argument('--output', type=str, default='output/brits.pt', help='output file name')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        help='torch device, e.g., cpu, cuda, cuda:0'
    )

    # ---- diffusion parameter estimation (b_*) ----
    parser.add_argument('--b_pI0', type=float, default=1e-3,
                        help='initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type=float, default=1e-3,
                        help='initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type=int, default=500,
                        help='optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type=float, default=3e-3,
                        help='learning rate in diffusion parameter estimation')

    # ---- BRITS hyperparameters ----
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hid_size', type=int, default=108, help='RNN hidden size')
    parser.add_argument('--impute_weight', type=float, default=0.3, help='imputation loss weight')
    parser.add_argument('--label_weight', type=float, default=1.0, help='label loss weight')

    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


'''https://github.com/caow13/BRITS'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
# import utils
import argparse
# import data_loader

# from ipdb import set_trace
from sklearn import metrics


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if weight is not None:
        loss = loss * weight
    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag == True:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, xdim, rnn_hid_size, impute_weight, label_weight):
        super().__init__()
        self.xdim = xdim
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.xdim * 2, self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size=self.xdim, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.xdim, output_size=self.xdim, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hid_size, self.xdim)
        self.feat_reg = FeatureRegression(self.xdim)
        self.weight_combine = nn.Linear(self.xdim * 2, self.xdim)
        self.dropout = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct):
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']
        labels = data['labels'].reshape((-1, 1))
        is_train = data['is_train'].reshape((-1, 1))
        h = Variable(torch.zeros((values.size(0), self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size(0), self.rnn_hid_size)))
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0
        imputations = []
        for t in range(min(values.size(1), masks.size(1), deltas.size(1))):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            h = h * gamma_h
            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)
            c_c = m * x + (1 - m) * c_h
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(c_c.unsqueeze(dim=1))
        imputations = torch.cat(imputations, dim=1)
        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)
        y_h = torch.sigmoid(y_h)
        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h, \
                'imputations': imputations, 'labels': labels, 'is_train': is_train, \
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
# import utils
import argparse
# import data_loader

# import rits
from sklearn import metrics


# from ipdb import set_trace

class BRITS(nn.Module):
    def __init__(self, xdim, rnn_hid_size, impute_weight, label_weight):
        super().__init__()
        self.xdim = xdim
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.build()

    def build(self):
        self.rits_f = RITS(self.xdim, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.rits_b = RITS(self.xdim, self.rnn_hid_size, self.impute_weight, self.label_weight)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        loss = loss_f + loss_b + loss_c
        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations
        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)
            if torch.cuda.is_available():
                indices = indices.cuda()
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])
        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
# import utils
# import models
import argparse
# import data_loader
import pandas as pd
import ujson as json

from sklearn import metrics


# from ipdb import set_trace

def to_var(var):
    if torch.is_tensor(var):
        device = var.device
        var = torch.autograd.Variable(var)
        var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        return {key: to_var(val) for key, val in var.items()}
    if isinstance(var, list):
        return [to_var(x) for x in var]


@torch.no_grad()
def brits_prep_rec(y, back):  # (samples, T + 1, nodes)
    evals = y.float().clone()
    values = evals.clone()
    if back:
        values[:, 1:] = 0
    else:
        values[:, : -1] = 0
    masks = torch.zeros_like(y)
    if back:
        masks[:, 0] = True
    else:
        masks[:, -1] = True
    eval_masks = masks.clone()
    deltas = torch.cat([torch.arange(y.size(dim=1) - 1, dtype=torch.float, device=y.device),
                        torch.zeros(1, dtype=torch.float, device=y.device)], dim=0).unsqueeze(dim=-1).expand(*y.size())
    return dict(values=values.contiguous(), masks=masks.contiguous(), evals=evals.contiguous(),
                eval_masks=eval_masks.contiguous(), deltas=deltas.contiguous())


@torch.no_grad()
def brits_prep(y, is_train):  # y: (samples, nodes, T + 1)
    n_samples = y.size(dim=0)
    y = y.transpose(1, 2)  # (samples, T + 1, nodes)
    return to_var(dict(
        forward=brits_prep_rec(y, back=False),
        backward=brits_prep_rec(y.flip(dims=[1]), back=True),
        labels=torch.zeros(n_samples, 1, dtype=torch.long, device=y.device),
        is_train=torch.tensor([is_train] * n_samples, dtype=torch.float, device=y.device),
    ))


def brits_run(data, args):
    """Run BRITS on a single dataset instance (loaded by Tester)."""
    bpar = b_estim(data, args)

    model = BRITS(data.num_nodes, args.hid_size, args.impute_weight, args.label_weight)
    model = model.to(args.device)

    # train
    I0 = (data.y[:, 0] == SIR_STATES.I).long().sum().item()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pbar = trange(args.epochs)
    for epoch in pbar:
        model.train()
        batch = brits_prep(diffus_gen(T=data.T.item(), n_nodes=data.num_nodes, edge_index=data.edge_index, I0=I0,
                                      n_samples=args.batch_size, pI=bpar.pI, pR=bpar.pR).transpose(0, 2).clone(),
                           is_train=1)
        ret = model.run_on_batch(batch, optimizer, epoch)
        pbar.set_description(f'epoch={epoch + 1} loss={ret["loss"].item():.4f}')

    # infer
    with torch.no_grad():
        model.eval()
        rec = brits_prep(data.y.unsqueeze(dim=0).clone(), is_train=0)
        ret = model.run_on_batch(rec, None)
        y_pred = ret['imputations'].long().clamp(0, data.y.max()).squeeze(dim=0).T.clone()
        return y_pred.clone()


def main(argv=None):
    args = get_args(argv)

    # Match the other runners (e.g., hermes.py): run one dataset specified by CLI.
    seed_all(args.seed)
    tester = Tester(args.data_dir, args.device, lambda data: brits_run(data, args))
    tester.test([args.dataset], seed=args.seed, rep=1)
    tester.save(args.output)


if __name__ == '__main__':
    main()

