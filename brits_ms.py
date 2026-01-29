

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

try:
    # Multi-snapshot tester (recommended).
    from inc.test_ms import Tester
except Exception:
    # Fall back to single-snapshot tester if needed.
    from inc.test import Tester

from tqdm import trange

# Project utilities (diffusion simulator, parameter estimator, seeding, etc.)
from inc.diffus import *


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BRITS baseline (multi-snapshot)')

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

    # ---- multi-snapshot observed times ----
    parser.add_argument(
        '--obs_time', '--snapshot',
        dest='obs_time',
        type=str,
        default='',
        help='extra observed snapshot times, comma-separated, e.g., 5,7,9. '
             'Final time T is always observed.'
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

    # ---- evaluation ----
    parser.add_argument('--rep', type=int, default=1, help='number of test repetitions')

    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _parse_obs_time_str(s: str) -> List[int]:
    """Parse comma-separated time indices.

    Empty string -> []. Whitespace is ignored.
    """
    s = '' if s is None else str(s)
    out: List[int] = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def get_obs_time(data, args: argparse.Namespace) -> List[int]:
    """Return sorted unique observed times (always includes final time T)."""
    T = int(data.T.item())
    obs_time = _parse_obs_time_str(getattr(args, 'obs_time', ''))
    obs_time.append(T)
    obs_time = sorted({t for t in obs_time if 0 <= int(t) <= T})
    if len(obs_time) == 0 or obs_time[-1] != T:
        obs_time.append(T)
    return obs_time


@torch.no_grad()
def make_obs_mask(T: int, obs_time: Sequence[int], device: torch.device) -> torch.Tensor:
    """Make boolean mask of shape (T+1,) indicating observed time steps."""
    mask = torch.zeros(T + 1, dtype=torch.bool, device=device)
    if len(obs_time) == 0:
        mask[T] = True
        return mask
    ts = torch.as_tensor(list(obs_time), dtype=torch.long, device=device).clamp(0, T)
    ts = ts.unique()
    mask[ts] = True
    if not bool(mask[T].item()):
        mask[T] = True
    return mask


# -----------------------------------------------------------------------------
# BRITS core (largely copied from https://github.com/caow13/BRITS)
# -----------------------------------------------------------------------------

def binary_cross_entropy_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    size_average: bool = True,
    reduce: bool = True,
) -> torch.Tensor:
    """A numerically-stable BCE-with-logits implementation.

    Kept to match the original notebook/code.
    """
    if target.size() != input.size():
        raise ValueError(f'Target size ({target.size()}) must be the same as input size ({input.size()})')
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if weight is not None:
        loss = loss * weight
    if not reduce:
        return loss
    if size_average:
        return loss.mean()
    return loss.sum()


class FeatureRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask diagonal to avoid trivial self-regression
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, xdim: int, rnn_hid_size: int, impute_weight: float, label_weight: float):
        super().__init__()
        self.xdim = int(xdim)
        self.rnn_hid_size = int(rnn_hid_size)
        self.impute_weight = float(impute_weight)
        self.label_weight = float(label_weight)

        self.rnn_cell = nn.LSTMCell(self.xdim * 2, self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size=self.xdim, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.xdim, output_size=self.xdim, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hid_size, self.xdim)
        self.feat_reg = FeatureRegression(self.xdim)
        self.weight_combine = nn.Linear(self.xdim * 2, self.xdim)
        self.dropout = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data: dict, direct: str):
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].reshape((-1, 1))
        is_train = data['is_train'].reshape((-1, 1))

        device = values.device
        h = Variable(torch.zeros((values.size(0), self.rnn_hid_size), device=device))
        c = Variable(torch.zeros((values.size(0), self.rnn_hid_size), device=device))

        x_loss = 0.0
        imputations = []

        T_seq = min(values.size(1), masks.size(1), deltas.size(1))
        for t in range(T_seq):
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

        # (unused in our setting, but keep original structure)
        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)
        y_h = torch.sigmoid(y_h)

        loss = x_loss * self.impute_weight + y_loss * self.label_weight
        return {
            'loss': loss,
            'predictions': y_h,
            'imputations': imputations,
            'labels': labels,
            'is_train': is_train,
            'evals': evals,
            'eval_masks': eval_masks,
        }

    def run_on_batch(self, data: dict, optimizer: Optional[optim.Optimizer], epoch: Optional[int] = None):
        ret = self(data, direct='forward')
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret


class BRITSModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, impute_weight: float, label_weight: float):
        super().__init__()
        self.rits_f = RITS(input_size, hidden_size, impute_weight, label_weight)
        self.rits_b = RITS(input_size, hidden_size, impute_weight, label_weight)

    def forward(self, data: dict):
        ret_f = self.rits_f(data, direct='forward')
        ret_b = self.rits_b(data, direct='backward')
        ret_b = self.reverse(ret_b)

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

    @staticmethod
    def get_consistency_loss(pred_f: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred_f - pred_b).mean() * 1e-1

    @staticmethod
    def reverse(ret: dict) -> dict:
        def reverse_tensor(tensor_: torch.Tensor) -> torch.Tensor:
            if tensor_.dim() <= 1:
                return tensor_
            # reverse along time dimension (dim=1)
            idx = torch.arange(tensor_.size(1) - 1, -1, -1, device=tensor_.device)
            return tensor_.index_select(1, idx)

        return {k: reverse_tensor(v) for k, v in ret.items()}

    def run_on_batch(self, data: dict, optimizer: Optional[optim.Optimizer], epoch: Optional[int] = None):
        ret = self(data)
        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret


class BRITS(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, impute_weight: float = 1.0, label_weight: float = 1.0):
        super().__init__()
        self.model = BRITSModel(input_size, hidden_size, impute_weight, label_weight)

    def forward(self, data: dict):
        return self.model(data)

    def run_on_batch(self, data: dict, optimizer: Optional[optim.Optimizer], epoch: Optional[int] = None):
        return self.model.run_on_batch(data, optimizer, epoch)


# -----------------------------------------------------------------------------
# Data preparation (multi-snapshot masking)
# -----------------------------------------------------------------------------

def to_var(var):
    """Recursively move nested structures into torch Variables on the same device."""
    if torch.is_tensor(var):
        dev = var.device
        var = torch.autograd.Variable(var)
        var = var.to(dev)
        return var
    if isinstance(var, (int, float, str)):
        return var
    if isinstance(var, dict):
        return {key: to_var(val) for key, val in var.items()}
    if isinstance(var, list):
        return [to_var(x) for x in var]
    return var


@torch.no_grad()
def _brits_make_deltas(obs_mask_1d: torch.Tensor) -> torch.Tensor:
    """Make a (T+1,) float tensor of time gaps since the last observation.

    In forward direction:
      delta[t] = 0 if t is observed else delta[t-1] + 1.
    In backward direction we apply the same logic on the reversed mask.
    """
    if obs_mask_1d.dim() != 1:
        raise ValueError(f'obs_mask_1d must be 1D, got shape={tuple(obs_mask_1d.shape)}')

    L = obs_mask_1d.numel()
    d = torch.zeros(L, dtype=torch.float, device=obs_mask_1d.device)
    for t in range(1, L):
        d[t] = 0.0 if bool(obs_mask_1d[t].item()) else (d[t - 1] + 1.0)
    return d


@torch.no_grad()
def brits_prep_rec(y: torch.Tensor, obs_mask_1d: torch.Tensor) -> dict:
    """Prepare one direction (forward OR backward) input for BRITS.

    Args:
        y: (samples, T+1, nodes) full ground-truth sequence (will be masked).
        obs_mask_1d: (T+1,) bool mask for which time steps are observed.

    Returns:
        dict with keys values/masks/evals/eval_masks/deltas, all shaped
        (samples, T+1, nodes).
    """
    if y.dim() != 3:
        raise ValueError(f'y must be 3D (samples,T+1,nodes), got {tuple(y.shape)}')

    # Eval targets (full sequence, used only for reporting; NOT fed as observed).
    evals = y.float().clone()

    # Observation mask broadcast to (samples, T+1, nodes)
    obs_mask = obs_mask_1d.to(device=y.device, dtype=torch.bool)
    masks = obs_mask.view(1, -1, 1).expand(y.size(0), -1, y.size(2)).float()

    # Only keep observed frames in values; unobserved are set to 0.
    values = evals * masks

    # delta features
    deltas_1d = _brits_make_deltas(obs_mask)
    deltas = deltas_1d.view(1, -1, 1).expand_as(values).contiguous()

    eval_masks = masks.clone()

    return {
        'values': values.contiguous(),
        'masks': masks.contiguous(),
        'evals': evals.contiguous(),
        'eval_masks': eval_masks.contiguous(),
        'deltas': deltas.contiguous(),
    }


@torch.no_grad()
def brits_prep(y: torch.Tensor, is_train: int, obs_mask_1d: torch.Tensor) -> dict:
    """Prepare BRITS batch dict.

    Args:
        y: (samples, nodes, T+1)
        is_train: 1 or 0
        obs_mask_1d: (T+1,) bool in *forward* time.

    Returns:
        dict consumable by BRITSModel.
    """
    if y.dim() != 3:
        raise ValueError(f'y must be 3D (samples,nodes,T+1), got {tuple(y.shape)}')

    n_samples = int(y.size(0))
    y = y.transpose(1, 2)  # (samples, T+1, nodes)

    obs_mask_1d = obs_mask_1d.to(device=y.device, dtype=torch.bool)
    obs_mask_bwd = obs_mask_1d.flip(dims=[0])

    return to_var({
        'forward': brits_prep_rec(y, obs_mask_1d),
        'backward': brits_prep_rec(y.flip(dims=[1]), obs_mask_bwd),
        # BRITS classification head is unused; keep placeholders
        'labels': torch.zeros(n_samples, 1, dtype=torch.long, device=y.device),
        'is_train': torch.full((n_samples, 1), float(is_train), dtype=torch.float, device=y.device),
    })


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def brits_run(data, args: argparse.Namespace) -> torch.Tensor:
    """Train BRITS on synthetic histories and infer a history for the given data.

    Returns:
        y_pred: (nodes, T+1) long tensor of reconstructed states.
    """
    device = args.device
    data = data.to(device)

    T = int(data.T.item())
    obs_time = get_obs_time(data, args)

    # Attach observation times for the tester (so metrics exclude observed frames)
    # `inc.test_ms` uses `data.obs_ts` or `data.obs_mask` if present.
    data.obs_ts = obs_time

    obs_mask = make_obs_mask(T, obs_time, device=device)

    # Estimate diffusion parameters (supports multi-snapshot via obs_time)
    bpar = b_estim(data, args, obs_time=obs_time)

    model = BRITS(data.num_nodes, args.hid_size, args.impute_weight, args.label_weight).to(device)

    # -------------------- train --------------------
    I0 = int((data.y[:, 0] == SIR_STATES.I).long().sum().item())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pbar = trange(args.epochs)
    for epoch in pbar:
        model.train()
        # Generate synthetic training batch (full history), then mask it.
        Y = diffus_gen(
            T=T,
            n_nodes=data.num_nodes,
            edge_index=data.edge_index,
            I0=I0,
            n_samples=args.batch_size,
            pI=bpar.pI,
            pR=bpar.pR,
        )  # (T+1, nodes, samples)

        batch_y = Y.transpose(0, 2).clone()  # -> (samples, nodes, T+1)
        batch = brits_prep(batch_y, is_train=1, obs_mask_1d=obs_mask)
        ret = model.run_on_batch(batch, optimizer, epoch)
        pbar.set_description(f'epoch={epoch + 1} loss={ret["loss"].item():.4f}')

    # -------------------- infer --------------------
    with torch.no_grad():
        model.eval()
        rec = brits_prep(data.y.unsqueeze(dim=0).clone(), is_train=0, obs_mask_1d=obs_mask)
        ret = model.run_on_batch(rec, None)

        # ret['imputations']: (1, T+1, nodes) float
        y_pred = ret['imputations']
        y_pred = y_pred.long().clamp(0, int(data.y.max().item()))
        y_pred = y_pred.squeeze(dim=0).T.contiguous()  # (nodes, T+1)
        return y_pred.detach().clone()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)

    # Reproducibility
    seed_all(args.seed)

    def _model_fn(data):
        return brits_run(data, args)

    tester = Tester(args.data_dir, args.device, _model_fn)
    tester.test([args.dataset], seed=args.seed, rep=args.rep)
    tester.save(args.output)


if __name__ == '__main__':
    main()
