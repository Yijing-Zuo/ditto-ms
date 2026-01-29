# -*- coding: utf-8 -*-
# NOTE: This file was extracted from `grin.ipynb`.
# - Spatiotemporal 0.1.1: https://github.com/TorchSpatiotemporal/tsl/tree/1ae3289e00b28d0e84dfd54799561162df1917cd
# - SPIN: https://github.com/Graph-Machine-Learning-Group/spin

from __future__ import annotations

import argparse

import torch
from tsl.nn.models.stgn import GRINModel

from inc.diffus import *
from inc.test import *


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='GRIN baseline (single-snapshot)')

    # ---- standard experiment args (align with other runners, e.g., hermes.py) ----
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed')
    parser.add_argument('--data_dir', type=str, default='input', help='dataset folder')
    parser.add_argument('--output', type=str, default='output/grin.pt', help='output file name')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='torch device, e.g., cpu | cuda | cuda:0',
    )

    # ---- diffusion parameter estimation (b_*) ----
    parser.add_argument(
        '--b_pI0',
        type=float,
        default=1e-3,
        help='initial infection rate in diffusion parameter estimation',
    )
    parser.add_argument(
        '--b_pR0',
        type=float,
        default=1e-3,
        help='initial recovery rate in diffusion parameter estimation',
    )
    parser.add_argument(
        '--b_steps',
        type=int,
        default=500,
        help='optimization steps in diffusion parameter estimation',
    )
    parser.add_argument(
        '--b_lr',
        type=float,
        default=3e-3,
        help='learning rate in diffusion parameter estimation',
    )

    # ---- GRINModel hyper-parameters ----
    # Ref: https://github.com/Graph-Machine-Learning-Group/spin/blob/main/config/imputation/grin.yaml
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--ff_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--decoder_order', type=int, default=1)
    parser.add_argument(
        '--layer_norm',
        action='store_true',
        help='enable layer norm inside the GRIN model',
    )
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--ff_dropout', type=float, default=0.0)
    parser.add_argument('--merge_mode', type=str, default='mlp')

    # ---- optimizer / training ----
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='Adam weight decay')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)

    # ---- evaluation ----
    parser.add_argument('--rep', type=int, default=1, help='repetitions per dataset')

    args = parser.parse_args()

    # Normalize device to torch.device (keeps compatibility with other modules).
    args.device = torch.device(args.device)

    return args


def grin_prep(y: torch.Tensor, edge_index: torch.Tensor, device: torch.device):
    """Prepare inputs for GRIN.

    Args:
        y: (samples, nodes, T+1) integer states.
        edge_index: (2, edges)
        device: torch device for mask tensor

    Returns:
        x: (samples, T+1, nodes, 1)
        mask: (samples, T+1, nodes, 1) with only last snapshot observed
        ei: edge_index (kept for API symmetry)
    """

    n_samples, n_nodes, T1 = y.size()
    T = T1 - 1

    # (samples, T+1, nodes, 1)
    x = y.transpose(1, 2).unsqueeze(dim=3).float()

    # only last snapshot observed
    mask = (
        torch.tensor([[0]] * T + [[1]], device=device)
        .expand(n_samples, n_nodes, -1, 1)
        .transpose(1, 2)
    )

    ei = edge_index
    return x, mask, ei


def grin_run(data, args: argparse.Namespace):
    """Train GRIN on synthetic histories generated from estimated diffusion params,
    then impute the unobserved history for `data`.

    This is a minimal refactor of the original notebook logic.
    """

    bpar = b_estim(data, args)  # Dict(pI=..., pR=...)

    T = data.T.item()
    n_nodes = data.num_nodes
    n_out = data.y[:, -1].max().item() + 1

    # train
    model = GRINModel(
        input_size=1,
        hidden_size=args.hidden_size,
        ff_size=args.ff_size,
        embedding_size=args.embedding_size,
        n_layers=args.n_layers,
        n_nodes=n_nodes,
        kernel_size=args.kernel_size,
        decoder_order=args.decoder_order,
        layer_norm=args.layer_norm,
        dropout=args.dropout,
        ff_dropout=args.ff_dropout,
        merge_mode=args.merge_mode,
    ).to(args.device)

    I0 = (data.y[:, 0] == SIR_STATES.I).long().sum().item()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    model.train()

    pbar = trange(1, args.epochs + 1)
    for epoch in pbar:
        opt.zero_grad()

        # generate synthetic histories for training
        Y_true = diffus_gen(
            T=data.T.item(),
            n_nodes=data.num_nodes,
            edge_index=data.edge_index,
            I0=I0,
            n_samples=args.batch_size,
            pI=bpar.pI,
            pR=bpar.pR,
        )  # (T+1, nodes, samples)

        # GRIN expects (samples, T+1, nodes, features)
        x, mask, _ = grin_prep(Y_true.transpose(0, 2), data.edge_index, device=args.device)

        z = model(x=x, mask=mask, edge_index=data.edge_index)[0]  # (samples, T+1, nodes, 1)

        # L1 loss on the unobserved part (t < T)
        loss = (
            z[:, :-1].flatten()
            - Y_true[:-1].transpose(1, 2).transpose(0, 1).flatten()
        ).abs().mean()

        pbar.set_description(f'[epoch={epoch}] loss={loss.item():.4f}')
        loss.backward()
        opt.step()

    # infer
    with torch.no_grad():
        model.eval()
        x, mask, ei = grin_prep(data.y.unsqueeze(dim=0).clone(), data.edge_index, device=args.device)
        z = model(x=x, mask=mask, edge_index=ei)[0]  # (1, T+1, nodes, 1)

        y_pred = data.y.clone()
        y_pred[:, :-1] = (
            z[0, :-1, :, 0]
            .clamp(0, n_out - 1)
            .T
            .round()
            .long()
        )  # (nodes, T)

        return y_pred.clone()


def main() -> None:
    args = get_args()

    # Tester expects a callable: model_fn(data) -> y_pred
    model_fn = lambda data: grin_run(data, args)

    tester = Tester(args.data_dir, args.device, model_fn)
    tester.test([args.dataset], seed=args.seed, rep=args.rep)
    tester.save(args.output)


if __name__ == '__main__':
    main()
