
from __future__ import annotations

import argparse
import os
from typing import List

import torch
from tqdm import trange
from tsl.nn.models.stgn import GRINModel

# Project utilities (same style as other runners)
from inc.diffus import SIR_STATES, b_estim, diffus_gen, seed_all

# Multi-snapshot aware tester (only evaluates on unobserved positions)
# If your repo uses inc.test as the ms tester, you can swap this import accordingly.
from inc.test_ms import Tester


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRIN baseline (multi-snapshot)")

    # ---- standard experiment args (align with hermes.py style) ----
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--seed", type=int, default=123456789, help="random seed")
    parser.add_argument("--data_dir", type=str, default="input", help="dataset folder")
    parser.add_argument("--output", type=str, default="output/grin.pt", help="output file name")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device, e.g., cpu | cuda | cuda:0",
    )

    # ---- multi-snapshot control ----
    # Keep hermes-style naming (obs_time) and also provide snapshot as an alias.
    parser.add_argument(
        "--obs_time",
        "--snapshot",
        dest="obs_time",
        type=str,
        default="",
        help="extra observed snapshot times, comma-separated (e.g., 5 or 3,7,9). "
             "Final time T is always observed automatically.",
    )

    # ---- diffusion parameter estimation (b_*) ----
    parser.add_argument("--b_pI0", type=float, default=1e-3,
                        help="initial infection rate in diffusion parameter estimation")
    parser.add_argument("--b_pR0", type=float, default=1e-3,
                        help="initial recovery rate in diffusion parameter estimation")
    parser.add_argument("--b_steps", type=int, default=500,
                        help="optimization steps in diffusion parameter estimation")
    parser.add_argument("--b_lr", type=float, default=3e-3,
                        help="learning rate in diffusion parameter estimation")

    # ---- GRINModel hyper-parameters ----
    # Ref: SPIN config/imputation/grin.yaml
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--ff_size", type=int, default=64)
    parser.add_argument("--embedding_size", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--decoder_order", type=int, default=1)
    parser.add_argument("--layer_norm", action="store_true",
                        help="enable layer norm inside GRIN")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ff_dropout", type=float, default=0.0)
    parser.add_argument("--merge_mode", type=str, default="mlp")

    # ---- optimizer / training ----
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--l2_reg", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)

    # ---- evaluation ----
    parser.add_argument("--rep", type=int, default=1, help="repetitions per dataset")

    return parser


def parse_obs_time(obs_time_str: str, T: int) -> List[int]:
    """
    Parse comma-separated times from CLI, clamp to [0, T], and always include T.

    hermes.py behavior:
      obs_time = [int(t) for t in args.obs_time.split(',') if t]
      obs_time.append(T)
      obs_time = sorted(set(obs_time))
    """
    times: List[int] = []
    if obs_time_str:
        for s in str(obs_time_str).split(","):
            s = s.strip()
            if not s:
                continue
            try:
                t = int(s)
            except ValueError:
                continue
            if 0 <= t <= T:
                times.append(t)

    times.append(T)  # final snapshot always observed
    times = sorted(set(times))
    return times


def build_time_mask(obs_time: List[int], T: int, device: torch.device) -> torch.Tensor:
    """
    Return a 1D mask over time: (T+1,) with 1 at observed times, else 0.
    """
    m = torch.zeros(T + 1, dtype=torch.long, device=device)
    if len(obs_time) > 0:
        idx = torch.tensor(obs_time, dtype=torch.long, device=device).clamp(0, T)
        m[idx.unique()] = 1
    return m


def grin_prep(y: torch.Tensor, obs_time: List[int], device: torch.device):
    """
    Prepare inputs for GRIN.

    Args:
        y: (samples, nodes, T+1) integer states.
        obs_time: list of observed snapshot times (must include T).
        device: torch device for mask.

    Returns:
        x:    (samples, T+1, nodes, 1) float, with missing frames zeroed
        mask: (samples, T+1, nodes, 1) long {0,1}, 1 means observed
    """
    n_samples, n_nodes, T1 = y.size()
    T = T1 - 1

    # time mask: (T+1,)
    tmask = build_time_mask(obs_time, T=T, device=device)  # long (T+1,)

    # GRIN input layout: (samples, T+1, nodes, 1)
    x = y.transpose(1, 2).unsqueeze(dim=3).float()

    # mask layout: (samples, T+1, nodes, 1)
    mask = tmask.view(1, T1, 1, 1).expand(n_samples, T1, n_nodes, 1)

    # IMPORTANT: avoid leaking ground-truth values at missing times
    x = x * mask.float()

    return x, mask


def grin_run_ms(data, args: argparse.Namespace) -> torch.Tensor:
    """
    Train GRIN on synthetic histories generated from estimated diffusion params,
    then impute the unobserved history for `data` under the multi-snapshot mask.
    """
    # ---- parse observed snapshots ----
    T = int(data.T.item())
    obs_time = parse_obs_time(args.obs_time, T)

    # attach obs info for the ms tester (so metrics only evaluate unobserved frames)
    # test_ms.py will look for data.obs_ts / obs_mask / obs_masks
    data.obs_ts = obs_time

    # ---- estimate diffusion params using observed snapshots ----
    bpar = b_estim(data, args, obs_time=obs_time)  # Dict(pI=..., pR=...)

    n_nodes = int(data.num_nodes)
    n_out = int(data.y[:, -1].max().item() + 1)

    # ---- build model ----
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

    # prior info used in original baseline (kept)
    I0 = int((data.y[:, 0] == SIR_STATES.I).long().sum().item())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    model.train()

    # ---- training ----
    pbar = trange(1, args.epochs + 1)
    for epoch in pbar:
        opt.zero_grad()

        # synthetic training histories: (T+1, nodes, samples)
        Y_true = diffus_gen(
            T=T,
            n_nodes=n_nodes,
            edge_index=data.edge_index,
            I0=I0,
            n_samples=args.batch_size,
            pI=bpar.pI,
            pR=bpar.pR,
        )

        # GRIN expects (samples, T+1, nodes, 1)
        # y for prep: (samples, nodes, T+1)
        y_snt = Y_true.transpose(0, 2)  # (samples, nodes, T+1)
        x, mask = grin_prep(y_snt, obs_time=obs_time, device=args.device)

        # model output: (samples, T+1, nodes, 1)
        z = model(x=x, mask=mask, edge_index=data.edge_index)[0]

        # ground-truth in same layout: (samples, T+1, nodes, 1)
        y_true_seq = Y_true.permute(2, 0, 1).unsqueeze(-1).float()

        # loss only on missing frames (mask == 0)
        unobs = (mask == 0)
        if bool(unobs.any()):
            loss = (z - y_true_seq).abs()[unobs].mean()
        else:
            # degenerate case: everything observed (rare), fallback to full loss
            loss = (z - y_true_seq).abs().mean()

        pbar.set_description(f"[epoch={epoch}] loss={loss.item():.4f}")
        loss.backward()
        opt.step()

    # ---- inference ----
    with torch.no_grad():
        model.eval()

        # build masked input from observed snapshots
        tmask_1d = build_time_mask(obs_time, T=T, device=args.device).bool()  # (T+1,)
        y_in = data.y.clone()
        y_in[:, ~tmask_1d] = 0  # hide unobserved frames

        x, mask = grin_prep(y_in.unsqueeze(0), obs_time=obs_time, device=args.device)

        z = model(x=x, mask=mask, edge_index=data.edge_index)[0]  # (1, T+1, nodes, 1)

        pred = z[0, :, :, 0].T  # (nodes, T+1)
        pred = pred.clamp(0, n_out - 1).round().long()

        y_pred = data.y.clone()
        y_pred[:, ~tmask_1d] = pred[:, ~tmask_1d]  # only fill missing times
        return y_pred


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # normalize device
    args.device = torch.device(args.device)

    # make sure output dir exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # run
    tester = Tester(args.data_dir, args.device, lambda data: grin_run_ms(data, args))
    tester.test([args.dataset], seed=args.seed, rep=args.rep)
    tester.save(args.output)


if __name__ == "__main__":
    main()
