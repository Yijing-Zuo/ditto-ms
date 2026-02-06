#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 3: Parameter Estimation Accuracy (beta scatter)

Goal:
    Draw ONE scatter plot: x = true beta, y = estimated beta_hat (from HERMES b_estim).

Setting (simple & consistent with exp1/exp2):
    - Synthetic diffusion on a fixed base graph (default: ba-si graph structure).
    - Two observed snapshots for estimation: obs_time = [floor(T/2), T].
    - For each trial, sample a true beta ~ Uniform[beta_min, beta_max], simulate SI diffusion,
      then estimate beta_hat using inc.diffus.b_estim (segmented mean-field pseudo-likelihood).

Baseline (NOT plotted):
    A very simple well-mixed/logistic estimator using infected fraction i_t:
        beta_base = (logit(i_T) - logit(i_mid)) / (T - mid)
    We only print its error metrics to show our estimator is more accurate.

Run from repo root (example):
    python experiments/exp3_beta_scatter.py --dataset ba-si --data_dir input --device cuda \
        --trials 300 --T 10 --beta_min 0.02 --beta_max 0.25 --b_steps 300

Outputs:
    - CSV:  output/exp3_beta_scatter/points.csv
    - Plot: output/exp3_beta_scatter/beta_scatter.png  (+ optional pdf)
"""

from __future__ import annotations

import os
import sys
import math
import csv
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# -------------------------------
# Make repo root importable
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse existing project code (HERMES parameter estimation)
from inc.diffus import diffus_gen, b_estim, SIR_STATES  # type: ignore
from inc.utils import seed_all  # type: ignore

# Plot (keep minimal, matplotlib only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _logit(x: float, eps: float = 1e-6) -> float:
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def beta_baseline_well_mixed(y: torch.Tensor, t1: int, t2: int) -> float:
    """
    Simple baseline estimator using only infected fraction (no graph):
        beta = (logit(i_t2) - logit(i_t1)) / (t2 - t1)
    where i_t is fraction of infected nodes at time t.
    """
    assert 0 <= t1 < t2 < y.size(1), "invalid t1/t2 for y shape (nodes, T+1)"
    i1 = float((y[:, t1] == SIR_STATES.I).float().mean().item())
    i2 = float((y[:, t2] == SIR_STATES.I).float().mean().item())
    return (_logit(i2) - _logit(i1)) / float(t2 - t1)


def make_si_trial_data(
    edge_index: torch.Tensor,
    n_nodes: int,
    T: int,
    I0: int,
    beta_true: float,
    device: torch.device,
) -> Data:
    """
    Simulate ONE SI diffusion history and pack into torch_geometric.data.Data:
        data.y: (nodes, T+1) with states {S=0, I=1}
        data.T: scalar tensor
    """
    # diffus_gen returns Y: (T+1, nodes, samples)
    Y = diffus_gen(
        T=T,
        n_nodes=n_nodes,
        edge_index=edge_index,
        I0=I0,
        n_samples=1,
        pI=beta_true,
        pR=0.0,  # SI
    )
    y = Y[:, :, 0].T.contiguous()  # (nodes, T+1)

    data = Data(
        edge_index=edge_index,
        y=y,
        T=torch.tensor(T, dtype=torch.long, device=device),
    )
    data.num_nodes = n_nodes
    return data


def save_points_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    fieldnames = ["trial", "beta_true", "beta_hat", "beta_base"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(beta_true, beta_hat, T, obs_time, out_png, rmse=None):

    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    beta_true = np.asarray(beta_true)
    beta_hat  = np.asarray(beta_hat)

    lo = float(min(beta_true.min(), beta_hat.min()))
    hi = float(max(beta_true.max(), beta_hat.max()))
    pad = 0.02 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)

    ax.scatter(beta_true, beta_hat,
               s=10, alpha=0.65, linewidths=0)  # s/alpha 你可微调


    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", linewidth=1.2)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\hat{\beta}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


    ax.set_aspect("equal", adjustable="box")


    info = fr"$T={T}$, obs={obs_time}"
    if rmse is not None:
        info += fr"\nRMSE={rmse:.4f}"
    ax.text(0.05, 0.95, info, transform=ax.transAxes,
            ha="left", va="top", fontsize=10)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    p = argparse.ArgumentParser()

    # base graph source (we only need edge_index & n)
    p.add_argument("--dataset", type=str, default="ba-si",
                   help="use an existing dataset to load the base graph structure (e.g., ba-si, er-si).")
    p.add_argument("--data_dir", type=str, default="input",
                   help="dataset folder (used by data loader in your repo).")

    # simulation controls
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--beta_min", type=float, default=0.02)
    p.add_argument("--beta_max", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123456789)
    p.add_argument("--I0_frac", type=float, default=0.05,
                   help="initial infected fraction for each simulated history (SI).")

    # device
    p.add_argument("--device", type=str, default="cuda")

    # HERMES parameter estimation hyperparams (reuse b_estim)
    p.add_argument("--b_pI0", type=float, default=0.05,
                   help="initial beta guess in b_estim")
    p.add_argument("--b_pR0", type=float, default=0.0,
                   help="ignored for SI (kept for compatibility)")
    p.add_argument("--b_steps", type=int, default=2000)
    p.add_argument("--b_lr", type=float, default=0.001)

    # output
    p.add_argument("--out_dir", type=str, default="output/exp3_beta_scatter")
    p.add_argument("--save_pdf", action="store_true")

    args = p.parse_args()

    # device resolve
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    _ensure_dir(args.out_dir)

    # ---- load base graph structure ----
    # To avoid forcing this script to depend on extra libs, we try to load a cached .pt.
    # In your repo, inc.data.data_load() handles caching; here we import it lazily.
    try:
        from inc.data import data_load  # type: ignore
        base = data_load(args.dataset, args.data_dir, device)
    except Exception as e:
        raise RuntimeError(
            "Failed to load base dataset graph. "
            "Please ensure your repo provides inc.data.data_load and the dataset cache exists."
        ) from e

    edge_index = base.edge_index
    n_nodes = int(base.num_nodes)

    # initial infected count
    I0 = max(1, int(round(args.I0_frac * n_nodes)))

    # obs times: two frames
    T = int(args.T)
    obs_mid = T // 2
    obs_time = [obs_mid, T]

    # build a minimal args namespace for b_estim()
    # b_estim expects args.b_pI0/b_pR0/b_steps/b_lr.
    b_args = argparse.Namespace(
        b_pI0=float(args.b_pI0),
        b_pR0=float(args.b_pR0),
        b_steps=int(args.b_steps),
        b_lr=float(args.b_lr),
        # keep obs_time for compatibility if your b_estim reads it
        obs_time=str(obs_mid),
        device=device,
    )

    # ---- run trials ----
    rows: List[Dict[str, Any]] = []
    betas_true: List[float] = []
    betas_hat: List[float] = []
    betas_base: List[float] = []

    for k in range(args.trials):
        seed_all(args.seed ^ k)

        beta_true = float(np.random.uniform(args.beta_min, args.beta_max))
        data_k = make_si_trial_data(
            edge_index=edge_index,
            n_nodes=n_nodes,
            T=T,
            I0=I0,
            beta_true=beta_true,
            device=device,
        )

        # estimate beta_hat using HERMES b_estim (segmented mean-field)
        bpar = b_estim(data_k, b_args, obs_time=obs_time)
        beta_hat = float(bpar["pI"] if isinstance(bpar, dict) else bpar.pI)

        # baseline (not plotted)
        beta_base = float(beta_baseline_well_mixed(data_k.y, obs_mid, T))

        rows.append({
            "trial": k,
            "beta_true": beta_true,
            "beta_hat": beta_hat,
            "beta_base": beta_base,
        })
        betas_true.append(beta_true)
        betas_hat.append(beta_hat)
        betas_base.append(beta_base)

        if (k + 1) % max(1, args.trials // 10) == 0:
            print(f"[trial {k+1:>3d}/{args.trials}] beta={beta_true:.4f}  hat={beta_hat:.4f}  base={beta_base:.4f}")

    # ---- save csv ----
    csv_path = os.path.join(args.out_dir, "points.csv")
    save_points_csv(csv_path, rows)
    print(f"[ok] saved points -> {csv_path}")

    # ---- metrics (print only) ----
    bt = np.array(betas_true, dtype=float)
    bh = np.array(betas_hat, dtype=float)
    bb = np.array(betas_base, dtype=float)

    rmse_hat = float(np.sqrt(np.mean((bh - bt) ** 2)))
    mae_hat = float(np.mean(np.abs(bh - bt)))
    rmse_base = float(np.sqrt(np.mean((bb - bt) ** 2)))
    mae_base = float(np.mean(np.abs(bb - bt)))

    print(f"[metric] ours   RMSE={rmse_hat:.6f}  MAE={mae_hat:.6f}")
    print(f"[metric] base   RMSE={rmse_base:.6f}  MAE={mae_base:.6f}  (not plotted)")

    # ---- plot (ONE scatter only: true beta vs beta_hat) ----
    title = rf"$T={T}$, obs={obs_time}, trials={args.trials}  (RMSE={rmse_hat:.4f})"
    fig_png = os.path.join(args.out_dir, "beta_scatter.png")
    plot_scatter(fig_png, bt, bh, args.beta_min, args.beta_max, title)
    print(f"[ok] saved figure -> {fig_png}")

    if args.save_pdf:
        # re-render once to pdf (keep consistent)
        fig_pdf = os.path.join(args.out_dir, "beta_scatter.pdf")
        # quick replot
        lo = min(args.beta_min, float(bt.min()), float(bh.min()))
        hi = max(args.beta_max, float(bt.max()), float(bh.max()))
        pad = 0.02 * (hi - lo + 1e-12)
        lo, hi = lo - pad, hi + pad
        plt.figure(figsize=(5.2, 5.0))
        plt.scatter(bt, bh, s=18, alpha=0.8)
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(r"True $\beta$")
        plt.ylabel(r"Estimated $\hat{\beta}$")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_pdf)
        plt.close()
        print(f"[ok] saved figure -> {fig_pdf}")


if __name__ == "__main__":
    main()
