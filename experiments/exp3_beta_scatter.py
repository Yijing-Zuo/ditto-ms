

import os
import sys
import csv
import argparse
import zlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data

# -------------------------------
# Make repo root importable.
# This supports either layout:
#   - <repo>/experiments/exp3_beta_scatter.py  (inc/ in parent)
#   - <repo>/exp3_beta_scatter.py              (inc/ in same dir)
# -------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
CANDIDATE_ROOTS = [HERE, os.path.abspath(os.path.join(HERE, ".."))]
for cand in CANDIDATE_ROOTS:
    if os.path.isdir(os.path.join(cand, "inc")):
        if cand not in sys.path:
            sys.path.insert(0, cand)
        break

from inc.data import data_load  # type: ignore
from inc.diffus import diffus_gen, b_estim, SIR_STATES  # type: ignore
from inc.utils import seed_all  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------
# True beta settings for main datasets (D1/D2)
# -------------------------------
# NOTE:
# True parameters are not stored in the cached .pt files, so we encode the *main experiment*
# settings here (DITTO paper Appendix D.1; our data.py follows the same settings).
def true_params_for_dataset(dataset: str) -> Tuple[float, float]:
    d = dataset.lower()
    is_sir = d.endswith("-sir")
    is_si = d.endswith("-si")
    if not (is_sir or is_si):
        raise ValueError("dataset name must end with -si or -sir, got: %s" % dataset)

    # D1: synthetic graphs (BA/ER)
    if d.startswith("ba-") or d.startswith("er-"):
        pI = 0.1
        pR = 0.1 if is_sir else 0.0
        return pI, pR

    # D2: real graphs (Oregon2/Prost) with synthetic diffusion
    if d.startswith("oregon2-") or d.startswith("prost-"):
        pI = 0.1
        pR = 0.05 if is_sir else 0.0
        return pI, pR

    raise ValueError(
        "Unsupported dataset for this exp3 scatter (needs synthetic diffusion with known beta): %s"
        % dataset
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stable_int_hash(s: str) -> int:
    """Stable 32-bit hash (avoid Python's randomized hash)."""
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def simulate_one_history(
    edge_index: torch.Tensor,
    n_nodes: int,
    T: int,
    I0: int,
    pI_true: float,
    pR_true: float,
    device: torch.device,
    max_tries: int = 50,
) -> Data:
    """
    Simulate ONE diffusion history with fixed parameters (pI_true, pR_true).

    For SIR (pR_true > 0), we optionally resample until at least one node reaches R
    (otherwise b_estim will infer n_cls=2 and skip estimating pR).

    Returns a torch_geometric Data with:
        data.edge_index
        data.y  : (nodes, T+1)
        data.T  : scalar tensor
    """
    last_y: Optional[torch.Tensor] = None

    for _try in range(max_tries):
        Y = diffus_gen(
            T=T,
            n_nodes=n_nodes,
            edge_index=edge_index,
            I0=I0,
            n_samples=1,
            pI=pI_true,
            pR=pR_true,
        )  # (T+1, nodes, 1)

        y = Y[:, :, 0].T.contiguous()  # (nodes, T+1)
        last_y = y

        if pR_true > 0:
            # Ensure recovered exists at final time so b_estim treats it as SIR.
            if not (y == SIR_STATES.R).any().item():
                continue

        data = Data(
            edge_index=edge_index,
            y=y,
            T=torch.tensor(T, dtype=torch.long, device=device),
        )
        data.num_nodes = n_nodes
        return data

    # Fallback: return the last simulated history even if it has no recovered state.
    assert last_y is not None
    data = Data(
        edge_index=edge_index,
        y=last_y,
        T=torch.tensor(T, dtype=torch.long, device=device),
    )
    data.num_nodes = n_nodes
    return data


def save_points_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    fieldnames = ["dataset", "trial", "param", "beta_true", "beta_hat"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def plot_scatter(
    rows: List[Dict[str, Any]],
    out_png: str,
    title: str,
    rmse: Optional[float] = None,
    save_pdf: bool = False,
) -> None:
    beta_true = np.asarray([float(r["beta_true"]) for r in rows], dtype=float)
    beta_hat = np.asarray([float(r["beta_hat"]) for r in rows], dtype=float)
    params = [str(r["param"]) for r in rows]

    mask_I = np.array([p == "pI" for p in params], dtype=bool)
    mask_R = np.array([p == "pR" for p in params], dtype=bool)

    lo = float(min(beta_true.min(), beta_hat.min()))
    hi = float(max(beta_true.max(), beta_hat.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)

    if mask_I.any():
        ax.scatter(beta_true[mask_I], beta_hat[mask_I], s=10, alpha=0.65, linewidths=0, label=r"$\beta_I$")
    if mask_R.any():
        ax.scatter(beta_true[mask_R], beta_hat[mask_R], s=14, alpha=0.65, linewidths=0, marker="x", label=r"$\beta_R$")

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", linewidth=1.2)

    ax.set_xlabel(r"True $\beta$")
    ax.set_ylabel(r"Estimated $\hat{\beta}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    info = title
    if rmse is not None:
        info += "\nRMSE=%.4f" % rmse
    ax.text(0.05, 0.95, info, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    if mask_R.any():
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)
        if mask_I.any():
            ax.scatter(beta_true[mask_I], beta_hat[mask_I], s=10, alpha=0.65, linewidths=0, label=r"$\beta_I$")
        if mask_R.any():
            ax.scatter(beta_true[mask_R], beta_hat[mask_R], s=14, alpha=0.65, linewidths=0, marker="x", label=r"$\beta_R$")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", linewidth=1.2)
        ax.set_xlabel(r"True $\beta$")
        ax.set_ylabel(r"Estimated $\hat{\beta}$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if mask_R.any():
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout(pad=0.2)
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


def parse_datasets_arg(s: str) -> List[str]:
    s = (s or "").strip()
    if s.lower() in {"main", "default"}:
        # 8 synthetic datasets in main experiments (D1+D2), where true beta is known.
        return [
            "ba-si", "ba-sir",
            "er-si", "er-sir",
            "oregon2-si", "oregon2-sir",
            "prost-si", "prost-sir",
        ]
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        type=str,
        default="main",
        help="Comma-separated datasets, or 'main' for BA/ER/Oregon2/Prost with SI+SIR.",
    )
    p.add_argument("--data_dir", type=str, default="input")
    p.add_argument("--device", type=str, default="cuda")

    # trials
    p.add_argument("--trials_per_dataset", type=int, default=30)
    p.add_argument("--seed", type=int, default=123456789)

    # plotting: default includes beta_R for SIR
    p.add_argument(
        "--only_betaI",
        action="store_true",
        help="If set, only plot infection beta_I (skip recovery beta_R).",
    )

    # b_estim hyperparams
    p.add_argument("--b_pI0", type=float, default=0.05)
    p.add_argument("--b_pR0", type=float, default=0.05)
    p.add_argument("--b_steps", type=int, default=300)
    p.add_argument("--b_lr", type=float, default=0.001)

    # output
    p.add_argument("--out_dir", type=str, default="output/exp3_beta_scatter_main")
    p.add_argument("--save_pdf", action="store_true")

    args = p.parse_args()

    # device resolve
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[warn] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    datasets = parse_datasets_arg(args.datasets)
    if not datasets:
        raise ValueError("Empty --datasets")

    _ensure_dir(args.out_dir)

    # b_estim expects these in args; we also pass obs_time explicitly.
    b_args = argparse.Namespace(
        b_pI0=float(args.b_pI0),
        b_pR0=float(args.b_pR0),
        b_steps=int(args.b_steps),
        b_lr=float(args.b_lr),
        obs_time="",  # not used when obs_time is passed explicitly
        device=device,
    )

    rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        # Load once to get base graph (edge_index, num_nodes) and T.
        base = data_load(dataset, args.data_dir, device)
        edge_index = base.edge_index
        n_nodes = int(base.num_nodes)
        T = int(base.T.item())

        # main protocol: observe { floor(T/2), T }
        t_obs = T // 2
        obs_time = [t_obs, T]

        # I0 from cached dataset (matches generation protocol)
        I0 = int((base.y[:, 0] == SIR_STATES.I).sum().item())
        I0 = max(1, I0)

        pI_true, pR_true = true_params_for_dataset(dataset)

        # run multiple trials on this dataset
        for k in range(int(args.trials_per_dataset)):
            seed_k = (int(args.seed) ^ _stable_int_hash(dataset) ^ int(k)) & 0xFFFFFFFF
            seed_all(seed_k)

            data_k = simulate_one_history(
                edge_index=edge_index,
                n_nodes=n_nodes,
                T=T,
                I0=I0,
                pI_true=pI_true,
                pR_true=pR_true,
                device=device,
            )

            est = b_estim(data_k, b_args, obs_time=obs_time)  # dict with keys pI, pR

            # record beta_I
            rows.append(
                dict(
                    dataset=dataset,
                    trial=k,
                    param="pI",
                    beta_true=float(pI_true),
                    beta_hat=float(est["pI"]),
                )
            )

            # record beta_R for SIR unless disabled
            if (not args.only_betaI) and (pR_true > 0):
                rows.append(
                    dict(
                        dataset=dataset,
                        trial=k,
                        param="pR",
                        beta_true=float(pR_true),
                        beta_hat=float(est.get("pR", 0.0)),
                    )
                )

        print(
            "[ok] %s: n=%d, T=%d, obs=%s, I0=%d, true=(pI=%.3f, pR=%.3f), trials=%d"
            % (dataset, n_nodes, T, str(obs_time), I0, pI_true, pR_true, int(args.trials_per_dataset))
        )

    # ---- save csv ----
    csv_path = os.path.join(args.out_dir, "points.csv")
    save_points_csv(csv_path, rows)
    print("[ok] saved points -> %s" % csv_path)

    # ---- metrics ----
    bt = np.array([float(r["beta_true"]) for r in rows], dtype=float)
    bh = np.array([float(r["beta_hat"]) for r in rows], dtype=float)
    rmse = float(np.sqrt(np.mean((bh - bt) ** 2)))
    mae = float(np.mean(np.abs(bh - bt)))
    print("[metric] overall RMSE=%.6f  MAE=%.6f  (points=%d)" % (rmse, mae, len(rows)))

    # ---- plot ----
    fig_png = os.path.join(args.out_dir, "beta_scatter.png")
    title = "datasets=%d, trials/ds=%d, only_betaI=%s" % (
        len(datasets),
        int(args.trials_per_dataset),
        str(bool(args.only_betaI)),
    )
    plot_scatter(rows, out_png=fig_png, title=title, rmse=rmse, save_pdf=bool(args.save_pdf))
    print("[ok] saved figure -> %s" % fig_png)


if __name__ == "__main__":
    main()
