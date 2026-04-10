# experiments/runtime_breakdown_ba.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import time
import argparse
import traceback
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

# Make the project root importable when this script is placed under experiments/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inc.data import data_simulate
from inc.diffus import b_estim
from inc.utils import seed_all
from hermes import q_train, t_mcmc


# ----------------------------
# helpers
# ----------------------------
def parse_int_list(text):
    if text is None:
        return []
    text = str(text).strip()
    if text == "":
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_device(device_str):
    device_str = (device_str or "auto").strip().lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_str)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    sync_device(device)
    t0 = time.perf_counter()
    out = fn()
    sync_device(device)
    return out, time.perf_counter() - t0


def format_size_label(n):
    n = int(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n:,}"


def parse_obs_time(obs_time_str, T):
    obs = parse_int_list(obs_time_str)
    obs = [int(t) for t in obs if 0 <= int(t) <= int(T)]
    if int(T) not in obs:
        obs.append(int(T))
    obs = sorted(set(obs))
    return obs


# ----------------------------
# data + compose
# ----------------------------
def make_ba_sir_data(num_nodes, args):
    """
    Create BA-SIR locally in this experiment script, without touching inc/data.py.
    Uses the same BA-SIR setting as the current 1k synthetic BA experiment,
    except graph size is user-controlled.
    """
    Gnx = nx.barabasi_albert_graph(
        n=int(num_nodes),
        m=int(args.ba_m),
        seed=int(args.seed),
    )

    # BA with m>=1 is connected in practice, but keep this for safety.
    if not nx.is_connected(Gnx):
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key=len)).copy()

    meta = {
        "graph_size_actual": int(Gnx.number_of_nodes()),
        "num_edges": int(Gnx.number_of_edges()),
    }

    params = dict(
        fraction_infected=float(args.fraction_infected),
        beta=float(args.sim_pI),
        gamma=float(args.sim_pR),
    )

    data = data_simulate(
        Gnx=Gnx,
        seed=int(args.seed),
        T=int(args.T),
        diffus="sir",
        params=params,
    ).to(args.device)

    data.name = f"ba-sir-n{meta['graph_size_actual']}"
    return data, meta


@torch.no_grad()
def compose_history(data, tI, tR):
    """
    Same compose logic as run_hermes(), timed separately.
    Output shape follows the current project convention: (nodes, T)
    and the final observed snapshot is not duplicated here.
    """
    tI = tI.round().long()
    tR = tR.round().long()

    y_pred = torch.zeros_like(data.y)  # (nodes, T+1)
    y_pred.scatter_(
        dim=1,
        index=torch.minimum(tI, data.T),
        src=torch.full_like(tI, 1),
    )
    y_pred.scatter_(
        dim=1,
        index=torch.minimum(tR, data.T),
        src=torch.full_like(tR, 2),
    )
    y_pred = y_pred[:, : data.T.item()].cummax(dim=1).values
    return y_pred


# ----------------------------
# one run
# ----------------------------
def init_row(requested_graph_size, run_order, obs_time):
    row = OrderedDict()
    row["run_order"] = int(run_order)
    row["graph_size_requested"] = int(requested_graph_size)
    row["graph_size_actual"] = None
    row["num_edges"] = None
    row["T"] = None
    row["obs_time"] = ",".join(str(t) for t in obs_time)
    row["status"] = "pending"
    row["fail_stage"] = ""
    row["error"] = ""

    row["graph_build_sec"] = None
    row["beta_est_sec"] = None
    row["proposal_train_sec"] = None
    row["mcmc_sec"] = None
    row["compose_sec"] = None
    row["total_algo_sec"] = None
    row["total_wall_sec"] = None

    row["est_pI"] = None
    row["est_pR"] = None
    return row


def run_one_size(requested_graph_size, run_order, args):
    obs_time = parse_obs_time(args.obs_time, args.T)
    row = init_row(
        requested_graph_size=requested_graph_size,
        run_order=run_order,
        obs_time=obs_time,
    )

    data = None
    meta = None
    bpar = None
    q_net = None
    tI = None
    tR = None
    y_pred = None
    stage = "start"

    try:
        seed_all(int(args.seed))
        cleanup()

        stage = "graph_build"
        (data, meta), graph_build_sec = timed_call(
            lambda: make_ba_sir_data(requested_graph_size, args),
            args.device,
        )
        row["graph_size_actual"] = int(meta["graph_size_actual"])
        row["num_edges"] = int(meta["num_edges"])
        row["T"] = int(data.T.item())
        row["graph_build_sec"] = float(graph_build_sec)

        stage = "beta_est"
        bpar, beta_est_sec = timed_call(
            lambda: b_estim(
                data=data,
                args=args,
                obs_time=obs_time,
                assumed_I0=args.assumed_I0,
            ),
            args.device,
        )
        row["beta_est_sec"] = float(beta_est_sec)
        row["est_pI"] = float(bpar.pI)
        row["est_pR"] = float(bpar.pR)

        stage = "proposal_train"
        q_net, proposal_train_sec = timed_call(
            lambda: q_train(
                data=data,
                obs_time=obs_time,
                bpar=bpar,
                args=args,
                assumed_I0=args.assumed_I0,
            ),
            args.device,
        )
        row["proposal_train_sec"] = float(proposal_train_sec)

        stage = "mcmc"
        (tI, tR), mcmc_sec = timed_call(
            lambda: t_mcmc(
                data=data,
                bpar=bpar,
                q_net=q_net,
                args=args,
                obs_time=obs_time,
                keepdim=True,
                assumed_I0=args.assumed_I0,
                diagnostics=False,
            ),
            args.device,
        )
        row["mcmc_sec"] = float(mcmc_sec)

        stage = "compose"
        y_pred, compose_sec = timed_call(
            lambda: compose_history(data, tI, tR),
            args.device,
        )
        row["compose_sec"] = float(compose_sec)

        row["total_algo_sec"] = float(
            row["beta_est_sec"]
            + row["proposal_train_sec"]
            + row["mcmc_sec"]
            + row["compose_sec"]
        )
        row["total_wall_sec"] = float(
            row["graph_build_sec"] + row["total_algo_sec"]
        )
        row["status"] = "ok"

    except Exception as exc:
        row["status"] = "failed"
        row["fail_stage"] = stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[failed] n={requested_graph_size}, stage={stage}, error={row['error']}", flush=True)
        traceback.print_exc()

    finally:
        del data, meta, bpar, q_net, tI, tR, y_pred
        cleanup()

    return row


# ----------------------------
# save / plot
# ----------------------------
def save_rows(rows, csv_path):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def make_plot(rows, fig_path, q_steps):
    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print(f"[warn] no successful runs; skip figure: {fig_path}", flush=True)
        return

    df = df.sort_values("run_order")

    stage_cols = [
        ("beta_est_sec", "beta estimation"),
        ("proposal_train_sec", "proposal training"),
        ("mcmc_sec", "MCMC"),
        ("compose_sec", "compose"),
    ]

    x = list(range(len(df)))
    xticklabels = [format_size_label(n) for n in df["graph_size_actual"].tolist()]
    bottoms = [0.0] * len(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    for col, label in stage_cols:
        vals = df[col].fillna(0.0).astype(float).tolist()
        ax.bar(x, vals, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    totals = df["total_algo_sec"].fillna(0.0).astype(float).tolist()
    for i, total in enumerate(totals):
        ax.text(
            i,
            total,
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("BA graph size n")
    ax.set_ylabel("running time (sec)")
    ax.set_title(f"HERMES runtime breakdown on BA-SIR (q_steps={q_steps})")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Runtime breakdown on BA-SIR for 50k -> 30k -> 1k in one run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # run order: do NOT sort, preserve the exact user input order
    parser.add_argument(
        "--graph_sizes",
        type=str,
        default="50000,30000,1000",
        help="Comma-separated BA sizes; order is preserved exactly.",
    )
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="output/runtime_breakdown_ba")

    # BA-SIR generation
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument(
        "--obs_time",
        type=str,
        default="5",
        help="Extra observed times excluding T; T is appended automatically.",
    )
    parser.add_argument("--ba_m", type=int, default=4)
    parser.add_argument("--fraction_infected", type=float, default=0.05)
    parser.add_argument("--sim_pI", type=float, default=0.1, help="Ground-truth infection rate for simulation.")
    parser.add_argument("--sim_pR", type=float, default=0.1, help="Ground-truth recovery rate for simulation.")

    # diffusion parameter estimation
    parser.add_argument("--b_pI0", type=float, default=0.001)
    parser.add_argument("--b_pR0", type=float, default=0.001)
    parser.add_argument("--b_steps", type=int, default=500)
    parser.add_argument("--b_lr", type=float, default=0.003)

    # proposal training
    parser.add_argument("--q_steps", type=int, default=200)   # user-requested change
    parser.add_argument("--q_lr", type=float, default=0.003)
    parser.add_argument("--q_hid", type=int, default=16)
    parser.add_argument("--q_gnn", type=int, default=3)
    parser.add_argument("--q_mlp", type=int, default=2)
    parser.add_argument("--q_samples", type=int, default=10)
    parser.add_argument("--q_zlim", type=int, default=16)

    # MCMC
    parser.add_argument("--p_coef", type=float, default=1.0)
    parser.add_argument("--t_samples", type=int, default=100)
    parser.add_argument("--t_steps", type=int, default=100)
    parser.add_argument("--t_keep", type=float, default=0.5)

    # optional initial infected prior
    parser.add_argument(
        "--assumed_I0",
        type=int,
        default=None,
        help="If None, use the current code behavior (read I0 from data.y[:,0]).",
    )

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    return args


# ----------------------------
# main
# ----------------------------
def main():
    args = get_args()
    graph_sizes = parse_int_list(args.graph_sizes)
    if len(graph_sizes) == 0:
        raise ValueError("--graph_sizes is empty.")

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "runtime_breakdown_ba.csv")
    fig_path = os.path.join(args.output_dir, "runtime_breakdown_ba.png")

    print("=" * 80, flush=True)
    print("BA-SIR runtime breakdown experiment", flush=True)
    print(f"device      : {args.device}", flush=True)
    print(f"graph_sizes : {graph_sizes}", flush=True)
    print(f"T           : {args.T}", flush=True)
    print(f"obs_time    : {parse_obs_time(args.obs_time, args.T)}", flush=True)
    print(f"q_steps     : {args.q_steps}", flush=True)
    print("=" * 80, flush=True)

    rows = []
    total_runs = len(graph_sizes)

    for run_order, requested_graph_size in enumerate(graph_sizes, start=1):
        print(
            f"\n=== [{run_order}/{total_runs}] running BA-SIR with n={requested_graph_size} ===",
            flush=True,
        )
        row = run_one_size(
            requested_graph_size=requested_graph_size,
            run_order=run_order,
            args=args,
        )
        rows.append(row)
        save_rows(rows, csv_path)

        if row["status"] == "ok":
            print(
                "[ok] "
                f"n={row['graph_size_actual']:,}, "
                f"m={row['num_edges']:,}, "
                f"build={row['graph_build_sec']:.2f}s, "
                f"beta={row['beta_est_sec']:.2f}s, "
                f"q_train={row['proposal_train_sec']:.2f}s, "
                f"mcmc={row['mcmc_sec']:.2f}s, "
                f"compose={row['compose_sec']:.2f}s, "
                f"total_algo={row['total_algo_sec']:.2f}s, "
                f"total_wall={row['total_wall_sec']:.2f}s, "
                f"est_pI={row['est_pI']:.4f}, "
                f"est_pR={row['est_pR']:.4f}",
                flush=True,
            )
        else:
            print(
                "[failed] "
                f"n={requested_graph_size:,}, "
                f"stage={row['fail_stage']}, "
                f"error={row['error']}",
                flush=True,
            )

    make_plot(rows, fig_path, q_steps=args.q_steps)

    print("\nSaved:")
    print(f"  CSV : {csv_path}", flush=True)
    print(f"  FIG : {fig_path}", flush=True)


if __name__ == "__main__":
    main()