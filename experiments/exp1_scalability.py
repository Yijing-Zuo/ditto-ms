#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 1: Scalability (HERMES)

Runtime profiling for:
    (1) vs T  (fixed n, vary T)
    (2) vs n  (fixed T, vary n)

Dataset: ba-sir
Observation: only TWO observed frames for each run:
    obs_time = [floor(T/2), T]

Defaults for vsT:
    T in {3,4,5,6,7,8,9,10}  (skip T=1,2)

Defaults for vsN:
    T_fixed = 10
    scale n by tiling disjoint copies (factors)

Run from repo root:
    python experiments/exp1_scalability.py --method hermes --dataset ba-sir --data_dir input --device cuda

Optional:
    --save_datasets to export generated .pt datasets (for traceability)
"""

from __future__ import annotations

import os
import sys
import gc
import csv
import time
import argparse
import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------
# Make repo root importable (so `import inc.*` and `import hermes` works)
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# project imports (re-use existing code)
from inc.data import data_load, data_make_states  # type: ignore
import hermes  # hermes.py must be import-safe (guarded by __main__)

# seeding util (fallback if inc.utils not present)
try:
    from inc.utils import seed_all  # type: ignore
except Exception:  # pragma: no cover
    import random
    import numpy as np

    def seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


CSV_FIELDS = [
    "timestamp",
    "exp",
    "method",
    "base_dataset",
    "dataset_pt",
    "seed",
    "device",
    "status",
    "n_nodes",
    "n_edges",
    "T",
    "obs_time_mid",
    "obs_time",
    "n_factor",
    # HERMES HP
    "b_pI0",
    "b_pR0",
    "b_steps",
    "b_lr",
    "q_steps",
    "q_lr",
    "q_hid",
    "q_gnn",
    "q_mlp",
    "q_samples",
    "q_zlim",
    "p_coef",
    "t_samples",
    "t_steps",
    "t_keep",
    # timings
    "b_estim_sec",
    "q_train_sec",
    "t_mcmc_sec",
    "total_sec",
    # meta
    "host",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_int_list(s: str) -> List[int]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = s.replace(",", " ").split()
    return [int(x) for x in parts]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_save_pt(data: Data, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    torch.save(data.cpu(), path)


def _variant_timespan(base: Data, T_new: int) -> Data:
    """
    Truncate/clip diffusion timespan to T_new and recompute y from (clipped) tI/tR.
    """
    assert T_new >= 1, "T_new should be >= 1"
    device = base.tI.device
    T_tensor = torch.tensor(T_new, dtype=base.T.dtype, device=device)

    # Clip hitting times beyond T_new to 'never' = T_new+1
    cap = torch.full_like(base.tI, T_new + 1)
    tI_new = torch.minimum(base.tI, cap)

    tR_new = None
    if hasattr(base, "tR") and getattr(base, "tR") is not None:
        capR = torch.full_like(base.tR, T_new + 1)
        tR_new = torch.minimum(base.tR, capR)

    y_new = data_make_states(T_new, tI_new, tR_new)

    out = Data(edge_index=base.edge_index.clone(), y=y_new, T=T_tensor, tI=tI_new.clone())
    if tR_new is not None:
        out.tR = tR_new.clone()

    out.num_nodes = y_new.size(0)
    return out


def _variant_scale_n(base: Data, factor: int) -> Data:
    """
    Tile disjoint copies of a base graph/history to scale n.
    This avoids rewriting the synthetic generator and is sufficient for runtime scaling.
    """
    assert factor >= 1
    if factor == 1:
        out = Data(edge_index=base.edge_index.clone(), y=base.y.clone(), T=base.T.clone(), tI=base.tI.clone())
        if hasattr(base, "tR") and getattr(base, "tR") is not None:
            out.tR = base.tR.clone()
        out.num_nodes = base.num_nodes
        return out

    n0 = int(base.num_nodes)
    eidx_list = [base.edge_index + k * n0 for k in range(factor)]
    edge_index = torch.cat(eidx_list, dim=1)

    y = base.y.repeat(factor, 1)
    tI = base.tI.repeat(factor)
    out = Data(edge_index=edge_index, y=y, T=base.T.clone(), tI=tI)
    if hasattr(base, "tR") and getattr(base, "tR") is not None:
        out.tR = base.tR.repeat(factor)

    out.num_nodes = y.size(0)
    return out


@dataclass
class HermesHP:
    """
    Defaults match your provided ba-sir command:
        --b_pI0 0.001 --b_pR0 0.001 --b_steps 500 --b_lr 0.003
        --q_steps 250 --q_lr 0.003 --q_hid 16 --q_gnn 3 --q_mlp 2 --q_samples 10 --q_zlim 16
        --p_coef 1.0 --t_samples 100 --t_steps 100 --t_keep 0.5
    """
    b_pI0: float = 0.001
    b_pR0: float = 0.001
    b_steps: int = 500
    b_lr: float = 0.003
    q_steps: int = 250
    q_lr: float = 0.003
    q_hid: int = 16
    q_gnn: int = 3
    q_mlp: int = 2
    q_samples: int = 10
    q_zlim: int = 16
    p_coef: float = 1.0
    t_samples: int = 100
    t_steps: int = 100
    t_keep: float = 0.5

    def to_namespace(self, *, device: torch.device, seed: int, obs_time_mid: int) -> argparse.Namespace:
        ns = argparse.Namespace(**asdict(self))
        ns.device = device
        ns.seed = seed
        # for compatibility; we pass obs_time explicitly in calls anyway
        ns.obs_time = str(obs_time_mid)
        return ns


def _run_hermes_once_timed(data: Data, args: argparse.Namespace, obs_time: List[int]) -> Dict[str, float]:
    """
    Run HERMES pipeline once and return per-stage runtimes (seconds):
        b_estim, q_train, t_mcmc, total
    """
    obs_time = sorted(set(int(t) for t in obs_time))

    # move to device (exclude copy time from timing)
    data = data.to(args.device)

    _sync(args.device)
    t0 = time.perf_counter()

    # 1) diffusion parameter estimation
    b0 = time.perf_counter()
    bpar = hermes.b_estim(data, args, obs_time=obs_time)
    _sync(args.device)
    t_b = time.perf_counter() - b0

    # 2) proposal training
    q0 = time.perf_counter()
    q_net = hermes.q_train(data, obs_time, bpar, args)
    _sync(args.device)
    t_q = time.perf_counter() - q0

    # 3) MCMC inference
    m0 = time.perf_counter()
    _ = hermes.t_mcmc(data, bpar, q_net, args, obs_time=obs_time, keepdim=True)
    _sync(args.device)
    t_m = time.perf_counter() - m0

    t_total = time.perf_counter() - t0

    # cleanup (outside timing)
    del q_net, bpar
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    return {"b_estim_sec": t_b, "q_train_sec": t_q, "t_mcmc_sec": t_m, "total_sec": t_total}


def _append_csv(path: str, row: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        fixed_row = {k: row.get(k, "") for k in CSV_FIELDS}
        writer.writerow(fixed_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="hermes", choices=["hermes"])
    parser.add_argument("--dataset", type=str, default="ba-sir")
    parser.add_argument("--data_dir", type=str, default="input")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123456789)

    # Exp controls
    parser.add_argument("--run_vsT", action="store_true", help="run scalability vs T")
    parser.add_argument("--run_vsN", action="store_true", help="run scalability vs n")

    # ba-sir: T=3..10 (skip 1,2)
    parser.add_argument("--T_list", type=str, default="3,4,5,6,7,8,9,10",
                        help="comma-separated T list for vsT")
    parser.add_argument("--T_fixed", type=int, default=10, help="fixed T for vsN")

    # scale-n factors (n = factor * n0). You can override to larger factors, script will mark OOM if happens.
    parser.add_argument("--n_factors", type=str, default="1,2,4,8,16",
                        help="tile factors for vsN (n = factor * n0)")
    parser.add_argument("--n_factor_for_vsT", type=int, default=1,
                        help="optionally scale n before varying T")

    # IO
    parser.add_argument("--gen_dir", type=str, default="",
                        help="where to save generated .pt datasets (default: <data_dir>/exp1_scalability)")
    parser.add_argument("--save_datasets", action="store_true", help="save generated datasets as .pt")
    parser.add_argument("--out_csv", type=str, default="output/exp1_scalability_runtime_ba_sir.csv")
    parser.add_argument("--overwrite_csv", action="store_true")

    # HERMES hyperparameters (defaults match your provided ba-sir command)
    parser.add_argument("--b_pI0", type=float, default=0.001)
    parser.add_argument("--b_pR0", type=float, default=0.001)
    parser.add_argument("--b_steps", type=int, default=500)
    parser.add_argument("--b_lr", type=float, default=0.003)
    parser.add_argument("--q_steps", type=int, default=250)
    parser.add_argument("--q_lr", type=float, default=0.003)
    parser.add_argument("--q_hid", type=int, default=16)
    parser.add_argument("--q_gnn", type=int, default=3)
    parser.add_argument("--q_mlp", type=int, default=2)
    parser.add_argument("--q_samples", type=int, default=10)
    parser.add_argument("--q_zlim", type=int, default=16)
    parser.add_argument("--p_coef", type=float, default=1.0)
    parser.add_argument("--t_samples", type=int, default=100)
    parser.add_argument("--t_steps", type=int, default=100)
    parser.add_argument("--t_keep", type=float, default=0.5)

    args_cli = parser.parse_args()

    if not args_cli.run_vsT and not args_cli.run_vsN:
        # default: run both
        args_cli.run_vsT = True
        args_cli.run_vsN = True

    # device resolve
    if args_cli.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args_cli.device)

    # output csv
    if args_cli.overwrite_csv and os.path.exists(args_cli.out_csv):
        os.remove(args_cli.out_csv)

    # gen dir
    gen_dir = args_cli.gen_dir.strip() or os.path.join(args_cli.data_dir, "exp1_scalability")
    _ensure_dir(gen_dir)

    # load base dataset on CPU (generation happens on CPU, then move to device for timing)
    base = data_load(args_cli.dataset, args_cli.data_dir, torch.device("cpu"))
    n0 = int(base.num_nodes)
    m0 = int(base.edge_index.size(1))
    T0 = int(base.T.item())
    print(f"[load] {args_cli.dataset}: n={n0}, m={m0}, T={T0}")

    # build HP template
    hp = HermesHP(
        b_pI0=args_cli.b_pI0,
        b_pR0=args_cli.b_pR0,
        b_steps=args_cli.b_steps,
        b_lr=args_cli.b_lr,
        q_steps=args_cli.q_steps,
        q_lr=args_cli.q_lr,
        q_hid=args_cli.q_hid,
        q_gnn=args_cli.q_gnn,
        q_mlp=args_cli.q_mlp,
        q_samples=args_cli.q_samples,
        q_zlim=args_cli.q_zlim,
        p_coef=args_cli.p_coef,
        t_samples=args_cli.t_samples,
        t_steps=args_cli.t_steps,
        t_keep=args_cli.t_keep,
    )

    host = platform.node()

    # -------------------------
    # run vsT
    # -------------------------
    if args_cli.run_vsT:
        T_list = _parse_int_list(args_cli.T_list)
        assert len(T_list) > 0, "T_list is empty"
        print(f"[exp] vsT: T_list={T_list}, n_factor_for_vsT={args_cli.n_factor_for_vsT}")

        base_scaled = _variant_scale_n(base, args_cli.n_factor_for_vsT)

        for T in T_list:
            if T > T0:
                print(f"[skip] T={T} > base T0={T0}.")
                continue
            if T <= 2:
                print(f"[skip] T={T} (skip T<=2 for this setting).")
                continue

            data_T = _variant_timespan(base_scaled, T)
            obs_mid = T // 2
            obs_time = [obs_mid, T]  # exactly TWO frames

            pt_path = ""
            if args_cli.save_datasets:
                pt_path = os.path.join(gen_dir, "vsT", f"{args_cli.dataset}_n{data_T.num_nodes}_T{T}.pt")
                _safe_save_pt(data_T, pt_path)

            seed_all(args_cli.seed)
            run_args = hp.to_namespace(device=device, seed=args_cli.seed, obs_time_mid=obs_mid)

            status = "ok"
            times = {"b_estim_sec": float("nan"), "q_train_sec": float("nan"),
                     "t_mcmc_sec": float("nan"), "total_sec": float("nan")}
            try:
                times = _run_hermes_once_timed(data_T, run_args, obs_time)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = "oom"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "exp": "vsT",
                "method": args_cli.method,
                "base_dataset": args_cli.dataset,
                "dataset_pt": pt_path,
                "seed": args_cli.seed,
                "device": str(device),
                "status": status,
                "n_nodes": int(data_T.num_nodes),
                "n_edges": int(data_T.edge_index.size(1)),
                "T": int(T),
                "obs_time_mid": int(obs_mid),
                "obs_time": ",".join(map(str, obs_time)),
                "n_factor": int(args_cli.n_factor_for_vsT),
                **asdict(hp),
                **times,
                "host": host,
            }
            _append_csv(args_cli.out_csv, row)
            print(f"[done] vsT T={T} n={row['n_nodes']} total={row['total_sec']:.3f}s status={status}")

            del data_T
            gc.collect()

    # -------------------------
    # run vsN
    # -------------------------
    if args_cli.run_vsN:
        factors = _parse_int_list(args_cli.n_factors)
        assert len(factors) > 0, "n_factors is empty"
        T_fixed = int(args_cli.T_fixed)
        if T_fixed > T0:
            print(f"[warn] T_fixed={T_fixed} > base T0={T0}, truncate to T0={T0}.")
            T_fixed = T0

        print(f"[exp] vsN: factors={factors}, T_fixed={T_fixed}")

        base_T = _variant_timespan(base, T_fixed)
        obs_mid = T_fixed // 2
        obs_time = [obs_mid, T_fixed]  # exactly TWO frames

        for fac in factors:
            data_N = _variant_scale_n(base_T, fac)

            pt_path = ""
            if args_cli.save_datasets:
                pt_path = os.path.join(gen_dir, "vsN", f"{args_cli.dataset}_n{data_N.num_nodes}_T{T_fixed}.pt")
                _safe_save_pt(data_N, pt_path)

            seed_all(args_cli.seed)
            run_args = hp.to_namespace(device=device, seed=args_cli.seed, obs_time_mid=obs_mid)

            status = "ok"
            times = {"b_estim_sec": float("nan"), "q_train_sec": float("nan"),
                     "t_mcmc_sec": float("nan"), "total_sec": float("nan")}
            try:
                times = _run_hermes_once_timed(data_N, run_args, obs_time)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = "oom"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "exp": "vsN",
                "method": args_cli.method,
                "base_dataset": args_cli.dataset,
                "dataset_pt": pt_path,
                "seed": args_cli.seed,
                "device": str(device),
                "status": status,
                "n_nodes": int(data_N.num_nodes),
                "n_edges": int(data_N.edge_index.size(1)),
                "T": int(T_fixed),
                "obs_time_mid": int(obs_mid),
                "obs_time": ",".join(map(str, obs_time)),
                "n_factor": int(fac),
                **asdict(hp),
                **times,
                "host": host,
            }
            _append_csv(args_cli.out_csv, row)
            print(f"[done] vsN fac={fac} n={row['n_nodes']} total={row['total_sec']:.3f}s status={status}")

            del data_N
            gc.collect()

    print(f"[ok] wrote CSV -> {args_cli.out_csv}")


if __name__ == "__main__":
    main()
