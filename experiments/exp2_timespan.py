#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 2: Effect of Timespan (ba-sir)

- Observations: only TWO frames per run:
    obs_time = { floor(T/2), T }

- Sweep T from 3 to 10 (skip 1,2).
- For each T, regenerate a dataset file:
    <data_dir>/exp2_timespan/T{T}/synthetic/ba-sir.pt

- Compare:
    1) HERMES   (hermes.py)
    2) CRI-MS   (cri_ms.py)
    3) DHREC-MS (dhrec_ms.py)

- Evaluation:
    Use the existing inc/tester inside each script (they already do),
    then this orchestrator reads the saved .pt result and writes a CSV
    with f1 and nrmse (no plotting).

Run from repo root:
    python experiments/exp2_timespan.py --data_dir input --device_hermes cuda --device_dhrec cuda --device_cri cpu
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------
# Make repo root importable (so `from inc.data import ...` works)
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inc.data import data_load, data_make_states  # type: ignore


CSV_FIELDS = [
    "timestamp",
    "dataset",
    "T",
    "obs_time_mid",
    "obs_time",
    "method",
    "device",
    "seed",
    "status",
    "f1",
    "nrmse",
    "result_pt",
    "data_dir_used",
    "data_pt_used",
    "cmd",
    "elapsed_sec",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_csv(path: str, row: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        fixed_row = {k: row.get(k, "") for k in CSV_FIELDS}
        w.writerow(fixed_row)


def _variant_timespan(base: Data, T_new: int) -> Data:
    """
    Create a new Data with timespan clipped to T_new:
      - tI, tR clipped to <= T_new+1
      - y rebuilt by data_make_states(T_new, tI_new, tR_new)
    """
    assert T_new >= 1
    device = base.tI.device  # typically CPU here
    T_tensor = torch.tensor(T_new, dtype=base.T.dtype, device=device)

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


def _save_ba_sir_variant(data_dir_T: str, data_T: Data) -> str:
    """
    Save variant dataset to:
      <data_dir_T>/synthetic/ba-sir.pt
    so that data_load('ba-sir', data_dir_T, device) will load this file.
    """
    pt_path = os.path.join(data_dir_T, "synthetic", "ba-sir.pt")
    _ensure_dir(os.path.dirname(pt_path))
    torch.save(data_T.cpu(), pt_path)
    return pt_path


def _load_metrics_from_tester_pt(result_pt: str, dataset: str) -> Dict[str, float]:
    """
    Each method script saves tester.res via torch.save(res, output).
    res format: res[dataset]['f1'] = [..], res[dataset]['nrmse'] = [..]
    """
    res = torch.load(result_pt, map_location="cpu", weights_only=False)
    f1 = float(res[dataset]["f1"][-1])
    nrmse = float(res[dataset]["nrmse"][-1])
    return {"f1": f1, "nrmse": nrmse}


def _run_cmd(cmd: List[str]) -> float:
    """
    Run a command and return wall-clock seconds.
    """
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()

    # sweep setting
    p.add_argument("--dataset", type=str, default="ba-sir")
    p.add_argument("--seed", type=int, default=123456789)
    p.add_argument("--data_dir", type=str, default="input")
    p.add_argument("--T_min", type=int, default=3)
    p.add_argument("--T_max", type=int, default=10)

    # where to store per-T datasets and per-run result .pt
    p.add_argument("--gen_root", type=str, default="", help="default: <data_dir>/exp2_timespan")
    p.add_argument("--out_root", type=str, default="output/exp2_timespan")
    p.add_argument("--out_csv", type=str, default="output/exp2_timespan_metrics_ba_sir.csv")
    p.add_argument("--overwrite_csv", action="store_true")

    # devices
    p.add_argument("--device_hermes", type=str, default="cuda")
    p.add_argument("--device_dhrec", type=str, default="cuda")
    p.add_argument("--device_cri", type=str, default="cpu")

    # HERMES hyperparameters (keep fixed)
    p.add_argument("--b_pI0", type=float, default=0.001)
    p.add_argument("--b_pR0", type=float, default=0.001)
    p.add_argument("--b_steps", type=int, default=500)
    p.add_argument("--b_lr", type=float, default=0.003)
    p.add_argument("--q_steps", type=int, default=250)
    p.add_argument("--q_lr", type=float, default=0.003)
    p.add_argument("--q_hid", type=int, default=16)
    p.add_argument("--q_gnn", type=int, default=3)
    p.add_argument("--q_mlp", type=int, default=2)
    p.add_argument("--q_samples", type=int, default=10)
    p.add_argument("--q_zlim", type=int, default=16)
    p.add_argument("--p_coef", type=float, default=1.0)
    p.add_argument("--t_samples", type=int, default=100)
    p.add_argument("--t_steps", type=int, default=100)
    p.add_argument("--t_keep", type=float, default=0.5)

    args = p.parse_args()

    if args.overwrite_csv and os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    gen_root = args.gen_root.strip() or os.path.join(args.data_dir, "exp2_timespan")
    _ensure_dir(gen_root)
    _ensure_dir(args.out_root)

    # Load base ba-sir once (CPU). For synthetic ba-sir, this is typically T=10 cached at <data_dir>/synthetic/ba-sir.pt.
    base = data_load(args.dataset, args.data_dir, torch.device("cpu"))
    base_T = int(base.T.item())

    print(f"[load base] dataset={args.dataset} base_T={base_T} (expect >= {args.T_max})")

    py = sys.executable

    # sweep
    for T in range(args.T_min, args.T_max + 1):
        if T <= 2:
            continue
        if T > base_T:
            print(f"[skip] T={T} > base_T={base_T}")
            continue

        obs_mid = T // 2
        obs_time = [obs_mid, T]  # exactly two frames
        obs_time_str = ",".join(map(str, obs_time))

        # 1) regenerate data for this T into a dedicated data_dir
        data_dir_T = os.path.join(gen_root, f"T{T}")
        data_T = _variant_timespan(base, T)
        data_pt = _save_ba_sir_variant(data_dir_T, data_T)
        print(f"[data] T={T} saved -> {data_pt} (obs={obs_time_str})")

        # 2) run methods
        # 2.1 HERMES: obs_time arg is "extra observed times"; final T always included inside hermes.py
        hermes_out = os.path.join(args.out_root, f"hermes_T{T}.pt")
        cmd_hermes = [
            py, "hermes.py",
            "--dataset", args.dataset,
            "--seed", str(args.seed),
            "--data_dir", data_dir_T,
            "--output", hermes_out,
            "--device", args.device_hermes,
            "--obs_time", str(obs_mid),
            "--b_pI0", str(args.b_pI0), "--b_pR0", str(args.b_pR0), "--b_steps", str(args.b_steps), "--b_lr", str(args.b_lr),
            "--q_steps", str(args.q_steps), "--q_lr", str(args.q_lr), "--q_hid", str(args.q_hid), "--q_gnn", str(args.q_gnn),
            "--q_mlp", str(args.q_mlp), "--q_samples", str(args.q_samples), "--q_zlim", str(args.q_zlim),
            "--p_coef", str(args.p_coef),
            "--t_samples", str(args.t_samples), "--t_steps", str(args.t_steps), "--t_keep", str(args.t_keep),
        ]

        # 2.2 CRI-MS: pass BOTH mid and T explicitly (cri_ms.py doesn't auto-append T)
        cri_out = os.path.join(args.out_root, f"cri_T{T}.pt")
        cmd_cri = [
            py, "cri_ms.py",
            "--dataset", args.dataset,
            "--seed", str(args.seed),
            "--data_dir", data_dir_T,
            "--output", cri_out,
            "--device", args.device_cri,
            "--obs_ts", obs_time_str,
        ]

        # 2.3 DHREC-MS: pass BOTH mid and T (dhrec_ms.py will ensure T included anyway)
        dhrec_out = os.path.join(args.out_root, f"dhrec_T{T}.pt")
        cmd_dhrec = [
            py, "dhrec_ms.py",
            "--dataset", args.dataset,
            "--seed", str(args.seed),
            "--data_dir", data_dir_T,
            "--output", dhrec_out,
            "--device", args.device_dhrec,
            "--b_pI0", str(args.b_pI0), "--b_pR0", str(args.b_pR0), "--b_steps", str(args.b_steps), "--b_lr", str(args.b_lr),
            "--obs_ts", obs_time_str,
        ]

        runs = [
            ("hermes", args.device_hermes, cmd_hermes, hermes_out),
            ("cri",    args.device_cri,    cmd_cri,    cri_out),
            ("dhrec",  args.device_dhrec,  cmd_dhrec,  dhrec_out),
        ]

        for method, device, cmd, out_pt in runs:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": args.dataset,
                "T": T,
                "obs_time_mid": obs_mid,
                "obs_time": obs_time_str,
                "method": method,
                "device": device,
                "seed": args.seed,
                "data_dir_used": data_dir_T,
                "data_pt_used": data_pt,
                "result_pt": out_pt,
                "cmd": " ".join(cmd),
            }

            status = "ok"
            elapsed = float("nan")
            f1 = float("nan")
            nrmse = float("nan")

            try:
                elapsed = _run_cmd(cmd)
                mets = _load_metrics_from_tester_pt(out_pt, args.dataset)
                f1, nrmse = mets["f1"], mets["nrmse"]
            except subprocess.CalledProcessError:
                status = "error"
            except FileNotFoundError:
                status = "missing_output"
            except Exception:
                status = "error"

            row.update({
                "status": status,
                "elapsed_sec": elapsed,
                "f1": f1,
                "nrmse": nrmse,
            })
            _append_csv(args.out_csv, row)

            print(f"[done] T={T} method={method:6s} status={status} "
                  f"f1={f1 if f1==f1 else float('nan'):.4f} "
                  f"nrmse={nrmse if nrmse==nrmse else float('nan'):.4f} "
                  f"sec={elapsed if elapsed==elapsed else float('nan'):.2f}")

    print(f"[ok] wrote CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
