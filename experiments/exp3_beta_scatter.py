import os
import sys
import csv
import argparse
from typing import Dict, Any, List, Tuple

import torch

HERE = os.path.abspath(os.path.dirname(__file__))
CANDIDATE_ROOTS = [HERE, os.path.abspath(os.path.join(HERE, ".."))]
for cand in CANDIDATE_ROOTS:
    if os.path.isdir(os.path.join(cand, "inc")):
        if cand not in sys.path:
            sys.path.insert(0, cand)
        break

from inc.data import data_load  # type: ignore
from inc.diffus import b_estim  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_datasets_arg(s: str) -> List[str]:
    s = (s or "").strip()
    if s.lower() in {"main", "default"}:
        # 8 datasets used in main experiments where true parameters are known by construction.
        return [
            "ba-si", "ba-sir",
            "er-si", "er-sir",
            "oregon2-si", "oregon2-sir",
            "prost-si", "prost-sir",
        ]
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_obs_time_arg(s: str, T: int) -> List[int]:
    """
    Parse --obs_time. If empty, use main protocol {floor(T/2), T}.
    Input format examples:
      --obs_time ""        -> [T//2, T]
      --obs_time "5"       -> [5, T]
      --obs_time "5,10"    -> [5, 10] (and ensures T included)
    """
    s = (s or "").strip()
    if not s:
        return [T // 2, T]
    ts = [int(x) for x in s.split(",") if x.strip() != ""]
    ts.append(T)
    ts = sorted(set(t for t in ts if 0 <= t <= T))
    if ts[-1] != T:
        ts.append(T)
    return ts


# -------------------------------
# Justified true parameters for MAIN synthetic datasets
# -------------------------------
def true_params_for_dataset(dataset: str) -> Tuple[float, float]:
    d = dataset.lower()
    is_sir = d.endswith("-sir")
    is_si = d.endswith("-si")
    if not (is_sir or is_si):
        raise ValueError(f"dataset must end with -si or -sir, got: {dataset}")

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
        f"Unsupported dataset for exp3 load-only scatter (needs known true beta/gamma): {dataset}"
    )


def save_points_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    fieldnames = ["dataset", "param", "beta_true", "beta_hat", "T", "obs_time"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


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

    # Observation times: if empty, use main protocol {floor(T/2), T}
    p.add_argument(
        "--obs_time",
        type=str,
        default="",
        help='Observation frames used for estimation. "" means {floor(T/2), T}. Example: "5" or "5,10".',
    )

    # b_estim hyperparams
    p.add_argument("--b_pI0", type=float, default=0.05)
    p.add_argument("--b_pR0", type=float, default=0.05)
    p.add_argument("--b_steps", type=int, default=300)
    p.add_argument("--b_lr", type=float, default=0.001)

    # output
    p.add_argument("--out_dir", type=str, default="output/exp3_beta_scatter_main")
    p.add_argument("--csv_name", type=str, default="points.csv")

    # whether to also output pR (gamma) for SIR datasets
    p.add_argument(
        "--only_betaI",
        action="store_true",
        help="If set, only output infection beta (pI).",
    )

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

    # b_estim expects these fields on args; we pass obs_time explicitly anyway.
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
        data = data_load(dataset, args.data_dir, device)
        T = int(data.T.item())
        obs_time = parse_obs_time_arg(args.obs_time, T)
        obs_time_str = ",".join(str(t) for t in obs_time)

        pI_true, pR_true = true_params_for_dataset(dataset)

        est = b_estim(data, b_args, obs_time=obs_time)  # dict with keys pI, pR

        # always output infection beta (pI)
        rows.append(
            dict(
                dataset=dataset,
                param="pI",
                beta_true=float(pI_true),
                beta_hat=float(est.get("pI", float("nan"))),
                T=T,
                obs_time=obs_time_str,
            )
        )

        # optionally output recovery beta (pR) for SIR datasets
        if (not args.only_betaI) and dataset.lower().endswith("-sir"):
            rows.append(
                dict(
                    dataset=dataset,
                    param="pR",
                    beta_true=float(pR_true),
                    beta_hat=float(est.get("pR", float("nan"))),
                    T=T,
                    obs_time=obs_time_str,
                )
            )

        print(
            "[ok] %s: T=%d, obs={%s}, true(pI=%.3f, pR=%.3f) -> hat(pI=%.4f, pR=%.4f)"
            % (
                dataset,
                T,
                obs_time_str,
                pI_true,
                pR_true,
                float(est.get("pI", 0.0)),
                float(est.get("pR", 0.0)),
            )
        )

    out_csv = os.path.join(args.out_dir, args.csv_name)
    save_points_csv(out_csv, rows)
    print("[ok] saved csv -> %s (rows=%d)" % (out_csv, len(rows)))


if __name__ == "__main__":
    main()
