
import os
import sys
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import torch

from inc.test_ms import Tester
try:
    from inc.header import SIR_STATES as _SIR_STATES
    SIR_STATES = _SIR_STATES
except Exception:
    class _SIR:
        S, I, R = 0, 1, 2
    SIR_STATES = _SIR()

def parse_obs_ts(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    ts = [int(x) for x in s.split(",") if x.strip() != ""]
    ts = sorted(set(ts))
    return ts if len(ts) > 0 else None


def main_factory(obs_ts: Optional[List[int]]):
    def main(data):
        # data.y: (nodes, T+1)
        y0 = data.y
        y = y0.detach().to("cpu", dtype=torch.long)

        T = int(data.T.item()) if torch.is_tensor(data.T) else int(data.T)
        n = int(data.num_nodes)

        assert y.dim() == 2 and y.size(0) == n and y.size(1) == T + 1, \
            f"Expect y shape (nodes, T+1)=({n},{T+1}), got {tuple(y.shape)}"

        obs_set = set(obs_ts or [])
        obs_set.add(T)

        def ratio_at(t: int):
            yt = y[:, t]
            cS = int((yt == SIR_STATES.S).sum().item())
            cI = int((yt == SIR_STATES.I).sum().item())
            cR = int((yt == SIR_STATES.R).sum().item())
            tot = cS + cI + cR
            if tot == 0:
                return (0, 0, 0, 0.0, 0.0, 0.0)
            return (cS, cI, cR, cS / tot, cI / tot, cR / tot)

        print("=" * 80, flush=True)
        print(f"[SIR PROFILE] nodes={n}, T={T}", flush=True)
        print(f"[SIR PROFILE] observed ts = {sorted(obs_set)}", flush=True)
        print("-" * 80, flush=True)
        print(f"{'t':>3}  {'tag':>6}  {'S%':>8}  {'I%':>8}  {'R%':>8}   {'(S,I,R counts)':>20}", flush=True)

        # per-time
        for t in range(T + 1):
            cS, cI, cR, rS, rI, rR = ratio_at(t)
            tag = "OBS" if t in obs_set else "UNOBS"
            print(f"{t:>3}  {tag:>6}  {rS:>8.4f}  {rI:>8.4f}  {rR:>8.4f}   ({cS},{cI},{cR})", flush=True)

        # aggregate on unobserved
        unobs_ts = [t for t in range(T + 1) if t not in obs_set]
        if len(unobs_ts) == 0:
            print("-" * 80, flush=True)
            print("[SIR PROFILE] No unobserved time steps under current obs_ts.", flush=True)
            print("=" * 80, flush=True)
            return y0

        cS_all = cI_all = cR_all = 0
        for t in unobs_ts:
            cS, cI, cR, *_ = ratio_at(t)
            cS_all += cS
            cI_all += cI
            cR_all += cR
        tot_all = cS_all + cI_all + cR_all
        rS_all = cS_all / tot_all
        rI_all = cI_all / tot_all
        rR_all = cR_all / tot_all

        print("-" * 80, flush=True)
        print(f"[SIR PROFILE] UNOBS ts = {unobs_ts}", flush=True)
        print(f"[SIR PROFILE] UNOBS weighted ratio: S={rS_all:.4f}, I={rI_all:.4f}, R={rR_all:.4f}", flush=True)
        print("=" * 80, flush=True)

        return y0

    return main


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="input")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--obs_ts", type=str, default=None, help='e.g. "0,3,5,9"')
    args = ap.parse_args()

    obs_ts = parse_obs_ts(args.obs_ts)
    device = torch.device(args.device)

    tester = Tester(args.data_dir, device, main_factory(obs_ts))
    tester.test(datasets=[args.dataset], seed=args.seed, rep=1)
