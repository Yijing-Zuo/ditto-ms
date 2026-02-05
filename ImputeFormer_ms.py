# ImputeFormer_ms.py
"""ImputeFormer_ms.py

Baseline wrapper following the *gin_ms.py* CLI style and using the unified
*inc/test.py* Tester to report f1/nrmse.

What this file provides
-----------------------
- A runnable multi-snapshot imputation baseline that takes a few observed
  snapshots (times) and reconstructs the full diffusion history.
- Same tester contract as other baselines: model_fn returns y_pred with shape
  [num_nodes, T] (times 0..T-1). The tester appends the final snapshot y[:, T]
  automatically via test_fix_obs.

Observation pattern
-------------------
- Use --obs_ts (or --obs_time alias) to specify observed time indices.
  Example: --obs_ts "0,3,5" or --obs_time "5".
- Use -1 to represent T.
- If --obs_ts is not provided, use the last --obs_k snapshots ending at T.
- We ALWAYS include the final snapshot at time T as observed.

About "official" ImputeFormer
-----------------------------
You asked to *lock* this wrapper to the official ImputeFormer implementation.
In this execution environment, outbound connections to GitHub raw assets are
blocked, so I cannot vendor the upstream source code here.

Instead, this file ships a self-contained, ImputeFormer-*style* Transformer
imputer with **low-rank attention** (Linformer-like) to mimic the paper's
low-rank inductive bias. The interfaces (args, tensor shapes, tester contract)
are the important part for your ditto-ms integration.

If you later provide (or vendor) the exact upstream class file, you can replace
`LockedImputeFormer` with the official class while keeping the training/eval
pipeline unchanged.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from inc.diffus import diffus_gen, b_estim, SIR_STATES
from inc.test import Tester
from inc.utils import seed_all


# ---------------------------------------------------------------------
# Helpers: obs_time parsing (mirrors gin_ms.py; ALWAYS includes T)
# ---------------------------------------------------------------------

def _parse_int_list(s: str) -> List[int]:
    """Parse comma/space separated ints. Example: '0, 3,5' -> [0,3,5]."""
    if s is None:
        return []
    s = s.replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _resolve_obs_ts(obs_ts: Optional[List[int]], obs_k: int, T: int) -> List[int]:
    """Resolve observed snapshot indices in [0, T] (inclusive).

    - If obs_ts is provided: use it (with -1 mapped to T), clamp into [0, T], unique+sorted.
    - Else: use last obs_k snapshots ending at T.
    """
    if obs_ts is not None:
        ts: List[int] = []
        for t in obs_ts:
            if t == -1:
                t = T
            t = max(0, min(int(t), T))
            ts.append(t)
        return sorted(set(ts))

    k = max(1, min(int(obs_k), T + 1))
    start = max(0, T - k + 1)
    return list(range(start, T + 1))


def _make_obs_time(args, T: int) -> List[int]:
    """Convert CLI args into final obs_time list, ALWAYS including T."""
    obs_ts = _resolve_obs_ts(args.obs_ts, args.obs_k, T)
    obs_time = sorted(set(obs_ts + [T]))
    return obs_time


# ---------------------------------------------------------------------
# ImputeFormer-style model (low-rank attention)
# ---------------------------------------------------------------------

class LowRankSelfAttention(nn.Module):
    """Multi-head self-attention with low-rank projection over sequence length.

    This is a Linformer-like approximation that reduces O(L^2) attention to
    O(L * r) where r=proj_k.

    Input/Output:
      x: [B, L, d_model]
      out: [B, L, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        proj_k: int,
        max_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert proj_k > 0, "proj_k must be > 0"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.proj_k = proj_k
        self.max_len = max_len

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Project K/V along the sequence length dimension (L -> proj_k)
        self.E_k = nn.Parameter(torch.randn(max_len, proj_k) / math.sqrt(proj_k))
        self.E_v = nn.Parameter(torch.randn(max_len, proj_k) / math.sqrt(proj_k))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}. Increase --max_len")

        qkv = self.qkv(x)  # [B, L, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, heads, L, d_head]
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Merge heads for projection and matmul
        # [B*heads, L, d_head]
        q = q.reshape(B * self.n_heads, L, self.d_head)
        k = k.reshape(B * self.n_heads, L, self.d_head)
        v = v.reshape(B * self.n_heads, L, self.d_head)

        # Project K and V along length dimension: [B*heads, proj_k, d_head]
        Ek = self.E_k[:L, :]  # [L, proj_k]
        Ev = self.E_v[:L, :]
        k_proj = torch.einsum("bld,lk->bkd", k, Ek)
        v_proj = torch.einsum("bld,lk->bkd", v, Ev)

        # Attention: [B*heads, L, proj_k]
        attn = torch.einsum("bld,bkd->blk", q, k_proj) / math.sqrt(self.d_head)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output: [B*heads, L, d_head]
        out = torch.einsum("blk,bkd->bld", attn, v_proj)

        # Restore heads: [B, L, d_model]
        out = out.view(B, self.n_heads, L, self.d_head).transpose(1, 2).reshape(B, L, self.d_model)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class ImputeFormerBlock(nn.Module):
    """Transformer block with low-rank self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        proj_k: int,
        max_len: int,
        dropout: float,
        ffn_mult: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LowRankSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            proj_k=proj_k,
            max_len=max_len,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm for stability
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LockedImputeFormer(nn.Module):
    """A self-contained ImputeFormer-style imputer.

    It consumes the *entire* timeline (length L=T+1) with missing values masked.

    Input:
      x: [B, L, in_dim] where in_dim = n_cls + 1
         - first n_cls channels: one-hot value at observed times, 0 otherwise
         - last channel: observed mask (1 observed, 0 missing)

    Output:
      logits: [B, L, n_cls]
    """

    def __init__(
        self,
        in_dim: int,
        n_cls: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        proj_k: int,
        dropout: float,
        ffn_mult: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_cls = n_cls
        self.d_model = d_model
        self.max_len = max_len

        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [
                ImputeFormerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    proj_k=proj_k,
                    max_len=max_len,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(n_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}. Increase --max_len")

        h = self.in_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos)[None, :, :]

        for blk in self.blocks:
            h = blk(h)

        h = self.out_norm(h)
        logits = self.out_proj(h)
        return logits


# ---------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------

def _build_imputeformer_input(y: torch.Tensor, obs_time: List[int], n_cls: int) -> torch.Tensor:
    """Build model input from integer labels with a global-time observation mask.

    Args:
        y: [B, L] long (0..n_cls-1)
        obs_time: list of observed time indices in [0, L-1]
        n_cls: number of classes

    Returns:
        x: [B, L, n_cls+1] float
            - x[..., :n_cls] = one-hot(y) * obs_mask
            - x[...,  n_cls] = obs_mask
    """
    B, L = y.shape
    device = y.device

    obs_mask = torch.zeros((L,), dtype=torch.bool, device=device)
    obs_mask[obs_time] = True

    # One-hot encode and zero-out unobserved positions
    x_val = F.one_hot(y.clamp(min=0, max=n_cls - 1), num_classes=n_cls).float()  # [B, L, n_cls]
    x_val = x_val * obs_mask.view(1, L, 1).float()

    # Add mask as an extra channel
    x_mask = obs_mask.view(1, L, 1).float().expand(B, L, 1)
    x = torch.cat([x_val, x_mask], dim=-1)
    return x


# ---------------------------------------------------------------------
# Main model_fn used by Tester
# ---------------------------------------------------------------------

args = None  # set in __main__


def _call_b_estim(data, args, obs_time: List[int]):
    """Call b_estim with best-effort compatibility across branches."""
    try:
        return b_estim(data, args, obs_time=obs_time)
    except TypeError:
        # Fallback: older signature b_estim(data, args)
        # Try to inject args.obs_time (string) for compatibility.
        try:
            setattr(args, "obs_time", ",".join(map(str, obs_time)))
        except Exception:
            pass
        return b_estim(data, args)


def imputeformer_run(data) -> torch.Tensor:
    """Train on simulated diffusion sequences then impute the test sequence.

    Returns:
        y_pred: [num_nodes, T] long
    """
    global args

    device = args.device
    T = int(data.T.item())
    L = T + 1
    n_nodes = int(data.num_nodes)

    # Determine #classes (SI:2, SIR:3)
    n_cls = int(data.y.max().item()) + 1

    obs_time = _make_obs_time(args, T)

    # Estimate diffusion parameters (used for synthetic training labels)
    bpar = _call_b_estim(data, args, obs_time=obs_time)

    # Initial infected count for simulation
    I0 = int((data.y[:, 0] == SIR_STATES.I).sum().item())

    # Model
    model = LockedImputeFormer(
        in_dim=n_cls + 1,
        n_cls=n_cls,
        d_model=args.units,
        n_heads=args.heads,
        n_layers=args.layers,
        proj_k=args.proj_k,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
        max_len=max(args.max_len, L),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ----------------
    # Train on simulated diffusion sequences
    # ----------------
    for _ in trange(1, args.epochs + 1, desc="train", leave=False):
        model.train()

        # labels: [batch, n_nodes, L]
        labels = diffus_gen(
            T=T,
            n_nodes=n_nodes,
            edge_index=data.edge_index,
            I0=I0,
            n_samples=args.batch_size,
            pI=bpar.pI,
            pR=bpar.pR,
        ).transpose(0, 2)

        # Subsample nodes to bound memory
        node_batch = min(int(args.node_batch), n_nodes)
        if node_batch < n_nodes:
            idx = torch.randint(0, n_nodes, (node_batch,), device=labels.device)
        else:
            idx = torch.arange(n_nodes, device=labels.device)

        # Each (diffusion sample, node) is one training instance: y: [B2, L]
        y = labels[:, idx, :].reshape(-1, L).long()

        x = _build_imputeformer_input(y, obs_time=obs_time, n_cls=n_cls)
        logits = model(x)  # [B2, L, n_cls]

        loss = F.cross_entropy(
            logits[:, :T, :].reshape(-1, n_cls),
            y[:, :T].reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    # ----------------
    # Inference on the test instance (only obs_time snapshots are revealed)
    # ----------------
    model.eval()

    y_true = data.y[:, :L].long().to(device)  # [n_nodes, L]
    y_pred_full = torch.empty((n_nodes, L), dtype=torch.long, device=device)

    with torch.no_grad():
        bs = max(1, int(args.eval_node_batch))
        for s in range(0, n_nodes, bs):
            e = min(n_nodes, s + bs)
            y_chunk = y_true[s:e]  # [b, L]
            x_chunk = _build_imputeformer_input(y_chunk, obs_time=obs_time, n_cls=n_cls)
            logits = model(x_chunk)  # [b, L, n_cls]
            pred = logits.argmax(dim=-1)  # [b, L]

            # Enforce consistency on observed snapshots (except final; tester fixes it anyway)
            for t in obs_time:
                if t < T:
                    pred[:, t] = y_chunk[:, t]

            y_pred_full[s:e] = pred

    # Return only 0..T-1 (tester appends y[:, T])
    return y_pred_full[:, :T]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    # IO / runtime
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=torch.device, required=True)

    # Diffusion parameter estimation (same knobs as other baselines)
    parser.add_argument("--b_pI0", type=float, required=True)
    parser.add_argument("--b_pR0", type=float, required=True)
    parser.add_argument("--b_steps", type=int, required=True)
    parser.add_argument("--b_lr", type=float, required=True)
    parser.add_argument("--b_pImax", type=float, default=1.0)

    # Multi-snapshot observation settings
    parser.add_argument(
        "--obs_ts",
        type=str,
        default=None,
        help='comma-separated observed time indices. Use -1 for T. Example: "0,3,5"',
    )
    parser.add_argument(
        "--obs_time",
        type=str,
        default=None,
        help="alias of --obs_ts (kept for compatibility with other scripts).",
    )
    parser.add_argument(
        "--obs_k",
        type=int,
        default=1,
        help="if obs_ts is None, use last k snapshots ending at T (default 1: final-only).",
    )

    # Model / training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--units", type=int, default=64, help="Transformer hidden size")
    parser.add_argument("--heads", type=int, default=4, help="#attention heads")
    parser.add_argument("--layers", type=int, default=4, help="#Transformer blocks")
    parser.add_argument("--proj_k", type=int, default=16, help="low-rank projection size (Linformer k)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=64, help="max T+1")

    parser.add_argument("--grad_clip", type=float, default=1.0)

    # For large graphs
    parser.add_argument(
        "--node_batch",
        type=int,
        default=512,
        help="#nodes sampled per training epoch (each node is a sequence instance)",
    )
    parser.add_argument(
        "--eval_node_batch",
        type=int,
        default=2048,
        help="#nodes per forward pass during inference",
    )

    args = parser.parse_args()

    # obs_ts/obs_time normalization
    obs_s = args.obs_ts if args.obs_ts is not None else args.obs_time
    if obs_s is not None and len(str(obs_s).strip()) > 0:
        obs = _parse_int_list(str(obs_s))
        args.obs_ts = sorted(set(obs))
        # Note: keep args.obs_time as-is; _make_obs_time uses args.obs_ts
    else:
        args.obs_ts = None

    return args


if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)

    tester = Tester(args.data_dir, args.device, imputeformer_run)
    tester.test([args.dataset], rep=1)
    tester.save(args.output)
