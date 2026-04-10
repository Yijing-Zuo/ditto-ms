# experiments/runtime_breakdown_ba_sparse.py
# -*- coding: utf-8 -*-

import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import runtime_breakdown_ba as base
from hermes import QNet
from inc.diffus import SIR_STATES


def apply_sparse_rhs_patch():
    @torch.no_grad()
    def _lik_step_sparse_rhs(self, y0, y1, lI1, lI0, lR1, lR0, yL=None, reach=None):
        n_nodes, n_samples = y0.shape
        lik = self.zero

        # R -> I part (unchanged)
        msk = (y1 == SIR_STATES.R)
        if yL is not None:
            msk = msk & (yL != SIR_STATES.R) & reach
        lik = lik + torch.where(
            msk,
            torch.where(y0 != SIR_STATES.R, lR1, lR0),
            self.zero,
        )

        # I -> S part
        uid = lI1.argsort(dim=0, descending=True)  # (nodes, samples)
        msk = (y1 == SIR_STATES.I) | (msk & (y0 != SIR_STATES.R))

        rem = torch.where(
            msk,
            (reach.long() if reach is not None else 1)
            + torch.sparse.mm(self.adj, msk.float()).long(),
            self.n_inf,
        )  # (nodes, samples)

        cols = torch.arange(n_samples, dtype=torch.long, device=rem.device)

        for i, u in enumerate(uid):
            rem_v = torch.full(
                (self.n_nodes, n_samples),
                self.n_inf,
                dtype=rem.dtype,
                device=rem.device,
            )
            rem_v = rem_v.scatter_reduce(
                dim=0,
                index=self.eidx[0, :, None].expand(-1, n_samples),
                src=rem[self.eidx[1]],
                reduce="amin",
                include_self=True,
            )
            rem_v = rem_v.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0)  # (samples,)
            rem_u = rem.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0)    # (samples,)

            opt = (rem_u > 1) & (rem_v > 1)
            if yL is not None:
                opt = opt & (
                    yL.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0) != SIR_STATES.I
                )

            msk_u = msk.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0)  # (samples,)
            msk_opt = msk_u & opt

            lik = lik + torch.where(
                msk_opt,
                torch.where(y0 == SIR_STATES.S, lI1, lI0),
                self.zero,
            )

            trs = (
                y0.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0)
                == SIR_STATES.S
            )  # (samples,)

            # ==========================================================
            # OLD:
            # rem = rem - (torch.sparse.mm(
            #     self.adj,
            #     torch.zeros(rem.size(), ...).scatter(...)
            # ) > 0).to(rem.dtype)
            #
            # NEW:
            # build RHS directly as sparse COO and use sparse indices.
            # ==========================================================
            active_cols = torch.nonzero(msk_u & trs, as_tuple=False).flatten()
            if active_cols.numel() > 0:
                rhs_row = u.index_select(0, active_cols)          # row index varies by sample
                rhs_col = active_cols
                rhs_idx = torch.stack([rhs_row, rhs_col], dim=0)
                rhs_val = torch.ones(
                    active_cols.numel(),
                    dtype=self.adj.dtype,
                    device=rem.device,
                )

                rhs = torch.sparse_coo_tensor(
                    indices=rhs_idx,
                    values=rhs_val,
                    size=rem.size(),   # (nodes, samples)
                    dtype=self.adj.dtype,
                    device=rem.device,
                ).coalesce()

                nbr_hit = torch.sparse.mm(self.adj, rhs)
                if nbr_hit.layout != torch.sparse_coo:
                    nbr_hit = nbr_hit.to_sparse_coo()
                nbr_hit = nbr_hit.coalesce()

                # IMPORTANT:
                # sparse tensor usually should not continue with `> 0` here.
                # Use sparse indices directly.
                if nbr_hit._nnz() > 0:
                    hit_idx = nbr_hit.indices()
                    rem.index_put_(
                        (hit_idx[0], hit_idx[1]),
                        -torch.ones(
                            hit_idx.size(1),
                            dtype=rem.dtype,
                            device=rem.device,
                        ),
                        accumulate=True,
                    )

            # self-row update (unchanged logic; in-place for less allocation)
            rem.index_put_(
                (u, cols),
                torch.where(
                    msk_u,
                    torch.where(trs, rem_u - 1, self.n_inf),
                    rem.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0),
                ),
                accumulate=False,
            )
            msk.index_put_((u, cols), msk_opt, accumulate=False)

        return lik

    QNet._lik_step = _lik_step_sparse_rhs
    print("[patch] QNet._lik_step -> sparse RHS version", flush=True)


def main():
    apply_sparse_rhs_patch()
    base.main()


if __name__ == "__main__":
    main()