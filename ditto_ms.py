from inc.diffus import *
from inc.nn import *
from inc.test import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'dataset name')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--data_dir', type = str, help = 'dataset folder')
    parser.add_argument('--output', type = str, help = 'output file name')
    parser.add_argument('--device', type = torch.device, help = 'torch device')
    parser.add_argument('--b_pI0', type = float, help = 'initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type = float, help = 'initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type = int, help = 'optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type = float, help = 'learning rate in diffusion parameter estimation')
    parser.add_argument('--q_steps', type = int, help = 'training steps for the proposal model')
    parser.add_argument('--q_lr', type = float, help = 'learning rate for the proposal model')
    parser.add_argument('--q_hid', type = int, help = 'hidden size of the proposal model')
    parser.add_argument('--q_gnn', type = int, help = 'number of layers of the GNN in the proposal model')
    parser.add_argument('--q_mlp', type = int, help = 'number of layers of the MLP in the proposal model')
    parser.add_argument('--q_samples', type = int, help = 'sample size to estimate the loss function of the proposal model')
    parser.add_argument('--q_zlim', type = int, help = 'a hyperparameter to stablize gradient')
    parser.add_argument('--p_coef', type = float, help = 'the coefficient gamma in the initial distribution P[y_0]')
    parser.add_argument('--t_samples', type = int, help = 'MCMC sample size')
    parser.add_argument('--t_steps', type = int, help = 'MCMC steps')
    parser.add_argument('--t_keep', type = float, help = 'moving average in MCMC')
    parser.add_argument('--obs_time', type = str, default = '', help = 'extra observed snapshot times, comma-separated, e.g., 5,7,9')
    # Multi-snapshot proposal diagnostics/safety.
    # The backward segment sampler uses rejection sampling under hard constraints.
    # If a segment is infeasible (empty support) or the proposal assigns vanishing
    # probability to feasible states, the rejection loop can otherwise run forever.
    parser.add_argument('--ms_max_rounds', type=int, default=2048,
                        help='max rejection rounds per backward step in multi-snapshot sampling')
    args = parser.parse_args()
    return args

class QNet(nn.Module):
    @classmethod
    def make(cls, data, args):
        return cls(
            eidx = data.edge_index,
            T = data.T.item(),
            hid = args.q_hid,
            gnn = args.q_gnn,
            mlp = args.q_mlp,
            n_nodes = data.num_nodes,
            zlim = args.q_zlim,
            ms_max_rounds = getattr(args, 'ms_max_rounds', 2048),
        ).to(args.device)
    def __init__(self, eidx, T, hid, gnn, mlp, n_nodes, zlim, ms_max_rounds=2048):
        super().__init__()
        self.eidx = eidx
        self.device = self.eidx.device
        self.n_nodes = n_nodes
        self.n_inf = self.n_nodes + 2
        self.n_edges = self.eidx.size(dim = 1)
        self.zlim = zlim
        # Maximum rejection rounds per backward step in multi-snapshot segment sampling.
        # This avoids infinite loops when a segment has empty/tiny support under hard constraints.
        self.ms_max_rounds = int(ms_max_rounds)
        self.T = T
        self.hid = int(hid)
        self.gnn_dep = int(gnn)
        self.mlp_dep = int(mlp)
        self.w = nn.Parameter(data = torch.randn((self.n_edges, self.hid), dtype = torch.float32, device = self.device), requires_grad = True)
        self.gnn = GNN(v_in = 1, e_in = self.hid, hid = self.hid, dep = self.gnn_dep)
        self.mlp = MLP([self.hid] * self.mlp_dep + [2 * self.T])
        self.rem = (pyg.utils.degree(self.eidx[1], num_nodes = self.n_nodes).long().unsqueeze(dim = 1) + 1).detach().clone() # (nodes, 1)
        self.neighbs = [[] for u in range(self.n_nodes)]
        for i in range(self.n_edges):
            self.neighbs[self.eidx[0, i].item()].append(self.eidx[1, i].item())
        for u in range(self.n_nodes):
            self.neighbs[u] = torch.tensor(self.neighbs[u], dtype = torch.long, device = self.device)
        self.adj = torch.sparse_coo_tensor(
            indices = torch.stack([self.eidx[1], self.eidx[0]], dim = 0),
            values = torch.ones(self.n_edges, dtype = torch.float32, device = self.device),
            size = (self.n_nodes, self.n_nodes),
        ).coalesce()
        self.zero = torch.tensor(0., dtype = torch.float, device = self.device)
    def clamp_z(self, z):
        return z.clamp(-self.zlim, self.zlim)
    def forward(self, y, orig = False): # y: (nodes, samples)
        n_nodes, n_samples = y.size()
        y = y.T.reshape((-1, 1)) # (samples*nodes, 1)
        eidx = (self.eidx.unsqueeze(dim = 1) + n_nodes * torch.arange(n_samples, dtype = torch.long, device = y.device).unsqueeze(dim = -1)).reshape((2, -1)) # (2, samples*edges)
        w = self.w.repeat(n_samples, 1) # (samples, hid)
        z, e = self.gnn(y.float(), eidx, w)
        z = self.mlp(z) # (samples*nodes, 2*T)
        z = z.T.reshape((2 * self.T, n_samples, -1)) # (2*T, samples, nodes)
        zI, zR = z[: self.T], z[self.T :] # (T, samples, nodes)
        zI, zR = zI.transpose(1, 2), zR.transpose(1, 2) # (T, nodes, samples)
        if orig:
            return zI, zR, self.clamp_z(zI), self.clamp_z(zR)
        else:
            return self.clamp_z(zI), self.clamp_z(zR)
    def lik(self, Y): # Y: (T+1, nodes, samples)
        n_samples = Y.size(dim = 2)
        zI0, zR0, zI, zR = self.forward(Y[-1], orig = True) # (T, nodes, samples)
        zI = zI.clone().detach().requires_grad_(True); zI.retain_grad()
        zR = zR.clone().detach().requires_grad_(True); zR.retain_grad()
        # R->I
        qR = torch.sigmoid(zR) # (T, nodes, samples) # prob of R->I
        lR1 = torch_log(qR) # (T, nodes, samples)
        lR0 = torch_log(1. - qR) # (T, nodes, samples)
        with torch.no_grad():
            mskR = (Y[1 :] == SIR_STATES.R) # (T, nodes, samples)
            trsR = (Y[: -1] != SIR_STATES.R) # (T, nodes, samples)
        # I->S
        zI_, uid = zI.sort(dim = 1, descending = True) # (T, nodes, samples)
        qI = torch.sigmoid(zI_) # (T, nodes, samples) # prob of I->S
        lI1 = torch_log(qI) # (T, nodes, samples)
        lI0 = torch_log(1. - qI) # (T, nodes, samples)
        with torch.no_grad():
            mskI = ((Y[1 :] >= SIR_STATES.I) & (Y[: -1] <= SIR_STATES.I)).flatten() # (T * nodes * samples)
            trsI = ((Y[: -1] != SIR_STATES.I)).flatten() # (T * nodes * samples)
            rem = torch.where(mskI, self.rem.expand(self.T, -1, n_samples).flatten(), self.n_inf) # (T * nodes * samples)
            ptr = torch.arange(self.T, dtype = torch.long, device = self.device).unsqueeze(dim = 1) * self.n_nodes # (T, 1)
            for i in range(uid.size(dim = 1)):
                uidi = (ptr + uid[:, i]).flatten() * n_samples # (T * samples)
                mski = mskI[uidi] # (T * samples)
                if mski.max():
                    trsi = trsI[uidi] # (T * samples)
                    vids, degi = [], [0]
                    for t in range(self.T):
                        for j in range(n_samples):
                            u = uid[t, i, j]
                            vids.append((t * self.n_nodes + u.unsqueeze(dim = 0)) * n_samples)
                            vid = self.neighbs[u.item()]
                            vids.append((t * self.n_nodes + vid) * n_samples)
                            degi.append(vid.size(dim = 0) + 1)
                    vids = torch.cat(vids, dim = 0) # (T * sum neighbs)
                    degi = torch.tensor(degi, dtype = torch.long, device = self.device) # (1 + T * samples)
                    indptr = degi.cumsum(dim = 0) # (1 + T * samples)
                    degi = degi[1 :] # (T * samples)
                    rems = rem.flatten()[vids] # (T * sum neighbs)
                    opti = (pysc.segment_min_csr(src = rems, indptr = indptr)[0] > 1) # (T * samples)
                    rem.flatten()[vids] = torch.where(mski.repeat_interleave(repeats = degi), torch.where(trsi.repeat_interleave(repeats = degi), rems - 1, self.n_inf), rems) # (T * sum neighbs)
                    mskI[uidi] &= opti # (T * samples)
        # likR + likI
        lik = (
            torch.where(mskR, torch.where(trsR, lR1, lR0), self.zero).reshape(-1, n_samples)
            + torch.where(
                mskI.reshape(-1, n_samples),
                torch.where(
                    trsI.reshape(-1, n_samples),
                    lI1.reshape(-1, n_samples),
                    lI0.reshape(-1, n_samples),
                ),
                self.zero,
            )
        ).sum(dim = 0) # (samples,)
        return lik, zI0, zR0, zI, zR # (samples,)
    @torch.no_grad()
    def clamp_grad(self, z0, grad):
        return torch.where(z0 < self.zlim, torch.where(z0 > -self.zlim, grad, F.relu(grad)), -F.relu(-grad))
    def backward(self, loss, zI0, zR0, zI, zR):
        loss.backward()
        z0 = torch.stack([zI0, zR0], dim = 0)
        z0.backward(torch.stack([self.clamp_grad(zI0, zI.grad), self.clamp_grad(zR0, zR.grad)], dim = 0))
    @torch.no_grad()
    def samp(self, y, zI, zR, n_samples, compute_lik = False): # y: (nodes,); zI, zR: (T, nodes, 1)
        zI, uid = zI.sort(dim = 1, descending = True) # (T, nodes, 1)
        uid = uid.squeeze(dim = 2) # (T, nodes)
        qI = torch.sigmoid(zI) # (T, nodes, 1) # prob of I->S
        xI = SIR_STATES.I - qI.expand(-1, -1, n_samples).bernoulli().long() # (T, nodes, samples) # 1 for I->S
        lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI)) # (T, nodes, samples)
        qR = torch.sigmoid(zR) # (T, nodes, 1) # prob of R->I
        xR = SIR_STATES.R - qR.expand(-1, -1, n_samples).bernoulli().long() # (T, nodes, samples) # 1 for R->I
        lR = torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR)) # (T, nodes, samples)
        y = y.unsqueeze(dim = 1).expand(-1, n_samples) # (nodes, samples)
        Y = torch.empty(self.T, self.n_nodes, n_samples, dtype = torch.long, device = self.device) # (T, nodes, samples)
        if compute_lik:
            lik = self.zero
        for t in range(self.T - 1, -1, -1):
            # R->I
            msk = (y == SIR_STATES.R) # (nodes, samples)
            y = torch.where(msk, xR[t], y) # (nodes, samples)
            if compute_lik:
                lik = lik + torch.where(msk, lR[t], self.zero).sum(dim = 0) # (samples,)
            # I->S
            msk = (y == SIR_STATES.I) # (nodes, samples)
            rem = torch.where(msk, self.rem, self.n_inf) # (nodes, samples)
            for i, u in enumerate(uid[t]):
                if msk[u].max():
                    vid = self.neighbs[u.item()] # (neighbs,)
                    opt = (rem[u] > 1) & (rem[vid].min(dim = 0).values > 1) # (samples,)
                    msk_opt = msk[u] & opt
                    y[u] = torch.where(msk_opt, xI[t, i], y[u]) # (samples,)
                    trs = (y[u] != SIR_STATES.I) # (samples,)
                    rem[u] = torch.where(msk[u], torch.where(trs, rem[u] - 1, self.n_inf), rem[u]) # (samples,)
                    rem[vid] = torch.where(msk[u].unsqueeze(dim = 0), torch.where(trs.unsqueeze(dim = 0), rem[vid] - 1, self.n_inf), rem[vid]) # (neighbs, samples)
                    msk[u] = msk_opt
            Y[t] = y
            if compute_lik:
                lik = lik + torch.where(msk[uid[t]], lI[t], self.zero).sum(dim = 0) # (samples,)
        Y = Y.detach().clone()
        if compute_lik:
            lik = lik.detach().clone()
            return Y, lik
        else:
            return Y

    @torch.no_grad()
    def ext_ok(self, x, yL, t, TL):
        d = t - TL
        ok = (x >= yL.unsqueeze(dim=1)).all(dim=0)  # (samples,)
        if not ok.max():
            return ok
        src = (yL == SIR_STATES.I).unsqueeze(dim=1)  # (nodes, 1)
        A = (x != SIR_STATES.S) & (yL != SIR_STATES.R).unsqueeze(dim=1)  # (nodes, samples)
        reach = src.expand(-1, x.size(dim=1)) & A  # (nodes, samples)
        for _ in range(d):
            nbr = (torch.sparse.mm(self.adj, reach.float()) > 0) & A  # (nodes, samples)
            reach = reach | nbr
        Iset = (x == SIR_STATES.I) & (yL != SIR_STATES.R).unsqueeze(dim=1)  # (nodes, samples)
        Rset = (x == SIR_STATES.R) & (yL != SIR_STATES.R).unsqueeze(dim=1)  # (nodes, samples)
        ok = ok & ~(Iset & ~reach).any(dim=0)
        # NOTE: allow one-step S->R (infection + recovery within the same discrete step),
        # so R-nodes at time t only need distance <= d (not d-1).
        ok = ok & ~(Rset & ~reach).any(dim=0)
        return ok

    @torch.no_grad()
    def _samp_step(self, y, zI, uid, zR, t, compute_lik=False, yL=None):
        """One backward step: sample y_t given y_{t+1}=y.

        This follows DITTO's original local-support (right-end feasibility) design via the
        `msk/rem` mechanism.

        Multi-snapshot add-on (Fix 2): if a left endpoint snapshot yL (= y_{TL}) is provided,
        we *hard clamp* the **left-monotonicity** necessary constraint directly inside the
        sampler so we never generate y_t < yL.

        Concretely (S < I < R):
          - Nodes with yL==R must stay R for all t>=TL  => disable backward R->I.
          - Nodes with yL==I must stay in {I,R}         => disable backward I->S on those nodes.

        IMPORTANT: We enforce this by masking *sampling choices*, not by post-hoc overwriting,
        so the returned `lik` remains the correct proposal log-probability.

        Parameters
        ----------
        y : LongTensor, (nodes, samples)
            Current snapshot y_{t+1} for a batch of samples.
        yL : LongTensor or None, (nodes,)
            Left observed snapshot y_{TL} (for monotonic clamping). If None, no clamping.
        """
        n_samples = y.size(dim=1)

        # Left-monotonic hard constraints (segment-wise), if provided.
        if yL is not None:
            force_R = (yL == SIR_STATES.R).unsqueeze(dim=1)  # (nodes, 1)
            force_I = (yL == SIR_STATES.I)                  # (nodes,)
        else:
            force_R = None
            force_I = None
        if compute_lik:
            lik = self.zero
        # R->I
        qR = torch.sigmoid(zR[t])  # (nodes, 1)
        xR = SIR_STATES.R - qR.expand(-1, n_samples).bernoulli().long()  # (nodes, samples)
        if compute_lik:
            lR = torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR))  # (nodes, samples)
        msk = (y == SIR_STATES.R)  # (nodes, samples)
        # Fix 2 (part 1): nodes already recovered at TL must remain R => do NOT sample R->I.
        if force_R is not None:
            msk = msk & (~force_R)
        y = torch.where(msk, xR, y)
        if compute_lik:
            lik = lik + torch.where(msk, lR, self.zero).sum(dim=0)  # (samples,)
        # I->S
        qI = torch.sigmoid(zI[t])  # (nodes, 1)  (already sorted by zI)
        xI = SIR_STATES.I - qI.expand(-1, n_samples).bernoulli().long()  # (nodes, samples)
        if compute_lik:
            lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI))  # (nodes, samples)
        msk = (y == SIR_STATES.I)  # (nodes, samples)
        rem = torch.where(msk, self.rem, self.n_inf)  # (nodes, samples)
        for i, u in enumerate(uid[t]):
            if msk[u].max():
                vid = self.neighbs[u.item()]  # (neighbs,)
                opt = (rem[u] > 1) & (rem[vid].min(dim=0).values > 1)  # (samples,)
                # Fix 2 (part 2): nodes infected at TL must never go below I => do NOT sample I->S.
                # Enforce by forcing `opt=False` (deterministic keep-I) for those nodes.
                if force_I is not None:
                    opt = opt & (~force_I[u])
                msk_opt = msk[u] & opt
                y[u] = torch.where(msk_opt, xI[i], y[u])  # (samples,)
                trs = (y[u] != SIR_STATES.I)  # (samples,)
                rem[u] = torch.where(msk[u], torch.where(trs, rem[u] - 1, self.n_inf), rem[u])
                rem[vid] = torch.where(msk[u].unsqueeze(dim=0),
                                       torch.where(trs.unsqueeze(dim=0), rem[vid] - 1, self.n_inf), rem[vid])
                msk[u] = msk_opt
        if compute_lik:
            lik = lik + torch.where(msk[uid[t]], lI, self.zero).sum(dim=0)  # (samples,)
            return y, lik
        else:
            return y

    @torch.no_grad()
    def _samp_seg(self, yR, zI, uid, zR, n_samples, TL, TR, yL=None, compute_lik=False):
        """
        Sample ONE segment (TL, TR] in *reverse* temporal order (backward sampling).

        Segment definition:
            - Left endpoint time  TL  (may be observed / clamped if yL is provided)
            - Right endpoint time TR  (always observed here; yR is the snapshot at TR)
            - We generate snapshots for times: TR-1, TR-2, ..., TL (if yL is None) or TL+1 (if yL is fixed)

        Why segment-wise?
            In multi-snapshot DITTO, we must satisfy *all* observed snapshots exactly.
            We sample each segment backward from the fixed right endpoint y_TR. However, in the multi-snapshot
            setting we also need a *left feasibility* guarantee: sampled states must still be extendable to
            match the left observed snapshot y_TL. This is the left-extendability hard constraint Ext(t).

        Parameters
        ----------
        yR : LongTensor, shape (nodes,)
            Fixed right endpoint snapshot y_{TR}.
        zI : FloatTensor, shape (T, nodes, 1)
            Proposal logits (already sorted along nodes dim) controlling backward I->S decisions.
            NOTE: must be consistent with uid.
        uid : LongTensor, shape (T, nodes)
            Node indices sorted by descending zI at each time t (DITTO's ordering trick).
        zR : FloatTensor, shape (T, nodes, 1)
            Proposal logits controlling backward R->I decisions (NOT sorted; original node order).
        n_samples : int
            How many independent histories to sample in parallel.
        TL, TR : int
            Segment endpoints (TL < TR).
        yL : LongTensor or None
            If provided, this is the *observed* left endpoint snapshot y_{TL} (hard constraint).
            In that case we will NOT sample time TL; we clamp it to yL.
        compute_lik : bool
            If True, also return log-probability under the proposal (needed by M-H acceptance).

        Returns
        -------
        Y_seg : LongTensor, shape (TR-TL, nodes, samples)
            The sampled segment snapshots in *forward* time indexing within the segment:
                Y_seg[k] corresponds to time (TL + k), for k=0..(TR-TL-1).
            If yL is provided, then Y_seg[0] == yL is included (clamped).
        lik_seg : FloatTensor, shape (samples,)   (only if compute_lik=True)
            Sum of log-probabilities of all backward steps performed inside this segment.
        """
        L = TR - TL  # number of time indices in [TL, TR) that we store in Y_seg
        Y = torch.empty(L, self.n_nodes, n_samples, dtype=torch.long, device=self.device)

        if compute_lik:
            # segment log-likelihood under the proposal Q_theta
            lik = torch.zeros(n_samples, dtype=torch.float, device=self.device)

        # Current "right" snapshot y_{t+1}. Start from fixed right endpoint y_{TR}.
        y = yR.unsqueeze(dim=1).expand(-1, n_samples)  # (nodes, samples)

        # ---------------------------------------------------------------------
        # Precompute yL-dependent tensors once per segment.
        # ---------------------------------------------------------------------
        if yL is not None:
            yL_col = yL.unsqueeze(dim=1)  # (nodes, 1) for monotonicity check x >= y_TL
            yL_not_R = (yL != SIR_STATES.R).unsqueeze(dim=1)  # (nodes, 1) exclude nodes fixed to R at TL
            src = (yL == SIR_STATES.I).unsqueeze(dim=1)  # (nodes, 1) infection sources at TL

            def ext_ok_fast(x, t):
                """
                Faster ext_ok that reuses yL-related precomputations.
                x: (nodes, samples) candidate snapshot at time t
                """
                d = t - TL
                ok = (x >= yL_col).all(dim=0)  # (samples,)
                if not ok.max():
                    return ok
                A = (x != SIR_STATES.S) & yL_not_R  # (nodes, samples)
                reach = src.expand(-1, x.size(dim=1)) & A  # (nodes, samples)
                for _ in range(d):
                    nbr = (torch.sparse.mm(self.adj, reach.float()) > 0) & A
                    new_reach = reach | nbr
                    if torch.equal(new_reach, reach):
                        reach = new_reach
                        break
                    reach = new_reach
                Iset = (x == SIR_STATES.I) & yL_not_R
                Rset = (x == SIR_STATES.R) & yL_not_R
                ok = ok & ~(Iset & ~reach).any(dim=0)
                ok = ok & ~(Rset & ~reach).any(dim=0)
                return ok
        else:
            ext_ok_fast = None

        # ---------------------------------------------------------------------
        # Segment-level empty-support check: the observed right endpoint itself
        # must be extendable from the observed left endpoint (when present).
        # ---------------------------------------------------------------------
        if ext_ok_fast is not None:
            okR = ext_ok_fast(yR.unsqueeze(dim=1), TR)  # (1,)
            if not bool(okR.item()):
                n_src = int((yL == SIR_STATES.I).sum().item())
                n_fixR = int((yL == SIR_STATES.R).sum().item())
                n_yR_I = int((yR == SIR_STATES.I).sum().item())
                n_yR_R = int((yR == SIR_STATES.R).sum().item())
                mono = bool((yR >= yL).all().item())
                raise RuntimeError(
                    "[DITTO-MS] Segment infeasible (empty support) under hard constraints. "
                    f"Segment (TL={TL}, TR={TR}, len={TR-TL}). "
                    f"Monotonic(y_TR>=y_TL)={mono}. "
                    f"#src_I@TL={n_src}, #fixed_R@TL={n_fixR}, #I@TR={n_yR_I}, #R@TR={n_yR_R}. "
                    "This typically means the observations cannot be bridged on the graph within the time budget "
                    "(e.g., src is empty/small, isolated targets, or timestamps over-constrain the diffusion)."
                )

        # ---------------------------------------------------------------------
        # First segment (TL=0) has no left-extendability constraint; keep original sampler.
        # For subsequent segments, use constructive hop-layer growth to enforce Ext(t) by construction.
        # ---------------------------------------------------------------------
        if yL is None:
            # no left constraint, sample purely by original backward local-support steps
            for k in range(L - 1, -1, -1):
                t = TL + k
                if compute_lik:
                    y, lik_t = self._samp_step(y, zI, uid, zR, t, compute_lik=True, yL=None)
                    lik = lik + lik_t
                else:
                    y = self._samp_step(y, zI, uid, zR, t, compute_lik=False, yL=None)
                Y[k] = y
            if compute_lik:
                return Y.detach().clone(), lik.detach().clone()
            else:
                return Y.detach().clone()

        # ---------------------------------------------------------------------
        # Constructive extendable sampler (Scheme A1):
        #   At each time t in (TL, TR):
        #     - pool := nodes with y_{t+1} != S and yL != R
        #     - build A_t (non-S set) by hop layers from src within pool:
        #         include all nodes within distance <= d-1
        #         optionally include some nodes at exact distance d
        #         force-include distance-d nodes that are needed to infect distance-(d+1) nodes
        #     - set y_t outside A_t to S (or R if blocked)
        #     - inside A_t, decide I/R using zR, but force I on nodes needed as infection sources
        # This eliminates the rejection loop for Ext(t).
        # ---------------------------------------------------------------------

        # Build unsorted zI (needed to derive p_add on boundary layer in original node order).
        # zI is sorted along nodes dim with permutation uid.
        zI_sorted_ = zI.squeeze(dim=2)  # (T, nodes)
        zI_unsorted = torch.empty_like(zI_sorted_)  # (T, nodes)
        for tt in range(self.T):
            zI_unsorted[tt, uid[tt]] = zI_sorted_[tt]

        # Precompute fixed masks from left observation.
        blocked = (yL == SIR_STATES.R).unsqueeze(dim=1)  # (nodes, 1)
        yL_not_R = ~blocked
        src = (yL == SIR_STATES.I).unsqueeze(dim=1)  # (nodes, 1)

        # We do NOT sample time TL itself; clamp it to yL.
        k_min = 1

        for k in range(L - 1, k_min - 1, -1):
            t = TL + k
            d = t - TL  # hop budget for Ext(t)

            y_next = y  # y_{t+1}, shape (nodes, samples)

            # pool = {u: y_{t+1,u} != S} \ blocked
            pool = (y_next != SIR_STATES.S) & yL_not_R  # (nodes, samples)

            # If any sample has non-empty pool but no src in pool, segment is infeasible for that sample.
            src_in_pool = (src.expand(-1, n_samples) & pool).any(dim=0)  # (samples,)
            if (~src_in_pool & pool.any(dim=0)).any():
                raise RuntimeError(
                    "[DITTO-MS] Constructive sampler hit infeasible intermediate state: "
                    f"at time t={t} (TL={TL},TR={TR}), pool non-empty but src not in pool for some samples. "
                    "This indicates either inconsistent observations or a bug in monotonic clamping."
                )

            # -----------------------------------------------------------------
            # Hop-layer BFS within pool from src, up to d+1 layers.
            # We need:
            #   - reach_{d-1}: nodes within dist <= d-1 (must be non-S at time t to support outer growth)
            #   - layer_d: nodes at exact dist d (optional, but some are forced)
            #   - layer_{d+1}: nodes at exact dist d+1 (cannot be non-S at time t; must be new at t+1)
            # -----------------------------------------------------------------
            frontier = src.expand(-1, n_samples) & pool  # layer 0
            reach = frontier.clone()
            reach_dminus1 = frontier.clone()  # will be overwritten if d-1 >= 1
            layer_d = torch.zeros_like(frontier)
            layer_d1 = torch.zeros_like(frontier)

            # Special: if d-1 == 0, then reach_dminus1 is just layer0.
            # We'll record reach after step (d-1) as reach_dminus1.
            for h in range(1, d + 2):  # compute layers 1..d+1
                nbr = (torch.sparse.mm(self.adj, frontier.float()) > 0) & pool & (~reach)
                frontier = nbr
                reach = reach | frontier
                if h == d - 1:
                    reach_dminus1 = reach.clone()
                if h == d:
                    layer_d = frontier.clone()
                if h == d + 1:
                    layer_d1 = frontier.clone()

            # If d == 1, loop sets reach_dminus1 when h==0 not visited; keep as layer0.
            if d == 1:
                reach_dminus1 = src.expand(-1, n_samples) & pool

            # Nodes at dist <= d-1 are always included in A_t (conservative constructive core).
            A = reach_dminus1.clone()

            # Force-include distance-d nodes that are adjacent to distance-(d+1) nodes,
            # because those layer_{d+1} nodes must be infected at t+1 and need an I neighbor at time t.
            if d >= 1:
                bnd_need = layer_d & (torch.sparse.mm(self.adj, layer_d1.float()) > 0)
            else:
                bnd_need = torch.zeros_like(layer_d)

            # Optional boundary nodes at dist d that are NOT needed for layer_{d+1}.
            bnd_opt = layer_d & (~bnd_need)

            # Sample add decision for optional boundary nodes using p_add = 1 - sigmoid(zI_unsorted[t]).
            # (High qI => more likely to become new, so lower include prob.)
            p_add = (1.0 - torch.sigmoid(zI_unsorted[t]).unsqueeze(dim=1)).clamp(1e-6, 1.0 - 1e-6)  # (nodes,1)
            if bnd_opt.any():
                rnd = torch.rand(self.n_nodes, n_samples, device=self.device)
                bnd_add = bnd_opt & (rnd <= p_add.expand(-1, n_samples))
            else:
                bnd_add = torch.zeros_like(bnd_opt)

            # Final A_t
            A = A | bnd_need | bnd_add

            # -----------------------------------------------------------------
            # Given A_t, construct y_t:
            #   - blocked nodes (yL==R): force R
            #   - nodes not in A and not blocked: S
            #   - nodes in A:
            #       if y_{t+1}==I => force I
            #       if y_{t+1}==R => sample I/R with qR, but force I if needed as infection source
            # -----------------------------------------------------------------
            y_t = torch.full((self.n_nodes, n_samples), SIR_STATES.S, dtype=torch.long, device=self.device)
            y_t = torch.where(blocked.expand(-1, n_samples), SIR_STATES.R, y_t)

            # new nodes are those in pool but not in A (they are S at time t, become non-S at t+1)
            new = pool & (~A)

            # Any node in A adjacent to any new node must be infected at time t to enable infection.
            need_source = A & (torch.sparse.mm(self.adj, new.float()) > 0)

            # Force I where y_{t+1}==I
            force_I_from_next = A & (y_next == SIR_STATES.I)

            # Candidate nodes with y_{t+1}==R that are not forced infection sources.
            candR = A & (y_next == SIR_STATES.R) & (~need_source) & (~force_I_from_next)

            # Sample I/R for candR using qR (prob of R->I backward => I at time t).
            if candR.any():
                qR_t = torch.sigmoid(zR[t]).clamp(1e-6, 1.0 - 1e-6)  # (nodes,1)
                rndR = torch.rand(self.n_nodes, n_samples, device=self.device)
                isI = candR & (rndR <= qR_t.expand(-1, n_samples))
                # Set sampled states
                y_t = torch.where(isI, SIR_STATES.I, y_t)
                y_t = torch.where(candR & (~isI), SIR_STATES.R, y_t)
                if compute_lik:
                    logq = torch_log(qR_t).expand(-1, n_samples)
                    log1q = torch_log(1.0 - qR_t).expand(-1, n_samples)
                    lik = lik + (torch.where(isI, logq, self.zero) + torch.where(candR & (~isI), log1q, self.zero)).sum(dim=0)
            else:
                if compute_lik:
                    pass

            # Force I for infection sources and nodes that are I at t+1
            y_t = torch.where(need_source | force_I_from_next, SIR_STATES.I, y_t)

            # -----------------------------------------------------------------
            # Proposal likelihood contributions from boundary add decisions.
            # We only account for sampled optional boundary nodes (bnd_opt).
            # Forced inclusions (reach<=d-1 and bnd_need) are deterministic (log prob 0).
            # -----------------------------------------------------------------
            if compute_lik and bnd_opt.any():
                p = p_add.expand(-1, n_samples)
                logp = torch_log(p)
                log1p = torch_log(1.0 - p)
                lik = lik + (torch.where(bnd_add, logp, self.zero) + torch.where(bnd_opt & (~bnd_add), log1p, self.zero)).sum(dim=0)

            # Commit and step left
            Y[k] = y_t
            y = y_t

        # Clamp left endpoint snapshot
        Y[0] = yL.unsqueeze(dim=1).expand(-1, n_samples)

        if compute_lik:
            return Y.detach().clone(), lik.detach().clone()
        else:
            return Y.detach().clone()

    @torch.no_grad()
    def samp_ms(self, y, zI, zR, n_samples, obs_time, compute_lik=False):

        # y: (nodes, T+1)
        obs_time = sorted(list(obs_time))
        segL = [0] + obs_time[:-1]
        segR = obs_time

        # Allocate full history tensor (we store times 0..T-1; y_T is not stored here).
        Y = torch.empty(self.T, self.n_nodes, n_samples, dtype=torch.long, device=self.device)

        # Hard constraints: directly fix observed snapshots (except the final y_T which is not in Y)
        for t in obs_time:
            if t < self.T:
                Y[t] = y[:, t].unsqueeze(dim=1).expand(-1, n_samples)

        if compute_lik:
            lik = self.zero  # will become (samples,) after first addition

        # Helper: fetch logits conditioned on the i-th observed snapshot.
        # If z has only one conditioning slice, reuse it for all segments.
        def _pick(z, i):
            # z: (T, nodes, K) or (T, nodes, 1)
            if z.size(dim=2) == 1:
                return z
            return z[:, :, i: i + 1]

        # Sample segments in reverse order (right endpoint always known).
        for i in range(len(segR) - 1, -1, -1):
            TL, TR = segL[i], segR[i]
            yR = y[:, TR]

            if TL > 0:
                yL = y[:, TL]
                start = TL + 1  # only fill (TL, TR)
            else:
                yL = None
                start = TL
            zI_R = _pick(zI, i)  # conditioned on y_TR
            zR_R = _pick(zR, i)

            if (yL is not None) and (zI.size(dim=2) > 1):
                # TL corresponds to obs_time[i-1]
                zI_L = _pick(zI, i - 1)
                zR_L = _pick(zR, i - 1)

                # Combine evidence in logit space, then clamp.
                zI_seg = self.clamp_z(zI_R + zI_L)
                zR_seg = self.clamp_z(zR_R + zR_L)
            else:
                # first segment (TL=0) OR caller provided only one logit slice
                zI_seg = zI_R
                zR_seg = zR_R

            zI_sorted, uid = zI_seg.sort(dim=1, descending=True)  # (T, nodes, 1)
            uid = uid.squeeze(dim=2)  # (T, nodes)
            if compute_lik:
                Y_seg, lik_seg = self._samp_seg(
                    yR, zI_sorted, uid, zR_seg, n_samples, TL, TR, yL=yL, compute_lik=True
                )
                if yL is not None:
                    Y[start:TR] = Y_seg[1:]  # skip the clamped y_TL
                else:
                    Y[start:TR] = Y_seg

                lik = lik + lik_seg
            else:
                Y_seg = self._samp_seg(yR, zI_sorted, uid, zR_seg, n_samples, TL, TR, yL=yL, compute_lik=False)
                if yL is not None:
                    Y[start:TR] = Y_seg[1:]
                else:
                    Y[start:TR] = Y_seg

        if compute_lik:
            return Y.detach().clone(), lik.detach().clone()
        else:
            return Y.detach().clone()


def q_loss(q_net, data, I0, bpar, n_samples):
    T = data.T.item()
    n_nodes = data.num_nodes
    Y = diffus_gen(T = T, n_nodes = n_nodes, edge_index = data.edge_index, I0 = I0, n_samples = n_samples, pI = bpar.pI, pR = bpar.pR) # (T+1, nodes, samples)
    q_liks, zI0, zR0, zI, zR = q_net.lik(Y = Y) # (samples,)
    return -q_liks.mean(), zI0, zR0, zI, zR

def q_train(data, bpar, args):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    q_net = QNet.make(data, args)
    q_net.train()
    opt = optim.AdamW(q_net.parameters(), lr = args.q_lr)
    pbar = trange(1, args.q_steps + 1)
    for step in pbar:
        opt.zero_grad()
        loss, zI0, zR0, zI, zR = q_loss(q_net, data, I0, bpar, args.q_samples)
        pbar.set_description(f'[step={step}] loss={loss.item():.4f}')
        q_net.backward(loss, zI0, zR0, zI, zR)
        opt.step()
    q_net.eval()
    return q_net

@torch.no_grad()
def t_mcmc(data, bpar, q_net, args, obs_time, keepdim=True):

    I0 = (data.y[:, 0] == 1).long().sum().item()

    obs_time = sorted(list(obs_time))
    y_obs = torch.stack([data.y[:, t] for t in obs_time], dim=1)  # (nodes, K_obs)
    zI, zR = q_net(y_obs)  # (T, nodes, K_obs)

    X, lqX = q_net.samp_ms(data.y, zI, zR, args.t_samples, obs_time=obs_time, compute_lik=True)
    lpX = diffus_liks(Y=X, edge_index=data.edge_index, I0=I0, coef=args.p_coef, pI=bpar.pI, pR=bpar.pR)

    tI_avg = data_make_t(X, SIR_STATES.I, dim=0).float().mean(dim=1, keepdim=keepdim)
    tR_avg = data_make_t(X, SIR_STATES.R, dim=0).float().mean(dim=1, keepdim=keepdim)

    pbar = trange(1, args.t_steps + 1)
    for step in pbar:
        Y, lqY = q_net.samp_ms(data.y, zI, zR, args.t_samples, obs_time=obs_time, compute_lik=True)
        lpY = diffus_liks(Y=Y, edge_index=data.edge_index, I0=I0, coef=args.p_coef, pI=bpar.pI, pR=bpar.pR)

        # Hastings acceptance
        a = torch.rand(args.t_samples, device=args.device) <= torch.exp(lpY + lqX - lpX - lqY)

        X = torch.where(a, Y, X)
        lqX = torch.where(a, lqY, lqX)
        lpX = torch.where(a, lpY, lpX)

        tI = data_make_t(X, SIR_STATES.I, dim=0).float().mean(dim=1, keepdim=keepdim)
        tR = data_make_t(X, SIR_STATES.R, dim=0).float().mean(dim=1, keepdim=keepdim)
        tI_avg = args.t_keep * tI_avg + (1.0 - args.t_keep) * tI
        tR_avg = args.t_keep * tR_avg + (1.0 - args.t_keep) * tR

    return tI_avg, tR_avg

def main(data):
    # parse obs times
    obs_time = [int(t) for t in args.obs_time.split(',') if t]
    obs_time.append(data.T.item())
    obs_time = sorted(obs_time)
    # estimate diffusion parameters
    bpar = b_estim(data, args)
    print(f'[est] pI={bpar.pI:.4f}, pR={bpar.pR:.4f}', flush = True)
    # train a proposal network
    q_net = q_train(data, bpar, args)
    # estimate transition times
    tI, tR = t_mcmc(data, bpar, q_net, args, obs_time = obs_time, keepdim = True) # (nodes, 1)
    T = data.T.item()
    tI = tI.round().long()
    tR = tR.round().long()
    # compose a history
    with torch.no_grad():
        y_pred = torch.zeros_like(data.y) # (nodes, T+1)
        y_pred.scatter_(dim = 1, index = torch.minimum(tI, data.T), src = torch.full_like(tI, 1))
        y_pred.scatter_(dim = 1, index = torch.minimum(tR, data.T), src = torch.full_like(tR, 2))
        y_pred = y_pred[:, : data.T.item()].cummax(dim = 1).values
        return y_pred

args = get_args()
tester = Tester(args.data_dir, args.device, main)
tester.test([args.dataset], seed = args.seed, rep = 1)
tester.save(args.output)