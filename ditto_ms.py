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

    def _lik_seg_local(self, Y, zI, zR, TL, TR):
        """
        Segment log-prob under the *original DITTO local-support* backward sampler.
        This is essentially the original `lik()` but restricted to t in [TL, TR-1].

        Parameters
        ----------
        Y : LongTensor, (T+1, nodes, samples)
        zI, zR : FloatTensor, (T, nodes, samples)  (already clamped / combined)
        TL, TR : int segment endpoints (TL < TR)
        """
        n_samples = Y.size(dim=2)
        L = int(TR - TL)
        if L <= 0:
            return torch.zeros(n_samples, dtype=torch.float32, device=self.device)

        # Slice the segment (TL..TR)
        Yseg = Y[TL: TR + 1]  # (L+1, nodes, samples)
        zIseg = zI[TL: TR]  # (L, nodes, samples)
        zRseg = zR[TL: TR]  # (L, nodes, samples)

        # -------------------------
        # R -> I (backward)
        # -------------------------
        qR = torch.sigmoid(zRseg)  # (L, nodes, samples)
        lR1 = torch_log(qR)
        lR0 = torch_log(1.0 - qR)
        with torch.no_grad():
            mskR = (Yseg[1:] == SIR_STATES.R)  # (L, nodes, samples)
            trsR = (Yseg[:-1] != SIR_STATES.R)  # (L, nodes, samples)

        # -------------------------
        # I -> S (backward) with DITTO ordering trick
        # -------------------------
        zI_, uid = zIseg.sort(dim=1, descending=True)  # (L, nodes, samples)
        qI = torch.sigmoid(zI_)  # (L, nodes, samples)
        lI1 = torch_log(qI)
        lI0 = torch_log(1.0 - qI)

        with torch.no_grad():
            # NOTE: this follows your existing `lik()` implementation style.
            mskI = ((Yseg[1:] >= SIR_STATES.I) & (Yseg[:-1] <= SIR_STATES.I)).flatten()
            trsI = ((Yseg[:-1] != SIR_STATES.I)).flatten()

            rem = torch.where(mskI, self.rem.expand(L, -1, n_samples).flatten(), self.n_inf)
            ptr = torch.arange(L, dtype=torch.long, device=self.device).unsqueeze(dim=1) * self.n_nodes  # (L,1)

            for i in range(uid.size(dim=1)):
                uidi = (ptr + uid[:, i]).flatten() * n_samples  # (L*samples)
                mski = mskI[uidi]
                if mski.max():
                    trsi = trsI[uidi]

                    vids, degi = [], [0]
                    for t in range(L):
                        for j in range(n_samples):
                            u = uid[t, i, j]
                            vids.append((t * self.n_nodes + u.unsqueeze(dim=0)) * n_samples)
                            vid = self.neighbs[u.item()]
                            vids.append((t * self.n_nodes + vid) * n_samples)
                            degi.append(vid.size(dim=0) + 1)

                    vids = torch.cat(vids, dim=0)
                    degi = torch.tensor(degi, dtype=torch.long, device=self.device)
                    indptr = degi.cumsum(dim=0)
                    degi = degi[1:]

                    rems = rem.flatten()[vids]
                    opti = (pysc.segment_min_csr(src=rems, indptr=indptr)[0] > 1)

                    rem.flatten()[vids] = torch.where(
                        mski.repeat_interleave(repeats=degi),
                        torch.where(trsi.repeat_interleave(repeats=degi), rems - 1, self.n_inf),
                        rems,
                    )
                    mskI[uidi] &= opti

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
        ).sum(dim=0)

        return lik

    def _lik_seg_a1(self, Y, zI, zR, TL, TR, eps=1e-6):
        S, I, R = SIR_STATES.S, SIR_STATES.I, SIR_STATES.R
        device = self.device
        n_samples = Y.shape[2]
        p_seg = self.zero.expand(n_samples).clone()


        yL = Y[TL]
        blocked = (yL == R)
        src = (yL == I)

        # IMPORTANT: only t in (TL, TR) i.e. TL+1 ... TR-1 (consistent with _samp_seg)
        for t in range(TL + 1, TR):
            d = t - TL
            y_t = Y[t]
            y_next = Y[t + 1]

            qR = torch.sigmoid(zR[t]).clamp(eps, 1.0 - eps)  # (n_nodes, 1)
            p_add = (1.0 - torch.sigmoid(zI[t])).clamp(eps, 1.0 - eps)  # (n_nodes, 1)

            pool = (y_next != S) & (~blocked)
            A_true = (y_t != S) & pool

            # replay growth to compute log prob of A_true
            A = src & pool
            frontier = A.clone()
            decided = A.clone()

            logp_A = torch.zeros(y_t.shape[1], device=device)

            for _ in range(d):
                cand = (torch.sparse.mm(self.adj, frontier.float()) > 0) & pool & (~decided)
                if not cand.any():
                    break

                add = cand & A_true
                logp_A += (add.float() * torch_log(p_add) + (cand & ~add).float() * torch_log(1 - p_add)).sum(0)

                A |= add
                frontier = add
                decided |= cand

            # if A_true contains nodes never reached by growth => prob 0
            missing = A_true & (~A)
            if missing.any():
                # set those samples to -inf
                bad = missing.any(dim=0)
                logp_A[bad] = -float("inf")

            # candR likelihood
            candR = A_true & (y_next == R)
            isI = (y_t == I)
            toI = candR & isI
            toR = candR & (~isI)  # should be R

            logp_R = (toI.float() * torch_log(qR) + toR.float() * torch_log(1 - qR)).sum(0)

            # if y_next==I and in A_true, must be I
            badI = (A_true & (y_next == I) & (y_t != I)).any(dim=0)
            if badI.any():
                logp_R[badI] = -float("inf")

            p_seg += (logp_A + logp_R)

        return p_seg

    def lik_ms(self, Y, obs_time):
        """
        Multi-snapshot proposal likelihood matching `samp_ms()`.
        """
        assert Y.size(dim=0) == self.T + 1, "lik_ms expects Y with shape (T+1, nodes, samples)"
        n_samples = Y.size(dim=2)

        # sanitize obs_time: unique, within (0..T], and must include T
        obs_time = sorted({int(t) for t in obs_time if 0 < int(t) <= self.T})
        if self.T not in obs_time:
            obs_time.append(self.T)
        K = len(obs_time)

        # Condition on all observed snapshots: concat along the "samples" dimension
        y_cond = torch.cat([Y[t] for t in obs_time], dim=1)

        # Forward once for all conditioning blocks
        zI0, zR0, zI, zR = self.forward(y_cond, orig=True)  # (T, nodes, samples*K)

        # reshape to (T, nodes, K, samples)
        zI0 = zI0.contiguous().view(self.T, self.n_nodes, K, n_samples)
        zR0 = zR0.contiguous().view(self.T, self.n_nodes, K, n_samples)
        zI = zI.contiguous().view(self.T, self.n_nodes, K, n_samples)
        zR = zR.contiguous().view(self.T, self.n_nodes, K, n_samples)

        # detach+clamp leaf trick (same training mechanism as original lik())
        zI = zI.clone().detach().requires_grad_(True)
        zI.retain_grad()
        zR = zR.clone().detach().requires_grad_(True)
        zR.retain_grad()

        lik = torch.zeros(n_samples, dtype=torch.float32, device=self.device)

        segL = [0] + obs_time[:-1]
        segR = obs_time

        for i in range(K):
            TL, TR = segL[i], segR[i]

            # logits conditioned on right endpoint snapshot y_TR (index i)
            zI_R = zI[:, :, i, :]  # (T, nodes, samples)
            zR_R = zR[:, :, i, :]

            # combine left+right logits for TL>0 as in samp_ms()
            if (TL > 0) and (K > 1):
                zI_L = zI[:, :, i - 1, :]
                zR_L = zR[:, :, i - 1, :]
                zI_seg = self.clamp_z(zI_R + zI_L)
                zR_seg = self.clamp_z(zR_R + zR_L)
            else:
                zI_seg = zI_R
                zR_seg = zR_R

            if TL == 0:
                lik = lik + self._lik_seg_local(Y, zI_seg, zR_seg, TL=TL, TR=TR)
            else:
                # _lik_seg_a1() only returns `lik` and does not accept return_ok / invalid_to_neg_inf
                lik = lik + self._lik_seg_a1(Y, zI_seg, zR_seg, TL=TL, TR=TR)

        return lik, zI0, zR0, zI, zR

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

    def _samp_seg(self, yL, yR, zI_sorted, uid, zR, TL, TR, compute_lik=False):
        S, I, R = self.S, self.I, self.R
        n_samples = yR.shape[1]
        device = self.device

        Y = torch.empty(TR - TL + 1, self.n_nodes, n_samples, dtype=torch.long, device=device)
        lik = 0.0 if compute_lik else None

        # Clamp left endpoint
        yL = yL.unsqueeze(dim=1).expand(-1, n_samples)
        yR = yR.unsqueeze(dim=1).expand(-1, n_samples)
        Y[0] = yL
        Y[TR - TL] = yR

        # TL=0 : keep original DITTO step
        if TL == 0:
            y = yR
            for t in range(TR - 1, 0, -1):
                y, p_x = self._samp_step(y, zI_sorted[t], uid[t], zR[t], compute_lik)
                Y[t] = y
                if compute_lik:
                    lik += p_x
            return Y, lik

        # -------------------------
        # TL > 0 : Route-B sampler
        # -------------------------

        # unsort zI for p_add lookup (keep your original logic)
        zI_unsorted = torch.empty_like(zI_sorted).squeeze(dim=2)  # (T, n_nodes)
        for tt in range(self.T):
            zI_unsorted[tt].scatter_(dim=0, index=uid[tt], src=zI_sorted[tt].squeeze(dim=1))

        blocked = (yL == R)
        src = (yL == I)

        for t in range(TR - 1, TL, -1):
            d = t - TL
            y_next = Y[t - TL + 1]  # (n_nodes, n_samples)

            # probabilities
            qR = torch.sigmoid(zR[t])  # (n_nodes, 1)
            p_add = 1.0 - torch.sigmoid(zI_unsorted[t]).unsqueeze(1)  # (n_nodes, 1)
            # (optional) clamp to avoid exactly 0/1
            qR = qR.clamp(self.eps, 1.0 - self.eps)
            p_add = p_add.clamp(self.eps, 1.0 - self.eps)

            pool = (y_next != S) & (~blocked)

            # ------------------------------------------------------------
            # Step 1: A_t generation WITHOUT forcing reach_{<=d-1}
            #         (outward growth from src for d hops)
            # ------------------------------------------------------------
            A = src & pool
            frontier = A.clone()
            decided = A.clone()  # nodes whose add/not-add decision has been made

            if compute_lik:
                logp_A = torch.zeros(n_samples, device=device)

            for _ in range(d):
                cand = (torch.sparse.mm(self.adj, frontier.float()) > 0) & pool & (~decided)
                if not cand.any():
                    break

                u = torch.rand(self.n_nodes, n_samples, device=device)
                add = cand & (u <= p_add)  # Bernoulli(p_add)
                if compute_lik:
                    logp_A += (add.float() * torch_log(p_add) + (cand & ~add).float() * torch_log(1 - p_add)).sum(0)

                A |= add
                frontier = add
                decided |= cand

            # ------------------------------------------------------------
            # Step 2: sample states in A, but DO NOT force all neighbors of new
            #         Enforce: each new has >=1 infected neighbor (only if otherwise 0-prob)
            # ------------------------------------------------------------
            x_t = torch.full_like(y_next, S)
            x_t[blocked] = R

            # forced I if y_next==I and in A
            mI = A & (y_next == I)
            x_t[mI] = I

            # candR nodes: y_next==R and in A => sample I/R via qR
            candR = A & (y_next == R)
            uR = torch.rand(self.n_nodes, n_samples, device=device)
            toI = candR & (uR <= qR)
            toR = candR & (~toI)
            x_t[toI] = I
            x_t[toR] = R

            if compute_lik:
                logp_R = (toI.float() * torch_log(qR) + toR.float() * torch_log(1 - qR)).sum(0)

            # new nodes
            new = pool & (~A)

            # ---- enforce infection-source constraint minimally ----
            # For each sample: if a new node has no infected neighbor, pick ONE neighbor in A and flip to I
            # (only when otherwise impossible / zero-prob forward)
            I_mask = (x_t == I)
            neighI = (torch.sparse.mm(self.adj, I_mask.float()) > 0)
            bad_new = new & (~neighI)  # nodes that violate ">=1 infected neighbor"
            if bad_new.any():
                # candidates that could be flipped to I: in A, and (y_next==I already I) OR (y_next==R and in candR)
                # (y_next==I in A are already I; so only need consider A & (y_next==R) that are currently R)
                fixable = A & (y_next == R)

                # For each bad_new node u, choose one neighbor v from fixable ∩ N(u) to flip to I
                # If none exists, that sample is infeasible under model (posterior prob 0), so leave as is (will be rejected later if you have a checker)
                neigh_fixable = (torch.sparse.mm(self.adj, fixable.float()) > 0)
                # We flip per bad_new node by sampling one neighbor index.
                # Implementation trick: do one pass "greedy-random" by picking first available neighbor per (u,sample)
                # (this avoids heavy per-node loops and still gives each choice positive prob if you randomize tie-breaking)
                # Here: random tie-breaking by multiplying adjacency mask with random noise.
                adj_dense = self.adj.to_dense()  # (n_nodes, n_nodes) maybe too big; if too big, replace with sparse gather kernels
                # NOTE: if graph is large, do not materialize dense. In that case implement sparse neighbor sampling separately.

                # For minimal code change, keep dense only if feasible in your scale.
                noise = torch.rand(self.n_nodes, self.n_nodes, device=device)
                # candidates matrix: (u,v,sample) => u in bad_new, v in fixable neighbor
                # build neighbor mask (u,v) then apply for each sample
                nb_mask = (adj_dense > 0)

                for s in range(n_samples):
                    bad_u = bad_new[:, s].nonzero(as_tuple=False).flatten()
                    if bad_u.numel() == 0:
                        continue
                    fix_v = fixable[:, s]
                    # for each bad u, pick v maximizing noise among allowed neighbors
                    for u_node in bad_u.tolist():
                        allowed = nb_mask[u_node] & fix_v
                        if allowed.any():
                            v_idx = (noise[u_node] * allowed.float()).argmax().item()
                            x_t[v_idx, s] = I

            # optional: if you want strict soundness, you can assert-check again and resample/reject,
            # but for minimal code change we just repair as above.

            Y[t - TL] = x_t

            if compute_lik:
                lik += (logp_A + logp_R)

        return Y, lik

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


def q_loss(q_net, data, I0, bpar, n_samples, obs_time):
    T = data.T.item()
    n_nodes = data.num_nodes
    Y = diffus_gen(
        T=T,
        n_nodes=n_nodes,
        edge_index=data.edge_index,
        I0=I0,
        n_samples=n_samples,
        pI=bpar.pI,
        pR=bpar.pR,
    )  # (T+1, nodes, samples)

    q_liks, zI0, zR0, zI, zR = q_net.lik_ms(Y=Y, obs_time=obs_time)
    return -q_liks.mean(), zI0, zR0, zI, zR

def q_train(data, bpar, args):
    I0 = (data.y[:, 0] == 1).long().sum().item()

    # Keep training obs_time consistent with main()/t_mcmc.
    obs_time = [int(t) for t in args.obs_time.split(',') if t]
    obs_time.append(data.T.item())
    obs_time = sorted({t for t in obs_time if 0 < t <= data.T.item()})

    q_net = QNet.make(data, args)
    q_net.train()
    opt = optim.AdamW(q_net.parameters(), lr=args.q_lr)
    pbar = trange(1, args.q_steps + 1)
    for step in pbar:
        opt.zero_grad()
        loss, zI0, zR0, zI, zR = q_loss(q_net, data, I0, bpar, args.q_samples, obs_time=obs_time)
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