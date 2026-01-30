import sys

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
    args = parser.parse_args()
    return args

class QNet(nn.Module):
    @classmethod
    def make(cls, data, obs_time, args):
        return cls(
            eidx = data.edge_index,
            T = data.T.item(),
            n_obs = len(obs_time),
            hid = args.q_hid,
            gnn = args.q_gnn,
            mlp = args.q_mlp,
            n_nodes = data.num_nodes,
            zlim = args.q_zlim,
        ).to(args.device)
    def __init__(self, eidx, T, n_obs, hid, gnn, mlp, n_nodes, zlim):
        super().__init__()
        self.eidx = eidx
        self.device = self.eidx.device
        self.n_nodes = n_nodes
        self.n_inf = self.n_nodes + 2
        self.n_edges = self.eidx.size(dim = 1)
        self.zlim = zlim
        # Maximum rejection rounds per backward step in multi-snapshot segment sampling.
        # This avoids infinite loops when a segment has empty/tiny support under hard constraints.
        self.T = T
        self.hid = int(hid)
        self.gnn_dep = int(gnn)
        self.mlp_dep = int(mlp)
        self.w = nn.Parameter(data = torch.randn((self.n_edges, self.hid), dtype = torch.float32, device = self.device), requires_grad = True)
        self.gnn = GNN(v_in = n_obs, e_in = self.hid, hid = self.hid, dep = self.gnn_dep)
        self.mlp = MLP([self.hid] * self.mlp_dep + [2 * self.T])
        self.rem = (pyg.utils.degree(self.eidx[1], num_nodes = self.n_nodes).long().unsqueeze(dim = 1) + 1).detach().clone() # (nodes, 1)
        self.neighbs = [[] for u in range(self.n_nodes)]
        for i in range(self.n_edges):
            self.neighbs[self.eidx[0, i].item()].append(self.eidx[1, i].item())
        for u in range(self.n_nodes):
            self.neighbs[u] = torch.tensor(self.neighbs[u], dtype = torch.long, device = self.device)
        self.adj = torch.sparse_coo_tensor(
            indices = torch.stack([self.eidx[1], self.eidx[0]], dim = 0),
            values = torch.ones(self.n_edges, dtype = torch.float, device = self.device),
            size = (self.n_nodes, self.n_nodes),
        ).coalesce()
        self.zero = torch.tensor(0., dtype = torch.float, device = self.device)
    def clamp_z(self, z):
        return z.clamp(-self.zlim, self.zlim)
    def forward(self, y, orig = False): # y: (nodes, samples, obs)
        n_nodes, n_samples, n_obs = y.size()
        y = y.permute(1, 0, 2).flatten(end_dim = 1) # (samples*nodes, obs)
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
    def _lik_step(self, y0, y1, lI1, lI0, lR1, lR0, yL=None, reach=None): # y*, l*, reach: (nodes, sampls)
        n_nodes, n_samples = y0.shape
        lik = self.zero
        # unreachable has log1=0
        # R->I
        msk = (y1 == SIR_STATES.R) # (nodes, samples)
        if yL is not None:
            msk = msk & (yL != SIR_STATES.R) & reach
        lik = lik + torch.where(msk, torch.where(y0 != SIR_STATES.R, lR1, lR0), self.zero)
        # I->S
        uid = lI1.argsort(dim = 0, descending = True) # (nodes, samples)
        msk = (y1 == SIR_STATES.I) | (msk & (y0 != SIR_STATES.R)) # (nodes, samples)
        rem = torch.where(msk, (reach.long() if reach is not None else 1) + torch.sparse.mm(self.adj, msk.float()).long(), self.n_inf)  # (nodes, samples)
        cols = torch.arange(n_samples, dtype = torch.long, device = rem.device)
        for i, u in enumerate(uid):
            rem_v = torch.full((self.n_nodes, n_samples), self.n_inf, dtype=rem.dtype, device=rem.device)
            rem_v = rem_v.scatter_reduce(dim=0, index=self.eidx[0, :, None].expand(-1, n_samples),
                                         src=rem[self.eidx[1]], reduce="amin", include_self=True)
            rem_v = rem_v.gather(dim=0, index=u.unsqueeze(dim=0)).squeeze(dim=0)  # (samples,)
            rem_u = rem.gather(dim = 0, index = u.unsqueeze(dim=0)).squeeze(dim=0) # (samples,)
            opt = (rem_u > 1) & (rem_v > 1) # (samples,)
            if yL is not None:
                opt = opt & (yL.gather(dim = 0, index = u.unsqueeze(dim=0)).squeeze(dim=0) != SIR_STATES.I)
            msk_u = msk.gather(dim = 0, index = u.unsqueeze(dim=0)).squeeze(dim=0) # (samples,)
            msk_opt = msk_u & opt # (samples,)
            lik = lik + torch.where(msk_opt, torch.where(y0 == SIR_STATES.S, lI1, lI0), self.zero)
            trs = (y0.gather(dim = 0, index = u.unsqueeze(dim=0)).squeeze(dim=0) == SIR_STATES.S)  # (samples,)
            rem = rem - (torch.sparse.mm(self.adj, torch.zeros(rem.size(), dtype=self.adj.dtype, device=rem.device).scatter(dim = 0, index = u.unsqueeze(dim=0), src = (msk_u & trs).unsqueeze(dim=0).to(self.adj.dtype))) > 0).to(rem.dtype) # (nodes, samples)
            # rem = rem - torch.zeros_like(rem).index_put(
            #     (self.eidx[1, :, None].expand(-1, n_samples).flatten(), cols[None].expand(self.n_edges, -1).flatten()),
            #     ((msk_u & trs) & (self.eidx[0, :, None] == u)).flatten().long(), # (edges * samples,)
            #     accumulate = False)
            rem = rem.index_put((u, cols), torch.where(msk_u, torch.where(trs, rem_u - 1, self.n_inf), rem.gather(dim = 0, index = u.unsqueeze(dim=0)).squeeze(dim=0)))
            msk = msk.index_put((u, cols), msk_opt)
        return lik
    def lik_ms(self, Y, obs_time): #(T+1, nodes, samples)
        assert Y.size(dim=0) == self.T + 1, "lik_ms expects Y with shape (T+1, nodes, samples)"
        n_samples = Y.size(dim=2)
        K = len(obs_time)
        y_cond = torch.stack([Y[t] for t in obs_time], dim=2) # (nodes, samples, obs)
        zI0, zR0, zI, zR = self.forward(y_cond, orig=True) # (T, nodes, samples)
        zI = zI.clone().detach().requires_grad_(True); zI.retain_grad()
        zR = zR.clone().detach().requires_grad_(True); zR.retain_grad()
        lik = torch.zeros(n_samples, dtype=zI.dtype, device=self.device)
        lI1, lI0 = F.logsigmoid(zI), F.logsigmoid(-zI)
        lR1, lR0 = F.logsigmoid(zR), F.logsigmoid(-zR)
        for i in range(K):
            TL, TR = obs_time[i - 1] if i > 0 else None, obs_time[i]
            if TL is None:
                for t in range(TR - 1, -1, -1):
                    lik = lik + self._lik_step(Y[t], Y[t + 1], lI1[t], lI0[t], lR1[t], lR0[t])
            else:
                yL = Y[TL] # (nodes, samples)
                L = TR - TL
                reach = torch.zeros(L, self.n_nodes, n_samples, dtype=torch.bool, device=self.device)
                reach[0] = (yL == SIR_STATES.I) # (nodes, samples)
                yL_not_R = (yL != SIR_STATES.R) # (nodes, samples)
                for d in range(1, L):
                    reach[d] = reach[d - 1] | ((torch.sparse.mm(self.adj, reach[d - 1].float()) > 0) & yL_not_R)
                for t in range(TR - 1, TL, -1):
                    lik = lik + self._lik_step(Y[t], Y[t + 1], lI1[t], lI0[t], lR1[t], lR0[t], yL = yL, reach = reach[t - TL])
        return lik, zI0, zR0, zI, zR
    @torch.no_grad()
    def clamp_grad(self, z0, grad):
        return torch.where(z0 < self.zlim, torch.where(z0 > -self.zlim, grad, F.relu(grad)), -F.relu(-grad))
    def backward(self, loss, zI0, zR0, zI, zR):
        loss.backward()
        z0 = torch.stack([zI0, zR0], dim = 0)
        z0.backward(torch.stack([self.clamp_grad(zI0, zI.grad), self.clamp_grad(zR0, zR.grad)], dim = 0))
    @torch.no_grad()
    def _samp_step(self, y, zI, zR, compute_lik=False, yL=None, reach=None):
        if reach is not None:# y: (nodes, samples); yL: (nodes,); z: (nodes, 1); reach: (nodes,)
            reach = reach.unsqueeze(dim=1) # (nodes, 1)
        n_samples = y.size(dim=1)
        if compute_lik:
            lik = self.zero
        # unreachable
        if reach is not None:
            y = torch.where(reach, y, yL.unsqueeze(dim=1))
        # R->I
        qR = torch.sigmoid(zR)  # (nodes, 1)
        xR = SIR_STATES.R - qR.expand(-1, n_samples).bernoulli().long()  # (nodes, samples)
        msk = (y == SIR_STATES.R)  # (nodes, samples)
        if yL is not None:
            msk = msk & (yL != SIR_STATES.R).unsqueeze(dim=1) & reach
        y = torch.where(msk, xR, y)
        if compute_lik:
            lik = lik + torch.where(msk, torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR)), self.zero).sum(dim=0)  # (samples,)
        # I->S
        zI, uid = zI.sort(dim = 0, descending = True)
        uid = uid.squeeze(dim = -1) # (nodes,)
        qI = torch.sigmoid(zI)  # (nodes, 1)  (already sorted by zI)
        xI = SIR_STATES.I - qI.expand(-1, n_samples).bernoulli().long()  # (nodes, samples)
        if compute_lik:
            lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI))  # (nodes, samples)
        msk = (y == SIR_STATES.I)  # (nodes, samples)
        rem = torch.where(msk, (reach.long() if reach is not None else 1) + torch.sparse.mm(self.adj, msk.float()).long(), self.n_inf)  # (nodes, samples)
        for i, u in enumerate(uid):
            if msk[u].max():
                vid = self.neighbs[u.item()]  # (neighbs,)
                rem_u = rem[u] # (samples,)
                rem_v = rem[vid] # (neighbs, samples)
                opt = (rem_u > 1) & (rem_v.min(dim=0).values > 1)  # (samples,)
                if yL is not None:
                    opt = opt & (yL[u] != SIR_STATES.I)
                msk_opt = msk[u] & opt
                y[u] = torch.where(msk_opt, xI[i], y[u])  # (samples,)
                trs = (y[u] != SIR_STATES.I)  # (samples,)
                rem[vid] = torch.where(msk[u].unsqueeze(dim=0), torch.where(trs.unsqueeze(dim=0), rem_v - 1, rem_v), rem_v)
                rem[u] = torch.where(msk[u], torch.where(trs, rem_u - 1, self.n_inf), rem[u])
                msk[u] = msk_opt
        if compute_lik:
            lik = lik + torch.where(msk[uid], lI, self.zero).sum(dim=0)  # (samples,)
            return y, lik
        else:
            return y, 0.
    @torch.no_grad()
    def _samp_seg(self, yR, zI, zR, n_samples, TL, TR, yL=None, compute_lik=False): # yR: (nodes,); zI: (T, nodes, 1); zR: (T, nodes, 1); return: Y: (TR - TL, nodes, samples), lik: (samples,)
        if compute_lik: # segment log-likelihood under the proposal Q_theta
            lik = torch.zeros(n_samples, dtype=torch.float, device=self.device)
        y = yR.unsqueeze(dim=1).expand(-1, n_samples)  # (nodes, samples)
        if yL is None: # no left constraint, sample purely by original backward local-support steps
            Y = torch.empty(TR, self.n_nodes, n_samples, dtype=torch.long, device=self.device)
            for t in range(TR - 1, -1, -1):
                y, lik_t = self._samp_step(y, zI[t], zR[t], compute_lik=compute_lik)
                if compute_lik:
                    lik = lik + lik_t
                Y[t] = y
        else:
            L = TR - TL
            Y = torch.empty(L, self.n_nodes, n_samples, dtype=torch.long, device=self.device)
            Y[0] = yL.unsqueeze(dim = -1)
            reach = torch.zeros(L, self.n_nodes, dtype=torch.bool, device=yL.device)
            reach[0] = (yL == SIR_STATES.I)

            yL_not_R = (yL != SIR_STATES.R)  # (nodes,)  FIX: keep as 1D boolean
            for d in range(1, L):
                nbr = (torch.sparse.mm(self.adj, reach[d - 1].float().unsqueeze(1)).squeeze(1) > 0)  # (nodes,)
                reach[d] = reach[d - 1] | (nbr & yL_not_R)
            for t in range(TR - 1, TL, -1):
                y, lik_t = self._samp_step(y, zI[t], zR[t], compute_lik=compute_lik, yL = yL, reach = reach[t - TL])
                if compute_lik:
                    lik = lik + lik_t
                Y[t - TL] = y
        if compute_lik:
            return Y.detach().clone(), lik.detach().clone()
        else:
            return Y.detach().clone(), 0.
    @torch.no_grad()
    def samp_ms(self, y, zI, zR, n_samples, obs_time, compute_lik=False): # z: (T, nodes, 1)
        # y: (nodes, T+1)
        Y = torch.empty(self.T, self.n_nodes, n_samples, dtype=torch.long, device=self.device)
        for t in obs_time:
            if t < self.T:
                Y[t] = y[:, t].unsqueeze(dim=1).expand(-1, n_samples)
        if compute_lik:
            lik = 0.  # will become (samples,) after first addition
        for i in range(len(obs_time) - 1, -1, -1): # Sample segments in reverse order (right endpoint always known).
            TL, TR = obs_time[i - 1] if i > 0 else None, obs_time[i]
            yL = None if TL is None else y[:, TL]
            yR = y[:, TR]
            Y[TL : TR], lik_seg = self._samp_seg(yR, zI, zR, n_samples, TL, TR, yL = yL, compute_lik = compute_lik)
            if compute_lik:
                lik = lik + lik_seg
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

def q_train(data, obs_time, bpar, args):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    q_net = QNet.make(data, obs_time, args)
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
    y_obs = torch.stack([data.y[:, t : t + 1] for t in obs_time], dim=2)  # (nodes, 1, obs)
    zI, zR = q_net(y_obs)  # (T, nodes, 1)

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
        pbar.set_description(f"[step={step}] acc={a.float().mean().item():.3f}")
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
    obs_time = sorted(set(obs_time))
    # estimate diffusion parameters
    bpar = b_estim(data, args)
    print(f'[est] pI={bpar.pI:.4f}, pR={bpar.pR:.4f}', flush = True)
    # train a proposal network
    q_net = q_train(data, obs_time, bpar, args)
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