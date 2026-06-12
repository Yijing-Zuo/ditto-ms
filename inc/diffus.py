from inc.utils import *

SIR_STATES = Dict(S = 0, I = 1, R = 2)

@torch.no_grad()
def diffus_trans(edge_index, y, wI, wR): # ei: long(2, edges); y: long(nodes, samples); wI: long(edges, samples), wR: long(nodes, samples)
    # S-->I
    y = torch.maximum(y, pysc.scatter_max(src = torch.minimum(y[edge_index[0]], wI), dim = 0, index = edge_index[1], dim_size = y.size(0))[0])
    if wR is not None:
        # I-->R
        y = torch.where((y == SIR_STATES.I) & (wR != 0), SIR_STATES.R, y)
    return y

@torch.no_grad()
def diffus_sim(edge_index, y0, WI, WR = None): # y0: (nodes, samples); WI: (T, edges, samples); WR: (T, nodes, samples)
    n_nodes, n_samples = y0.size()
    T = WI.size(dim = 0)
    Y = torch.empty(size = (T + 1, n_nodes, n_samples), dtype = y0.dtype, device = y0.device)
    Y[0] = y0
    for t in range(T):
        Y[t + 1] = diffus_trans(edge_index, Y[t], WI[t], None if WR is None else WR[t])
    return Y # (T + 1, nodes, samples)

@torch.no_grad()
def diffus_gen(T, n_nodes, edge_index, I0, n_samples, pI, pR):
    n_edges = edge_index.size(dim = 1)
    y0 = torch.full((n_nodes, n_samples), SIR_STATES.S, dtype = torch.long, device = edge_index.device)
    if I0 > 0:
        idx = torch.ones(n_samples, n_nodes, device = edge_index.device).multinomial(I0, replacement = False).T # (I0, samples)
        y0.scatter_(index = idx, dim = 0, src = torch.full_like(idx, SIR_STATES.I))
    WI = (torch.rand(T, n_edges, n_samples, device = y0.device) < pI).long()
    WR = (torch.rand(T, n_nodes, n_samples, device = y0.device) < pR).long() if pR > 0 else None
    return diffus_sim(edge_index, y0, WI, WR) # (T+1, nodes, samples)
def resolve_assumed_I0(data, assumed_I0 = None):
    if assumed_I0 is None:
        I0 = (data.y[:, 0] == SIR_STATES.I).long().sum().item()
    else:
        I0 = int(assumed_I0)
    return max(0, min(int(data.num_nodes), I0))

def diffus_liks(Y, edge_index, I0, coef, pI, pR): # Y: (T+1, nodes, samples) # assuming Y feasible
    log1pI = torch_log(1. - pI) if isinstance(pI, torch.Tensor) else math_log(1. - pI)
    log_pR = torch_log(pR)      if isinstance(pR, torch.Tensor) else math_log(pR)
    log1pR = torch_log(1. - pR) if isinstance(pR, torch.Tensor) else math_log(1. - pR)
    T, n_nodes, n_samples = Y.size(); T -= 1
    liks = -coef * (((Y[0] == SIR_STATES.I).float().sum() - I0).abs() + (Y[0] == SIR_STATES.R).float().sum()) # (samples,)
    zero = torch.tensor(0., dtype = torch.float, device = Y.device)
    for t in range(T):
        # S->I
        eI = pysc.scatter_sum(src = (Y[t, edge_index[0]] == SIR_STATES.I).float(), dim = 0, index = edge_index[1], dim_size = n_nodes) # (nodes, samples)
        qI = (1. - pI) ** eI # (nodes, samples)
        liks = liks + torch.where(Y[t] == SIR_STATES.S, torch.where(Y[t + 1] >= SIR_STATES.I, torch_log(1. - qI), eI * log1pI), zero).sum(dim = 0) # (samples,)
        # I->R
        liks = liks + torch.where(Y[t + 1] == SIR_STATES.R, torch.where(Y[t] <= SIR_STATES.I, log_pR, log1pR), zero).sum(dim = 0) # (samples,)
    return liks

class BPar(nn.Module):
    def __init__(self, pI, pR, device, pImax = 0.9999):
        super().__init__()
        self.pI = nn.Parameter(torch.tensor(pI, dtype = torch.float, device = device), requires_grad = True)
        self.pR = nn.Parameter(torch.tensor(pR, dtype = torch.float, device = device), requires_grad = True)
        self.pImax = pImax
    @torch.no_grad()
    def clamp_(self):
        self.pI.clamp_(0., self.pImax)
        self.pR.clamp_(0., 1.)
    def __repr__(self, digits = 4):
        return f'pI={round(self.pI.item(), digits)} pR={round(self.pR.item(), digits)}'
    def dict(self):
        return Dict(pI = self.pI.item(), pR = self.pR.item())

def _mf_init_from_prior(data, n_nodes, device, assumed_I0 = None):
    # prior: only use I0 count (same as current code)
    I0 = resolve_assumed_I0(data, assumed_I0)
    lI = torch.full((n_nodes,), I0 / n_nodes, dtype=torch.float, device=device)
    lS = torch.full((n_nodes,), 1. - I0 / n_nodes, dtype=torch.float, device=device)
    lR = torch.zeros(n_nodes, dtype=torch.float, device=device)
    return lS, lI, lR

def _mf_init_from_snapshot(y, device):
    # hard clamp to observed snapshot y (nodes,)
    lS = (y == SIR_STATES.S).float().to(device)
    lI = (y == SIR_STATES.I).float().to(device)
    lR = (y == SIR_STATES.R).float().to(device)
    return lS, lI, lR

def b_lik(bpar, data, obs_time=None, assumed_I0 = None):
    device = data.y.device
    n_nodes = data.num_nodes
    T = data.T.item()
    ei = data.edge_index
    pI, pR = bpar.pI, bpar.pR

    # -------- obs times --------
    if obs_time is None:
        obs_time = [T]
    obs_time = sorted(set(int(t) for t in obs_time if 0 <= int(t) <= T))
    if len(obs_time) == 0 or obs_time[-1] != T:
        obs_time.append(T)

    # -------- segmented mean-field --------
    lS, lI, lR = _mf_init_from_prior(data, n_nodes, device, assumed_I0 = assumed_I0)
    t_prev = 0
    lik_total = 0.0

    for t_obs in obs_time:
        # forward from t_prev -> t_obs
        for _ in range(t_obs - t_prev):
            aI = pysc.scatter_mul(
                src=(1. - lI * pI)[ei[0]],
                dim=0,
                index=ei[1],
                dim_size=n_nodes
            )
            lS_new = lS * aI
            kI = lI + lS * (1. - aI)
            lI_new = kI * (1. - pR)
            lR_new = lR + kI * pR
            lS, lI, lR = lS_new, lI_new, lR_new

        # score snapshot at t_obs
        lik = torch.stack([lS, lI, lR], dim=0)               # (3, nodes)
        y = data.y[:, t_obs].unsqueeze(dim=0)                # (1, nodes)
        prob = lik.gather(dim=0, index=y).squeeze(dim=0)     # (nodes,)
        lik_total = lik_total + torch_log(prob).mean()

        # clamp for next segment (if any)
        lS, lI, lR = _mf_init_from_snapshot(data.y[:, t_obs], device)
        t_prev = t_obs

    # optional: normalize by number of observed frames, keeps loss scale stable
    lik_total = lik_total / len(obs_time)
    return lik_total

def b_estim(data, args, obs_time=None, assumed_I0 = None):
    T = data.T.item()
    n_cls = data.y[:, T].max().item() + 1
    pI, pR = args.b_pI0, (args.b_pR0 if n_cls == 3 else 0.)
    device = data.y.device

    # if caller doesn't pass obs_time, parse from args (compatible with current main)
    if obs_time is None and hasattr(args, "obs_time"):
        tmp = [int(t) for t in str(args.obs_time).split(',') if t]
        tmp.append(T)
        obs_time = tmp
    if assumed_I0 is None and hasattr(args, 'assumed_I0'):
        assumed_I0 = args.assumed_I0

    bpar = BPar(pI=pI, pR=pR, device=device)
    bpar.train()
    opt = optim.AdamW(bpar.parameters(), lr=args.b_lr, betas=(0.5, 0.5))
    pbar = trange(1, args.b_steps + 1)

    for step in pbar:
        opt.zero_grad()
        loss = -b_lik(bpar, data, obs_time=obs_time, assumed_I0 = assumed_I0)
        loss.backward()
        opt.step()
        bpar.clamp_()
        pbar.set_description(f'[step={step}] {bpar}')

    bpar.eval()
    return bpar.dict()

