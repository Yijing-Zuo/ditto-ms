from inc.diffus import *
from inc.test import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--data_dir', type=str, help='dataset folder')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--device', type=torch.device, help='torch device')

    # diffusion parameter estimation (kept as in the baseline)
    parser.add_argument('--b_pI0', type=float, help='initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type=float, help='initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type=int, help='optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type=float, help='learning rate in diffusion parameter estimation')

    # multi-snapshot options (aligned with ditto_ms.py)
    parser.add_argument(
        '--obs_ts', type=str, default=None,
        help='comma-separated observed time indices, e.g. "0,3,5"; '
             'None means single-snapshot (use only the final snapshot)'
    )
    parser.add_argument(
        '--obs_k', type=int, default=None,
        help='number of observed snapshots; if None, set to len(obs_ts) when obs_ts is given, '
             'otherwise 1 (single-snapshot)'
    )
    args = parser.parse_args()

    # parse obs_ts in the same spirit as inc/ditto_ms.py
    if args.obs_ts is not None and len(args.obs_ts.strip()) > 0:
        obs = [int(x) for x in args.obs_ts.split(',') if x.strip() != '']
        obs = sorted(set(obs))
        args.obs_ts = obs
        if args.obs_k is None:
            args.obs_k = len(obs)
    else:
        args.obs_ts = None
        if args.obs_k is None:
            args.obs_k = 1
    return args


def pcdsvc_greedy(bpar, G, y):
    """
    One-step backward inference (t -> t-1) under the PCDSVC-style greedy rule.
    This is the original single-snapshot kernel kept unchanged; it maps a snapshot y(t)
    to a previous snapshot x(t-1).

    Fix:
      - Some SI datasets legitimately have pR = 0 (no recovery). In that case, lr = -log(pR)
        should NOT be evaluated (it is unused anyway because there is no R state in y).
      - Also guard against boundary numeric issues (pI -> 1, pR -> 0) to avoid log(0).
    """
    n = G.number_of_nodes()

    # ---- numeric guards / SI-safe handling ----
    eps = 1e-12

    # pI is used in all cases; only clip the upper bound to avoid log(0) at (1 - pI).
    pI = float(getattr(bpar, 'pI', 0.0))
    if not np.isfinite(pI):
        pI = 0.0
    pI = min(max(pI, 0.0), 1.0 - eps)
    l1s = -np.log(1.0 - pI)

    # pR is only needed when there are recovered nodes in the current snapshot.
    # For SI datasets, y never contains state=2, so skip log(pR) entirely.
    if np.any(y == 2):
        pR = float(getattr(bpar, 'pR', 0.0))
        if not np.isfinite(pR):
            pR = eps
        pR = min(max(pR, eps), 1.0)
        lr = -np.log(pR)
    else:
        lr = 0.0
    # ------------------------------------------

    # x: 2 means "unknown yet / start from R" in the original code, then moves to 1 or 0
    x = np.where(y == 2, 2, 0)

    we = l1s
    ws = np.zeros(n, dtype=np.float32)
    wi = np.zeros(n, dtype=np.float32)

    for u in range(n):
        if y[u] == 0:  # S
            for v in G.neighbors(u):
                wi[v] += l1s
        elif y[u] == 1:  # I
            ws[u] -= 1.
        else:  # R
            ws[u] += lr - 1.
            wi[u] += lr

    # R --> I
    pbar = tqdm(disable=True)
    while True:
        mvs = []
        for u in range(n):
            if y[u] >= 1 and x[u] != 1:
                cur = wi[u]
                for v in G.neighbors(u):
                    if y[v] >= 1 and x[v] != 1:
                        cur -= we
                mvs.append((cur, u))
        if len(mvs) == 0:
            break
        mv = min(mvs, key=lambda mv: mv[0])
        if mv[0] >= 0.:
            dom = True
            for u in range(n):
                if y[u] == 1:
                    dm = (x[u] == 1)
                    for v in G.neighbors(u):
                        dm |= (x[v] == 1)
                        if dm:
                            break
                    dom &= dm
            if dom:
                break
        x[mv[1]] = 1
        pbar.update(1)

    # I --> S
    for u in range(n):
        if x[u] == 2 and ws[u] < 0:
            dm = False
            for v in G.neighbors(u):
                dm |= (x[v] == 1)
                if dm:
                    break
            if dm:
                x[u] = 0
                pbar.update(1)
    pbar.close()
    return x


def _parse_obs_ts(args, T):
    """
    Build a sorted & unique list of observed time indices within [0, T],
    ensuring the final snapshot T is always included as the anchor.
    """
    if args.obs_ts is None:
        obs_ts = []
    else:
        obs_ts = [t for t in args.obs_ts if 0 <= t <= T]

    if T not in obs_ts:
        obs_ts.append(T)
    obs_ts = sorted(set(obs_ts))
    return obs_ts


def _build_observation_map(data, obs_ts):
    """
    Return {t: np.array states at time t} for all observed times t.
    """
    y = data.y  # (nodes, T+1)
    obs = {}
    for t in obs_ts:
        obs[t] = y[:, t].cpu().detach().numpy()
    return obs


def pcdsvc_run(data):
    """
    Multi-snapshot backward reconstruction:
    - Split the timeline by observed time points (including T as anchor).
    - For each segment [t_prev, t_next], start from the observed y(t_next)
      and repeatedly apply the single-step greedy kernel to obtain y(t_prev+1),...,y(t_prev),
      clamping to ground-truth observation whenever we hit an observed time.
    """
    bpar = b_estim(data, args)  # estimate diffusion parameters once

    with torch.no_grad():
        T = int(data.T.item())
        n_nodes = data.num_nodes
        n_cls = int(data.y.max().item() + 1)

        # build graph & observations
        G = pyg.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
        obs_ts = _parse_obs_ts(args, T)           # ensure T is present
        obs_map = _build_observation_map(data, obs_ts)

        # init buffer and seed observed frames
        y_pred = np.zeros((n_nodes, T + 1), dtype=np.int32)
        for t in obs_ts:
            y_pred[:, t] = obs_map[t]

        # include 0 to cover the entire span; we will reconstruct down to t=0
        time_cuts = sorted(set(obs_ts + [0]))

        # process segments from later to earlier
        for idx in trange(len(time_cuts) - 1, 0, -1, desc='ms-backtrack'):
            t_prev = time_cuts[idx - 1]
            t_next = time_cuts[idx]   # t_next > t_prev
            y_cur = y_pred[:, t_next]  # this is observed or already clamped

            # step-by-step: t_next -> t_prev
            for t in range(t_next, t_prev, -1):
                y_prev = pcdsvc_greedy(bpar, G, y_cur)

                # if the new time (t-1) is observed, clamp to the observation
                if (t - 1) in obs_map:
                    y_prev = obs_map[t - 1]

                y_pred[:, t - 1] = y_prev
                y_cur = y_prev

        # clip to valid classes and return torch tensor on the original device
        return torch.tensor(np.minimum(y_pred, n_cls - 1),
                            dtype=torch.long, device=data.y.device)


args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, pcdsvc_run)
tester.test([args.dataset], rep=1)
tester.save(args.output)

