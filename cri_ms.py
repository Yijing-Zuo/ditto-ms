from inc.diffus import *
from inc.test import *

import argparse
import numpy as np
import networkx as nx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--data_dir', type=str, help='dataset folder')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--device', type=torch.device, help='torch device')

    # Multi-snapshot options (aligned with inc/ditto_ms.py).
    parser.add_argument(
        '--obs_ts', type=str, default=None,
        help='comma-separated observed time indices, e.g. "0,3,5"; '
             'None means single-snapshot (use only the final snapshot)'
    )
    parser.add_argument(
        '--obs_k', type=int, default=None,
        help='number of observed snapshots; if None, set to len(obs_ts) when obs_ts '
             'is given, otherwise 1 (single-snapshot)'
    )

    args = parser.parse_args()
    # normalize obs_ts / obs_k
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


def _bfs_dist(G: nx.Graph, s: int, cutoff: int = None) -> dict:
    """
    Return {node: hop_distance} for nodes reachable from s.

    cutoff: optional BFS depth limit. When we only care whether dist <= T_thr,
            setting cutoff=T_thr can be much faster on large graphs.
    """
    if cutoff is None:
        return dict(nx.single_source_shortest_path_length(G, s))
    return dict(nx.single_source_shortest_path_length(G, s, cutoff=int(cutoff)))


def cri_cluster(G: nx.Graph, obs_mask: np.ndarray, T_thr: int):
    """
    CRI clustering (greedy k-center) — optimized for speed.

    Original bottleneck:
      The legacy version repeatedly called BFS from *every infected node* inside
      the greedy loop, causing huge runtimes on large VI.

    Fix (same concept, much faster):
      Run BFS only from the current centers (with cutoff=T_thr), and maintain
      each infected node's distance to its nearest center incrementally.

    Output:
      clusters: list of infected-node lists (one per center)
      VI: list of all infected nodes
    """
    n = int(obs_mask.shape[0])
    VI = [u for u in range(n) if obs_mask[u] == 1]
    if len(VI) == 0:
        return [], VI
    if len(VI) == 1:
        return [VI], VI

    # If T_thr <= 0, every infected node must be its own center (no BFS needed).
    if T_thr <= 0:
        return [[u] for u in VI], VI

    VI_arr = np.asarray(VI, dtype=np.int64)
    idx_of = {int(u): i for i, u in enumerate(VI_arr)}  # node -> index in VI

    # --- 2-BFS "double sweep" heuristic to pick an initial far-apart pair ---
    # We do NOT use cutoff here; only 2 BFS calls, so it's cheap and gives a better pair.
    u0 = int(VI_arr[0])
    dist0 = _bfs_dist(G, u0, cutoff=None)

    u1 = next((int(u) for u in VI_arr if int(u) not in dist0), None)
    if u1 is None:
        u1 = max((int(u) for u in VI_arr), key=lambda u: dist0.get(u, -1))

    dist1 = _bfs_dist(G, u1, cutoff=None)
    u2 = next((int(u) for u in VI_arr if int(u) not in dist1), None)
    if u2 is None:
        u2 = max((int(u) for u in VI_arr), key=lambda u: dist1.get(u, -1))

    centers = [u1]
    if u2 != u1:
        centers.append(u2)

    # --- Greedy k-center with incremental nearest-center distances ---
    is_center = np.zeros(len(VI_arr), dtype=bool)
    d_near = np.full(len(VI_arr), np.inf, dtype=np.float32)

    # For each center s, store only distances to infected nodes (shape: [|VI|]).
    dist_to_VI = {}  # center -> np.ndarray(|VI|,)

    def add_center(s: int):
        nonlocal d_near
        if s in dist_to_VI:
            return

        # Key speedup: cutoff=T_thr (we only need to know if dist <= T_thr)
        dist_s = _bfs_dist(G, s, cutoff=T_thr)
        ds = np.fromiter((dist_s.get(int(u), np.inf) for u in VI_arr),
                         dtype=np.float32, count=len(VI_arr))
        dist_to_VI[s] = ds

        if s in idx_of:
            is_center[idx_of[s]] = True

        d_near = np.minimum(d_near, ds)
        d_near[is_center] = 0.0

    for s in centers:
        add_center(int(s))

    with tqdm(desc='cluster', leave=False) as pbar:
        while True:
            # farthest infected node from current centers (excluding centers)
            if (~is_center).any():
                tmp = d_near.copy()
                tmp[is_center] = -1.0
                far_i = int(np.argmax(tmp))
                far_d = float(tmp[far_i])
            else:
                far_i, far_d = -1, -1.0

            if far_d <= float(T_thr):
                break

            far_node = int(VI_arr[far_i])
            centers.append(far_node)
            add_center(far_node)
            pbar.update(1)

    # --- Assign each infected node to its nearest center (vectorized) ---
    center_list = list(dict.fromkeys(centers))  # unique, keep order
    dist_mat = np.vstack([dist_to_VI[s] for s in center_list])  # (k, |VI|)
    assign = dist_mat.argmin(axis=0)  # (|VI|,)

    clusters_map = {s: [] for s in center_list}
    for i, u in enumerate(VI_arr):
        s = center_list[int(assign[i])]
        clusters_map[s].append(int(u))

    clusters = [lst for lst in clusters_map.values() if len(lst) > 0]
    return clusters, VI


def cri_rev_infect(G: nx.Graph, VI_all, Vi, y):
    """
    Reverse infection for a cluster Vi (same as legacy version, but faster):

      - Expand BFS "wavefronts" tagged by sources x in Vi (pairs (u, x)).
      - Stop once any node receives all tags.
      - Choose the best center s (min sum of tag distances).
      - For each x in Vi: set predicted infection time tI[x] = dist(s, x),
        and mark y[x, tI[x]:] = 1.

    Speed fixes:
      - Avoid allocating n empty dicts: create per-node dicts on-demand.
      - Avoid O(n) scan each layer (track max_seen incrementally).
      - Avoid final O(n) scan for candidates (track candidates during BFS).
    """
    n = G.number_of_nodes()
    ni = len(Vi)
    if ni == 0:
        return

    # g[u] is a dict {x: dist(u, x)}; allocate lazily
    g = [None] * n

    def has_label(u: int, x: int) -> bool:
        du = g[u]
        return (du is not None) and (x in du)

    frontier = set()
    for x in Vi:
        x = int(x)
        frontier.add((x, x))
        for v in G[x]:
            frontier.add((int(v), x))

    t_layer = 0
    max_seen = 0
    candidates = set()

    with tqdm(desc='rev_infect.expand', leave=False) as pbar:
        while frontier and max_seen < ni:
            next_frontier = set()
            for u, x in frontier:
                if has_label(u, x):
                    continue
                if g[u] is None:
                    g[u] = {}
                g[u][x] = t_layer

                lu = len(g[u])
                if lu > max_seen:
                    max_seen = lu
                if lu == ni:
                    candidates.add(u)

                for v in G[u]:
                    v = int(v)
                    if not has_label(v, x):
                        next_frontier.add((v, x))

            frontier = next_frontier
            t_layer += 1
            pbar.update(1)

    if not candidates:
        # Fallback (should be rare if clustering worked):
        x0 = int(Vi[0])
        y[x0, 0:] = 1
        return

    s = min(candidates, key=lambda u: sum(g[u].values()))

    # Set first-infection time for each tagged x in the cluster and mark trajectory
    tI_map = g[s]
    for x, t0 in tI_map.items():
        t0 = int(t0)
        if 0 <= x < y.shape[0]:
            t0 = max(0, min(t0, y.shape[1] - 1))  # clamp to [0, T]
            y[x, t0:] = 1


@torch.no_grad()
def cri_ms_run(data):
    """
    Multi-snapshot CRI:
      - For each observed time t_obs:
         * extract infected mask (I=1, others=0),
         * cluster with radius threshold = t_obs,
         * reverse-infect per cluster to get first-infection times,
         * mark y[:, tI: ] = 1 for those nodes,
      - Merge across snapshots by OR (sum since we fill with 1's from tI onward).
    """
    T = int(data.T.item())
    n_nodes = int(data.num_nodes)
    n_cls = int(data.y.max().item() + 1)

    # Determine observed times to use
    if args.obs_ts is not None and len(args.obs_ts) > 0:
        obs_times = [t for t in args.obs_ts if 0 <= t <= T]
        obs_times = sorted(set(obs_times))
        if len(obs_times) == 0:
            obs_times = [T]
    else:
        obs_times = [T]

    # Build graph
    G = pyg.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)

    # Accumulator for predictions
    y_pred = np.zeros((n_nodes, T + 1), dtype=np.int32)

    for t_obs in tqdm(obs_times, desc='obs_times', leave=False):
        obs_vec = (data.y[:, t_obs].cpu().detach().numpy() & 1).astype(np.int32)
        if obs_vec.sum() == 0:
            continue

        clusters, VI_all = cri_cluster(G, obs_vec, T_thr=int(t_obs))
        for Vi in tqdm(clusters, desc=f'rev_infect@t={t_obs}', leave=False):
            cri_rev_infect(G, VI_all, Vi, y_pred)

    return torch.tensor(np.minimum(y_pred, n_cls - 1), dtype=torch.long, device=data.y.device)


# ---- entry point ----
args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, cri_ms_run)
tester.test([args.dataset], rep=1)
tester.save(args.output)

