from inc.diffus import *
from inc.test import *
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--data_dir', type=str, help='dataset folder')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--device', type=torch.device, help='torch device')

    # Diffusion parameter estimation (same as baseline)
    parser.add_argument('--b_pI0', type=float, help='initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type=float, help='initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type=int, help='optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type=float, help='learning rate in diffusion parameter estimation')

    # GCN hyperparameters (same as baseline)
    parser.add_argument('--lr', type=float, help='learning rate for GCN')
    parser.add_argument('--epochs', type=int, help='training epochs for GCN')
    parser.add_argument('--batch_size', type=int, help='batch size when training GCN')
    parser.add_argument('--units', type=int, help='hidden size of GCN')
    parser.add_argument('--layers', type=int, help='number of layers in GCN')
    parser.add_argument('--dropout', type=float, help='dropout rate in GCN')

    # === New: multi-snapshot controls (align with inc/ditto_ms.py). ===
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

    # Normalize obs_ts / obs_k exactly like inc/ditto_ms.py does. :contentReference[oaicite:8]{index=8}
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


def _select_obs_times(T: int, obs_ts: list | None) -> list:
    """
    Choose which snapshots to treat as 'observed' inputs.
    - If obs_ts is provided, we use it (clamped to [0, T]).
    - Otherwise, we default to the final snapshot only (t = T).
    """
    if obs_ts is None or len(obs_ts) == 0:
        return [T]
    times = []
    for t in obs_ts:
        if t < 0:
            t = 0
        if t > T:
            t = T
        times.append(t)
    times = sorted(set(times))
    if len(times) == 0:
        times = [T]
    return times


def _build_input_from_labels(labels: torch.Tensor, obs_times: list) -> torch.Tensor:
    """
    labels: (batch, nodes, T+1), dtype=long
    returns x: (batch*nodes, K) where K = len(obs_times)
    """
    # Stack observed snapshots as feature channels.
    xs = [labels[:, :, t] for t in obs_times]  # each: (batch, nodes)
    x = torch.stack(xs, dim=2)                 # (batch, nodes, K)
    x = x.reshape(-1, x.size(2))               # (batch*nodes, K)
    return x


def _build_input_from_data_y(data, obs_times: list) -> torch.Tensor:
    """
    data.y: (nodes, T+1), dtype=long
    returns x: (nodes, K) where K = len(obs_times)
    """
    xs = [data.y[:, t] for t in obs_times]   # each: (nodes,)
    x = torch.stack(xs, dim=1)               # (nodes, K)
    return x


def gcn_run(data):
    # Estimate diffusion parameters as in the baseline GCN. :contentReference[oaicite:9]{index=9}
    bpar = b_estim(data, args)

    # Shapes / counts
    T = data.T.item()
    n_nodes = data.num_nodes
    n_cls = data.y.max().item() + 1

    # Observed time indices used as inputs (multi-snapshot). :contentReference[oaicite:10]{index=10}
    obs_times = _select_obs_times(T, args.obs_ts)
    k_in = len(obs_times)

    # Model: input dim = #observed snapshots, output dim = T * n_cls (class per time). :contentReference[oaicite:11]{index=11}
    model = gnn.GCN(k_in, args.units, args.layers, T * n_cls, args.dropout)
    model = model.to(args.device)

    # ----------------------------
    # Train (self-supervised on synthetic data from bpar).
    # ----------------------------
    model.train()
    I0 = (data.y[:, 0] == SIR_STATES.I).long().sum().item()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    pbar = trange(1, args.epochs + 1)
    for epoch in pbar:
        opt.zero_grad()

        # Simulate histories with estimated parameters; arrange as (batch, nodes, T+1).
        labels = diffus_gen(
            T=data.T.item(), n_nodes=data.num_nodes, edge_index=data.edge_index,
            I0=I0, n_samples=args.batch_size, pI=bpar.pI, pR=bpar.pR
        ).transpose(0, 2)  # (batch, nodes, T + 1)

        # Build per-node, multi-snapshot inputs from the chosen observed times.
        x = _build_input_from_labels(labels, obs_times)  # (batch*nodes, K)
        x = x.float()

        # Replicate edges across batch. (2, E*batch)
        edge_index = (
            data.edge_index.unsqueeze(dim=2)
            + n_nodes * torch.arange(args.batch_size, dtype=torch.long, device=x.device)
        ).flatten(start_dim=1)

        # Predict all previous T states for each node (0..T-1).
        logits = F.log_softmax(model(x, edge_index).view(-1, n_cls), dim=-1)
        target = labels[:, :, :T].flatten()  # (batch*nodes*T,)
        loss = F.nll_loss(logits, target)

        pbar.set_description(f'epoch={epoch} loss={loss.item():.4f}')

        # Standard backward/update.
        loss.backward()
        opt.step()

    # ----------------------------
    # Inference (condition on the provided observed snapshots in data.y).
    # ----------------------------
    with torch.no_grad():
        model.eval()

        # Prepare inference inputs from observed times.
        x_inf = _build_input_from_data_y(data, obs_times).float()  # (nodes, K)
        y_logits = model(x_inf, data.edge_index).view(n_nodes, T, n_cls)  # (nodes, T, n_cls)
        y_pred = y_logits.argmax(dim=2).contiguous()  # (nodes, T), dtype=long

        # Optional: enforce hard consistency on any observed snapshots that lie inside [0, T-1].
        # (If an observed snapshot includes the final T, it is *input* only; we do not predict y_T.)
        for t in obs_times:
            if 0 <= t < T:
                y_pred[:, t] = data.y[:, t]

        return y_pred.clone()


args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, gcn_run)
tester.test([args.dataset], rep=1)
tester.save(args.output)
