from inc.diffus import *      # diffus_gen, SIR_STATES, b_estim, etc.
from inc.test import *        # Tester
import argparse
import torch
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    # dataset & runtime
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--data_dir', type=str, help='dataset folder')
    parser.add_argument('--output', type=str, help='output file name')
    parser.add_argument('--device', type=torch.device, help='torch device')

    # diffusion parameter estimation (used to synthesize training labels)
    parser.add_argument('--b_pI0', type=float, help='initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type=float, help='initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type=int, help='optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type=float, help='learning rate in diffusion parameter estimation')
    parser.add_argument('--b_pImax', type=float, default=1.0, help='upper bound to clamp pI during estimation (safety)')

    # GIN model & training
    parser.add_argument('--lr', type=float, help='learning rate for GIN')
    parser.add_argument('--epochs', type=int, help='training epochs for GIN')
    parser.add_argument('--batch_size', type=int, help='batch size when training GIN')
    parser.add_argument('--units', type=int, help='hidden size of GIN')
    parser.add_argument('--layers', type=int, help='number of layers in GIN')
    parser.add_argument('--dropout', type=float, help='dropout rate in GIN')

    # multi-snapshot settings (align with ditto_ms.py)  :contentReference[oaicite:4]{index=4}
    parser.add_argument(
        '--obs_ts', type=str, default=None,
        help='comma-separated observed time indices, e.g. "0,3,5"; '
             'None means use obs_k snapshots ending at T (default: final-only)'
    )
    parser.add_argument(
        '--obs_k', type=int, default=None,
        help='number of observed snapshots; if None, set to len(obs_ts) when obs_ts is given, '
             'otherwise 1 (single-snapshot)'
    )
    args = parser.parse_args()

    # normalize obs_ts / obs_k following ditto_ms.py conventions  :contentReference[oaicite:5]{index=5}
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


def _resolve_obs_ts(args, T: int):
    """
    Resolve the list of observed snapshot indices for a given horizon T.
    Rules:
      - If args.obs_ts is provided: use it directly (values in [0, T] allowed; -1 means T).
      - Else: use the last args.obs_k snapshots ending at T (inclusive),
              i.e., max(0, T - obs_k + 1) ... T.
    """
    if args.obs_ts is not None:
        ts = [(T if (t == -1) else int(t)) for t in args.obs_ts]
        ts = [min(max(0, t), T) for t in ts]  # clamp into [0, T]
        ts = sorted(set(ts))
        return ts
    # default: use last k snapshots ending at T
    k = int(max(1, min(args.obs_k, T + 1)))
    start = max(0, T - k + 1)
    return list(range(start, T + 1))


def gin_run(data):
    # Estimate diffusion parameters for synthetic training labels
    bpar = b_estim(data, args)

    # Problem sizes
    T = int(data.T.item())
    n_nodes = int(data.num_nodes)
    n_cls = int(data.y.max().item() + 1)

    # Resolve observed snapshot indices for training/inference
    obs_ts = _resolve_obs_ts(args, T)           # indices in [0, T], inclusive
    k_in = int(len(obs_ts))                     # number of observed snapshots (channels)

    # Build model: GIN maps k_in-channel node features to T * n_cls logits per node
    model = gnn.GIN(k_in, args.units, args.layers, T * n_cls, args.dropout)  # :contentReference[oaicite:6]{index=6}
    model = model.to(args.device)

    # -------------------------
    # Train on simulated labels
    # -------------------------
    model.train()
    I0 = int((data.y[:, 0] == SIR_STATES.I).long().sum().item())
    opt = optim.Adam(model.parameters(), lr=args.lr)
    pbar = trange(1, args.epochs + 1)

    for epoch in pbar:
        opt.zero_grad()

        # Simulate training batch: (T+1, nodes, batch) -> transpose -> (batch, nodes, T+1)
        labels = diffus_gen(
            T=T, n_nodes=n_nodes, edge_index=data.edge_index,
            I0=I0, n_samples=args.batch_size, pI=bpar.pI, pR=bpar.pR
        ).transpose(0, 2)  # (batch, nodes, T+1)

        # Collect multi-snapshot inputs at obs_ts -> x: (batch * nodes, k_in)
        # Note: obs_ts indices are in [0, T] inclusive; labels' last dim matches that.
        x = labels[:, :, obs_ts]                            # (batch, nodes, k_in)
        x = x.reshape((-1, k_in))                           # (batch*nodes, k_in)

        # Batch-edge indexing: replicate the graph 'batch' times with node-ID offsets
        edge_index = (
            data.edge_index.unsqueeze(dim=2)
            + n_nodes * torch.arange(args.batch_size, dtype=torch.long, device=x.device)
        ).flatten(start_dim=1)  # (2, batch * n_edges)

        # Forward & loss: predict states for times 0..T-1  (exclude the final observed time T)
        logits = model(x.float(), edge_index).view(-1, n_cls)  # (batch*nodes*T, n_cls) after view below
        logits = F.log_softmax(logits, dim=-1)

        target = labels[:, :, :T].flatten()                 # (batch*nodes*T,)
        loss = F.nll_loss(logits, target)

        loss.backward()
        opt.step()

        pbar.set_description(f'epoch={epoch} loss={loss.item():.4f}')

    # -------------
    # Inference
    # -------------
    with torch.no_grad():
        model.eval()

        # Build input features from the observed snapshots of the test instance
        # data.y: (nodes, T+1)
        obs = data.y[:, obs_ts]                     # (nodes, k_in)
        y_pred = model(obs.float(), data.edge_index) \
            .view(n_nodes, T, n_cls) \
            .argmax(dim=2)                          # (nodes, T), states for times [0..T-1]

        return y_pred.clone()


if __name__ == '__main__':
    args = get_args()
    seed_all(args.seed)
    tester = Tester(args.data_dir, args.device, gin_run)
    tester.test([args.dataset], rep=1)
    tester.save(args.output)
