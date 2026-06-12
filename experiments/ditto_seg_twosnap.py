from inc.diffus import *
from inc.test import *
from ditto import run_ditto_on_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = None, help = 'single dataset name')
    parser.add_argument('--datasets', type = str, default = None, help = 'comma-separated dataset names')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--data_dir', type = str, help = 'dataset folder')
    parser.add_argument('--output', type = str, help = 'output file name')
    parser.add_argument('--device', type = torch.device, help = 'torch device')

    # segment-and-stitch specific
    parser.add_argument('--split_time', type = int, default = None,
                        help = 'global split time; default=floor(T/2)')

    # same DITTO args as ditto.py
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
    parser.add_argument('--q_zlim', type = int, help = 'a hyperparameter to stabilize gradient')

    parser.add_argument('--p_coef', type = float, help = 'the coefficient gamma in the initial distribution P[y_0]')
    parser.add_argument('--t_samples', type = int, help = 'MCMC sample size')
    parser.add_argument('--t_steps', type = int, help = 'MCMC steps')
    parser.add_argument('--t_keep', type = float, help = 'moving average in MCMC')

    args = parser.parse_args()

    if args.device is None:
        args.device = torch_device()

    return args


def resolve_datasets(args):
    if args.datasets is not None:
        datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]
        assert len(datasets) > 0, '--datasets is empty'
        return datasets

    assert args.dataset is not None, 'either --dataset or --datasets is required'
    return [args.dataset]


@torch.no_grad()
def make_segment_data(data, t_start, t_end):
    """
    Build a local segment subproblem from global times [t_start, t_end].

    Local time 0     <-> global time t_start
    Local time seg_T <-> global time t_end
    """
    assert 0 <= t_start < t_end <= data.T.item(), 'invalid segment boundary'

    seg_y = data.y[:, t_start : t_end + 1].detach().clone()  # (nodes, seg_T+1)

    seg = Dict(
        edge_index = data.edge_index,
        num_nodes = data.num_nodes,
        y = seg_y,
        T = torch.tensor(t_end - t_start, dtype = data.T.dtype, device = data.T.device),
        name = f'{data.name}[{t_start},{t_end}]',
    )
    return seg


@torch.no_grad()
def stitch_two_segments(y1_full, y2_full):
    """
    y1_full: global [0, ..., t_split]
    y2_full: global [t_split, ..., T]

    Direct stitching rule:
    - keep segment 1's terminal snapshot at t_split
    - append segment 2 from t_split+1 onward
    """
    y_full = torch.cat([y1_full, y2_full[:, 1:]], dim = 1)
    return y_full


def run_ditto_seg_on_data(data, args):
    """
    Naive two-snapshot extension baseline:
      1) run single-snapshot DITTO on [0, t_split], using y_{t_split} as final snapshot
      2) run single-snapshot DITTO on [t_split, T], using y_T as final snapshot
      3) directly stitch the two histories
    """
    T = data.T.item()
    split_time = args.split_time if args.split_time is not None else (T // 2)
    assert 1 <= split_time < T, f'split_time must be in [1, {T - 1}]'

    seg1 = make_segment_data(data, 0, split_time)
    seg2 = make_segment_data(data, split_time, T)

    print(f'[split] dataset={data.name} T={T} split={split_time}', flush = True)

    # segment 1: [0, split_time]
    print(f'[seg1] run DITTO on [0, {split_time}] with final snapshot y_{split_time}', flush = True)
    y1_pred = run_ditto_on_data(seg1, args)                    # (nodes, split_time)
    y1_full = torch.cat([y1_pred, seg1.y[:, -1 :]], dim = 1)  # (nodes, split_time+1)

    # segment 2: [split_time, T]
    print(f'[seg2] run DITTO on [{split_time}, {T}] with final snapshot y_{T}', flush = True)
    y2_pred = run_ditto_on_data(seg2, args)                    # (nodes, T-split_time)
    y2_full = torch.cat([y2_pred, seg2.y[:, -1 :]], dim = 1)  # (nodes, T-split_time+1)

    # stitch
    y_full = stitch_two_segments(y1_full, y2_full)             # (nodes, T+1)

    assert y_full.size(1) == T + 1, 'stitched history has wrong length'
    assert torch.equal(y_full[:, split_time], data.y[:, split_time]), 'split snapshot mismatch after stitching'
    assert torch.equal(y_full[:, -1], data.y[:, -1]), 'final snapshot mismatch after stitching'

    # Tester expects shape (nodes, T), i.e. all times except the final snapshot
    return y_full[:, : T]


if __name__ == '__main__':
    args = get_args()
    datasets = resolve_datasets(args)

    tester = Tester(
        args.data_dir,
        args.device,
        lambda data: run_ditto_seg_on_data(data, args),
    )

    tester.test(datasets, seed = args.seed, rep = 1)

    if args.output is not None:
        tester.save(args.output)