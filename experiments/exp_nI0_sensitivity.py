import os
import sys
import os.path as osp

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inc.data import *
from inc.test import TEST_METRICS
from hermes import run_hermes

DEFAULT_DATASETS = ['ba-si', 'ba-sir']
DEFAULT_RATIOS = [0.5, 0.75, 1.0, 1.25, 1.5]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type = str, default = ','.join(DEFAULT_DATASETS), help = 'comma-separated dataset names')
    parser.add_argument('--ratios', type = str, default = ','.join(map(str, DEFAULT_RATIOS)), help = 'comma-separated I0 ratios')
    parser.add_argument('--seed', type = int, default = 12345, help = 'random seed reused across all runs')
    parser.add_argument('--data_dir', type = str, required = True, help = 'dataset folder')
    parser.add_argument('--output', type = str, required = True, help = 'output csv file name')
    parser.add_argument('--device', type = torch.device, default = torch_device(), help = 'torch device')

    # same HERMES hyperparameters as hermes.py
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
    return parser.parse_args()


def parse_list(raw, cast_fn):
    return [cast_fn(x.strip()) for x in raw.split(',') if x.strip()]


def ratio_to_I0(true_I0, ratio, n_nodes):
    assumed_I0 = int(np.floor(true_I0 * ratio + 0.5))
    return max(0, min(int(n_nodes), assumed_I0))


def run_one(data, args, ratio):
    true_I0 = resolve_assumed_I0(data, None)
    assumed_I0 = ratio_to_I0(true_I0, ratio, data.num_nodes)
    if args.seed is not None:
        seed_all(args.seed)

    y_pred, extra = run_hermes(data, args, assumed_I0 = assumed_I0, return_extra = True)

    return Dict(
        dataset = data.name,
        assumed_I0 = assumed_I0,
        ratio = ratio,
        estimated_pI = extra.pI,
        estimated_pR = extra.pR,
        F1 = TEST_METRICS['f1'](data, y_pred),
        NRMSE = TEST_METRICS['nrmse'](data, y_pred),
    )


def main():
    args = get_args()
    datasets = parse_list(args.datasets, str)
    ratios = parse_list(args.ratios, float)

    rows = []
    for dataset in datasets:
        data = data_load(dataset, args.data_dir, args.device)
        for ratio in ratios:
            print(f'[dataset={dataset}] [ratio={ratio}]', flush = True)
            row = run_one(data, args, ratio)
            rows.append(dict(row))
            print(
                f"[dataset={dataset}] [ratio={ratio}] assumed_I0={row.assumed_I0} "
                f"pI={row.estimated_pI:.4f} pR={row.estimated_pR:.4f} "
                f"F1={row.F1:.4f} NRMSE={row.NRMSE:.4f}",
                flush = True,
            )
            gc.collect()
            if getattr(args.device, 'type', None) == 'cuda':
                torch.cuda.empty_cache()

    cols = ['dataset', 'assumed_I0', 'ratio', 'estimated_pI', 'estimated_pR', 'F1', 'NRMSE']
    df = pd.DataFrame(rows, columns = cols)

    out_dir = osp.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok = True)
    df.to_csv(args.output, index = False)
    print(f'[saved] {args.output}', flush = True)


if __name__ == '__main__':
    main()