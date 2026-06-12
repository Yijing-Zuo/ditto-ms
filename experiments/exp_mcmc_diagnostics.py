import gc
import os
import sys
import os.path as osp
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inc.data import *
from inc.test import TEST_METRICS
from hermes import b_estim, q_train, t_mcmc

DEFAULT_T_STEPS = [25, 50, 100, 200]
DEFAULT_REPS = 4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, required = True, help = 'dataset name')
    parser.add_argument('--seed', type = int, default = 12345, help = 'base seed used to fit bpar/q_net and derive MCMC seeds')
    parser.add_argument('--data_dir', type = str, required = True, help = 'dataset folder')
    parser.add_argument('--output_prefix', type = str, required = True, help = 'output prefix for csv/png files')
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
    parser.add_argument('--t_steps', type = int, default = 100, help = 'unused default; sweep values are controlled internally')
    parser.add_argument('--t_keep', type = float, help = 'moving average in MCMC')
    parser.add_argument('--obs_time', type = str, default = '', help = 'extra observed snapshot times, comma-separated, e.g., 5,7,9')
    return parser.parse_args()


def parse_obs_time(data, obs_time):
    out = [int(t) for t in str(obs_time).split(',') if t]
    out.append(data.T.item())
    return sorted(set(out))


@torch.no_grad()
def compose_history(data, tI, tR):
    tI = tI.round().long()
    tR = tR.round().long()
    y_pred = torch.zeros_like(data.y)
    y_pred.scatter_(dim = 1, index = torch.minimum(tI, data.T), src = torch.full_like(tI, SIR_STATES.I))
    y_pred.scatter_(dim = 1, index = torch.minimum(tR, data.T), src = torch.full_like(tR, SIR_STATES.R))
    return y_pred[:, : data.T.item()].cummax(dim = 1).values


def save_trace_plot(df, y_col, ylabel, title, fpath):
    plt.figure()
    for t_steps, group in df.groupby('t_steps'):
        mean_trace = group.groupby('step')[y_col].mean().reset_index()
        plt.plot(mean_trace['step'], mean_trace[y_col], label = f'S={t_steps}')
    plt.xlabel('MCMC step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fpath, dpi = 200)
    plt.close()


def save_metric_plot(df, metric, ylabel, title, fpath):
    stats = df.groupby('t_steps')[metric].agg(['mean', 'std']).reset_index()
    plt.figure()
    plt.errorbar(stats['t_steps'], stats['mean'], yerr = stats['std'], marker = 'o')
    plt.xlabel('t_steps (S)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fpath, dpi = 200)
    plt.close()


def main():
    args = get_args()
    out_dir = osp.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok = True)

    data = data_load(args.dataset, args.data_dir, args.device)
    obs_time = parse_obs_time(data, args.obs_time)

    # Fix parameter estimation + proposal training.
    seed_all(args.seed)
    bpar = b_estim(data, args, obs_time = obs_time)
    print(f'[est] pI={bpar.pI:.4f}, pR={bpar.pR:.4f}', flush = True)
    q_net = q_train(data, obs_time, bpar, args)

    trace_rows = []
    metric_rows = []
    mcmc_seeds = [args.seed ^ (rep + 1) for rep in range(DEFAULT_REPS)]

    for t_steps in DEFAULT_T_STEPS:
        args_run = deepcopy(args)
        args_run.t_steps = t_steps
        for rep, mcmc_seed in enumerate(mcmc_seeds):
            print(f'[dataset={args.dataset}] [S={t_steps}] [rep={rep}] [seed={mcmc_seed}]', flush = True)
            seed_all(mcmc_seed)
            tI, tR, diag_rows = t_mcmc(
                data,
                bpar,
                q_net,
                args_run,
                obs_time = obs_time,
                keepdim = True,
                diagnostics = True,
            )

            y_pred = compose_history(data, tI, tR)
            f1 = TEST_METRICS['f1'](data, y_pred)
            nrmse = TEST_METRICS['nrmse'](data, y_pred)

            metric_rows.append(dict(
                dataset = args.dataset,
                t_steps = t_steps,
                rep = rep,
                seed = mcmc_seed,
                estimated_pI = bpar.pI,
                estimated_pR = bpar.pR,
                F1 = f1,
                NRMSE = nrmse,
            ))

            for row in diag_rows:
                trace_rows.append(dict(
                    dataset = args.dataset,
                    t_steps = t_steps,
                    rep = rep,
                    seed = mcmc_seed,
                    **row,
                ))

            gc.collect()
            if getattr(args.device, 'type', None) == 'cuda':
                torch.cuda.empty_cache()

    traces = pd.DataFrame(trace_rows, columns = [
        'dataset', 't_steps', 'rep', 'seed', 'step', 'accept_rate',
        'mean_tI', 'mean_tR', 'mean_tI_avg', 'mean_tR_avg', 'mean_lp'
    ])
    metrics = pd.DataFrame(metric_rows, columns = [
        'dataset', 't_steps', 'rep', 'seed', 'estimated_pI', 'estimated_pR', 'F1', 'NRMSE'
    ])

    traces.to_csv(f'{args.output_prefix}_traces.csv', index = False)
    metrics.to_csv(f'{args.output_prefix}_metrics.csv', index = False)

    save_trace_plot(
        traces,
        y_col = 'accept_rate',
        ylabel = 'acceptance rate',
        title = f'MCMC acceptance trajectory ({args.dataset})',
        fpath = f'{args.output_prefix}_accept.png',
    )
    save_trace_plot(
        traces,
        y_col = 'mean_tI_avg',
        ylabel = 'mean infection hitting time',
        title = f'MCMC infection-time trace ({args.dataset})',
        fpath = f'{args.output_prefix}_mean_tI.png',
    )
    save_trace_plot(
        traces,
        y_col = 'mean_tR_avg',
        ylabel = 'mean recovery hitting time',
        title = f'MCMC recovery-time trace ({args.dataset})',
        fpath = f'{args.output_prefix}_mean_tR.png',
    )
    save_metric_plot(
        metrics,
        metric = 'F1',
        ylabel = 'F1',
        title = f'Final F1 vs MCMC steps ({args.dataset})',
        fpath = f'{args.output_prefix}_F1.png',
    )
    save_metric_plot(
        metrics,
        metric = 'NRMSE',
        ylabel = 'NRMSE',
        title = f'Final NRMSE vs MCMC steps ({args.dataset})',
        fpath = f'{args.output_prefix}_NRMSE.png',
    )

    print(f'[saved] {args.output_prefix}_traces.csv', flush = True)
    print(f'[saved] {args.output_prefix}_metrics.csv', flush = True)
    print(f'[saved] {args.output_prefix}_accept.png', flush = True)
    print(f'[saved] {args.output_prefix}_mean_tI.png', flush = True)
    print(f'[saved] {args.output_prefix}_mean_tR.png', flush = True)
    print(f'[saved] {args.output_prefix}_F1.png', flush = True)
    print(f'[saved] {args.output_prefix}_NRMSE.png', flush = True)


if __name__ == '__main__':
    main()