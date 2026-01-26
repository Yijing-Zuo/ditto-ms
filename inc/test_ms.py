import functools as fnt
import numpy as np
import torch
from sklearn import metrics as skm

from inc.data import *

@torch.no_grad()
def _get_obs_mask(data):
    T = int(data.T.item())
    n_nodes = int(data.num_nodes)
    dev = data.y.device
    for name in ('obs_mask', 'obs_masks'):
        if hasattr(data, name):
            m = getattr(data, name)
            if isinstance(m, torch.Tensor):
                mask = m
            else:
                mask = torch.as_tensor(m)
            if mask.dim() == 1:  # (T+1,) -> (nodes, T+1)
                mask = mask.view(1, -1).expand(n_nodes, T + 1)
            mask = mask.to(device=dev, dtype=torch.bool)
            return mask
    if hasattr(data, 'obs_ts') and getattr(data, 'obs_ts') is not None:
        ts = getattr(data, 'obs_ts')
        if not isinstance(ts, torch.Tensor):
            ts = torch.as_tensor(ts, dtype=torch.long, device=dev)
        mask = torch.zeros(n_nodes, T + 1, dtype=torch.bool, device=dev)
        ts = ts.clamp_(0, T)
        if ts.numel() > 0:
            mask[:, ts.unique()] = True
        return mask
    mask = torch.zeros(n_nodes, T + 1, dtype=torch.bool, device=dev)
    mask[:, T] = True
    return mask

@torch.no_grad()
def test_fix_obs(data, y_pred):
    T = int(data.T.item())
    dev = data.y.device
    if y_pred.size(1) == T:
        y_pred = torch.cat([y_pred, data.y[:, -1:]], dim=1)
    elif y_pred.size(1) != T + 1:
        y_pred = y_pred[:, : T + 1]

    y_pred = y_pred.to(device=dev, dtype=data.y.dtype)
    obs_mask = _get_obs_mask(data)  # (nodes, T+1)
    return torch.where(obs_mask, data.y, y_pred)

@torch.no_grad()
def test_skm(skm_fn, data, y_pred, **kwargs):
    y_fixed = test_fix_obs(data, y_pred)
    obs_mask = _get_obs_mask(data)
    unobs = ~obs_mask
    y_true = torch2np(data.y[unobs])
    y_pred = torch2np(y_fixed[unobs])
    return float(skm_fn(y_true, y_pred, **kwargs))

@torch.no_grad()
def test_nrmse(data, y_pred):
    y_pred = test_fix_obs(data, y_pred)
    tI_pred = data_make_t(y_pred, SIR_STATES.I, dim = -1)
    mse = skm.mean_squared_error(torch2np(data.tI), torch2np(tI_pred))
    if hasattr(data, 'tR'):
        tR_pred = data_make_t(y_pred, SIR_STATES.R, dim = -1)
        mseR = skm.mean_squared_error(torch2np(data.tR), torch2np(tR_pred))
        mse = (mse + mseR) / 2.
    nrmse = np.sqrt(mse) / (data.T.item() + 1)
    return float(nrmse)


TEST_METRICS = {
    None: lambda data, y_pred: test_fix_obs(data, y_pred).tolist(),
    'acc':  fnt.partial(test_skm, skm.accuracy_score),
    'prc':  fnt.partial(test_skm, skm.precision_score, average='macro', zero_division=0),
    'rec':  fnt.partial(test_skm, skm.recall_score,    average='macro', zero_division=0),
    'f1':   fnt.partial(test_skm, skm.f1_score,        average='macro', zero_division=0),
    'nrmse': test_nrmse,
}

class Tester:
    def __init__(self, data_dir, device, model_fn):
        self.data_dir = data_dir
        self.device = device
        self.model_fn = model_fn
        self.res = dict()

    def test_once(self, dataset, seed=None):
        data = data_load(dataset, self.data_dir, self.device)
        if seed is not None:
            seed_all(seed)
        y_pred = self.model_fn(data)
        if dataset not in self.res:
            self.res[dataset] = dict()
        res = self.res[dataset]
        for metric, fn in TEST_METRICS.items():
            if metric not in res:
                res[metric] = list()
            self.res[dataset][metric].append(fn(data, y_pred))

    def test_dataset(self, dataset, seed=None, rep=5, verbose=True):
        for i in range(rep):
            if verbose:
                print(f'[{dataset} #{i}]', flush=True)
            self.test_once(dataset, seed=None if seed is None else (seed ^ i))
            if verbose:
                print(
                    f'[{dataset} #{i}]',
                    ', '.join([
                        f'{metric}={scores[-1]:.4f}'
                        for metric, scores in self.res[dataset].items()
                        if metric is not None
                    ]),
                    flush=True
                )

    def test(self, datasets=None, seed=None, **kwargs):
        if datasets is None:
            datasets = DATASETS.keys()
        for dataset in datasets:
            self.test_dataset(dataset=dataset, seed=seed, **kwargs)

    def print(self, brief=True):
        for dataset, metrics in self.res.items():
            for metric, scores in metrics.items():
                if metric is not None:
                    print(f'{dataset} {metric}:', end='')
                    if brief:
                        print(f' {np.mean(scores):.4f} ({np.std(scores):.4f})')
                    else:
                        for score in scores:
                            print(f' {score:.4f}', end='')
                        print('')

    def save(self, f, verbose=True):
        return torch.save(self.res, f)
