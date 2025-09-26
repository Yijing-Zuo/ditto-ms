# tests/test_ms_selfcheck.py
import torch
from types import SimpleNamespace
from inc.samp_ms import samp_ms
import collections

import numpy as np

class DummyQNet:
    def __init__(self, edge_index, T, device="cpu"):
        self.T = int(T)
        self.n_nodes = int(edge_index.max().item()) + 1
        self.device = torch.device(device)
        neigh = [[] for _ in range(self.n_nodes)]
        for u, v in edge_index.t().tolist():
            neigh[u].append(v); neigh[v].append(u)
        self.neighbs = [torch.tensor(n, device=self.device, dtype=torch.long) for n in neigh]
        deg = torch.tensor([len(n) for n in neigh], device=self.device).long()
        self.rem   = (deg + 1).view(-1, 1)  # (nodes,1)
        self.n_inf = torch.full((self.n_nodes, 1), self.n_nodes + 2, device=self.device)
        self.zero  = torch.tensor(0.0, device=self.device)

    def forward(self, yT, orig=True):
        T, n, dev = self.T, self.n_nodes, self.device
        zI = torch.zeros(T, n, 1, device=dev)
        zR = torch.zeros(T, n, 1, device=dev)
        return zI, zR, zI, zR

def test_selfcheck():
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4],
                               [1,0,2,1,3,2,4,3]], dtype=torch.long)
    T, n = 6, 5
    q_net = DummyQNet(edge_index, T)
    y3 = torch.tensor([0,1,1,0,0], dtype=torch.long)
    y5 = torch.tensor([0,2,1,1,0], dtype=torch.long)
    y_T = y5.clone()
    obs_times  = [3, 5, T]
    obs_states = torch.stack([y3, y5, y_T])

    zI0, zR0, zI, zR = q_net.forward(y_T, orig=True)

    Y, lq = samp_ms(q_net, y_T, zI, zR, n_samples=8, compute_lik=True,
                    obs_times=obs_times, obs_states=obs_states)

    assert Y.shape == (T, n, 8)
    assert lq.shape == (8,)
    assert (Y[1:] >= Y[:-1]).all().item()
    assert torch.equal(Y[3, :, 0], y3)
    assert torch.equal(Y[5, :, 0], y5)
