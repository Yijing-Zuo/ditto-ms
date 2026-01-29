# ! pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
#! pip install --no-index torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
# ! pip install torch-geometric==2.0.4
# ! pip install ndlib==5.1.1


# ! pip install einops==0.6.0
# ! pip install test_tube==0.7.5


try:
    from tsl.nn.base import StaticGraphEmbedding
except Exception:
    import torch
    from torch import nn

    class StaticGraphEmbedding(nn.Module):
        def __init__(self, n_nodes, out_channels):
            super().__init__()
            self.emb = nn.Embedding(n_nodes, out_channels)

        def forward(self, token_index=None):
            if token_index is None:
                token_index = torch.arange(self.emb.num_embeddings, device=self.emb.weight.device)
            return self.emb(token_index)
from tsl.nn.layers import PositionalEncoding
from tsl.nn.layers.norm import LayerNorm
from tsl.nn.blocks.encoders import MLP
from tsl.nn.functional import sparse_softmax
from tsl.engines import Imputer, Predictor
from tsl.ops.connectivity import weighted_degree
#from tsl.data import Batch, SpatioTemporalDataModule, ImputationDataset

#SPINModel
'''https://github.com/Graph-Machine-Learning-Group/spin/blob/main/spin/layers/postional_encoding.py'''
from typing import Optional

from torch import nn

class PositionalEncoder(nn.Module):

    def __init__(self, in_channels, out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(PositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels,
                       n_layers=n_layers, activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self, x, node_emb=None, node_index=None):
        if node_emb is None:
            node_emb = self.node_emb(token_index=node_index)
        # x: [b s c], node_emb: [n c] -> [b s n c]
        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        #print('u:', tuple(x.shape), 'node_emb:', tuple(node_emb.shape))#####
        out = self.mlp(x)
        out = self.positional(out)
        return out

'''https://github.com/Graph-Machine-Learning-Group/spin/blob/main/spin/layers/additive_attention.py'''
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter
from torch_scatter.utils import broadcast


class AdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 dim: int = -2,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=dim, **kwargs)

        self.output_size = output_size
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size

        self.msg_size = msg_size or self.output_size
        self.msg_layers = msg_layers

        assert reweight in ['softmax', 'l1', None]
        self.reweight = reweight

        self.root_weight = root_weight
        self.dropout = dropout

        # key bias is discarded in softmax
        self.lin_src = Linear(self.src_size, self.output_size,
                              weight_initializer='glorot',
                              bias_initializer='zeros')
        self.lin_tgt = Linear(self.tgt_size, self.output_size,
                              weight_initializer='glorot', bias=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias=False)
        else:
            self.register_parameter('lin_skip', None)

        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            MLP(self.output_size, self.msg_size, self.output_size,
                n_layers=self.msg_layers, dropout=self.dropout,
                activation='prelu')
        )

        if self.reweight == 'softmax':
            self.msg_gate = nn.Linear(self.output_size, 1, bias=False)
        else:
            self.msg_gate = nn.Sequential(nn.Linear(self.output_size, 1),
                                          nn.Sigmoid())

        if norm:
            self.norm = LayerNorm(self.output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self, x: PairTensor, edge_index: Adj, mask: OptTensor = None):
        # if query/key not provided, defaults to x (e.g., for self-attention)
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        N_src, N_tgt = x_src.size(self.node_dim), x_tgt.size(self.node_dim)

        msg_src = self.lin_src(x_src)
        msg_tgt = self.lin_tgt(x_tgt)

        msg = (msg_src, msg_tgt)

        # propagate_type: (msg: PairTensor, mask: OptTensor)
        out = self.propagate(edge_index, msg=msg, mask=mask,
                             size=(N_src, N_tgt))

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def normalize_weights(self, weights, index, num_nodes, mask=None):
        # mask weights
        if mask is not None:
            fill_value = float("-inf") if self.reweight == 'softmax' else 0.
            weights = weights.masked_fill(torch.logical_not(mask), fill_value)
        # eventually reweight
        if self.reweight == 'l1':
            expanded_index = broadcast(index, weights, self.node_dim)
            weights_sum = scatter(weights, expanded_index, self.node_dim,
                                  dim_size=num_nodes, reduce='sum')
            weights_sum = weights_sum.index_select(self.node_dim, index)
            weights = weights / (weights_sum + 1e-5)
        elif self.reweight == 'softmax':
            weights = sparse_softmax(weights, index, num_nodes=num_nodes,
                                     dim=self.node_dim)
        return weights

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i,
                mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.output_size}, '
                f'dim={self.node_dim}, '
                f'root_weight={self.root_weight})')


class TemporalAdditiveAttention(AdditiveAttention):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('dim', 1)
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         msg_size=msg_size,
                         msg_layers=msg_layers,
                         root_weight=root_weight,
                         reweight=reweight,
                         dropout=dropout,
                         norm=norm,
                         **kwargs)

    def forward(self, x: PairTensor, mask: OptTensor = None,
                temporal_mask: OptTensor = None,
                causal_lag: Optional[int] = None):
        # x: [b s * c]    query: [b l * c]    key: [b s * c]
        # mask: [b s * c]    temporal_mask: [l s]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        l, s = x_tgt.size(self.node_dim), x_src.size(self.node_dim)
        i = torch.arange(l, dtype=torch.long, device=x_src.device)
        j = torch.arange(s, dtype=torch.long, device=x_src.device)

        # compute temporal index, from j to i
        if temporal_mask is None and isinstance(causal_lag, int):
            temporal_mask = tuple(torch.tril_indices(l, l, offset=-causal_lag,
                                                     device=x_src.device))
        if temporal_mask is not None:
            assert temporal_mask.size() == (l, s)
            i, j = torch.meshgrid(i, j)
            edge_index = torch.stack((j[temporal_mask], i[temporal_mask]))
        else:
            edge_index = torch.cartesian_prod(j, i).T

        return super(TemporalAdditiveAttention, self).forward(x, edge_index,
                                                              mask=mask)

'''https://github.com/Graph-Machine-Learning-Group/spin/blob/main/spin/layers/temporal_graph_additive_attention.py'''
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, OptPairTensor

class TemporalGraphAdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 temporal_self_attention: bool = True,
                 mask_temporal: bool = True,
                 mask_spatial: bool = True,
                 norm: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TemporalGraphAdditiveAttention, self).__init__(node_dim=-2,
                                                             **kwargs)

        # store dimensions
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size
        self.output_size = output_size
        self.msg_size = msg_size or self.output_size

        self.mask_temporal = mask_temporal
        self.mask_spatial = mask_spatial

        self.root_weight = root_weight
        self.dropout = dropout

        if temporal_self_attention:
            self.self_attention = TemporalAdditiveAttention(
                input_size=input_size,
                output_size=output_size,
                msg_size=msg_size,
                msg_layers=msg_layers,
                reweight=reweight,
                dropout=dropout,
                root_weight=False,
                norm=False
            )
        else:
            self.register_parameter('self_attention', None)

        self.cross_attention = TemporalAdditiveAttention(input_size=input_size,
                                                         output_size=output_size,
                                                         msg_size=msg_size,
                                                         msg_layers=msg_layers,
                                                         reweight=reweight,
                                                         dropout=dropout,
                                                         root_weight=False,
                                                         norm=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias_initializer='zeros')
        else:
            self.register_parameter('lin_skip', None)

        if norm:
            self.norm = LayerNorm(output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        if self.self_attention is not None:
            self.self_attention.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, x: OptPairTensor,
                edge_index: Adj, edge_weight: OptTensor = None,
                mask: OptTensor = None):
        # inputs: [batch, steps, nodes, channels]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        n_src, n_tgt = x_src.size(-2), x_tgt.size(-2)

        # propagate query, key and value
        #print('src:', x_src.shape, 'tgt:', x_tgt.shape, 'ei:', edge_index.shape, 'mask:', mask.shape, f'mask_spatial={self.mask_spatial}')
        out = self.propagate(x=(x_src, x_tgt),
                             edge_index=edge_index, edge_weight=edge_weight,
                             mask=mask if self.mask_spatial else None,
                             size=(n_src, n_tgt))

        if self.self_attention is not None:
            s, l = x_src.size(1), x_tgt.size(1)
            if s == l:
                attn_mask = ~torch.eye(l, l, dtype=torch.bool,
                                       device=x_tgt.device)
            else:
                attn_mask = None
            temp = self.self_attention(x=(x_src, x_tgt),
                                       mask=mask if self.mask_temporal else None,
                                       temporal_mask=attn_mask)
            out = out + temp

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_weight: OptTensor, mask_j: OptTensor) -> Tensor:
        # [batch, steps, edges, channels]

        out = self.cross_attention((x_j, x_i), mask=mask_j)
        #print('out:', out.shape)

        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        return out

'''https://github.com/Graph-Machine-Learning-Group/spin/blob/main/spin/models/spin.py'''
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import LayerNorm
from torch_geometric.typing import OptTensor

class SPINModel(nn.Module):

    def __init__(self, input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 u_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 reweight: Optional[str] = 'softmax',
                 n_layers: int = 4,
                 eta: int = 3,
                 message_layers: int = 1):
        super(SPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=hidden_size,
                                       n_layers=2,
                                       n_nodes=n_nodes)

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = TemporalGraphAdditiveAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                norm=True,
                root_weight=True,
                dropout=0.0
            )
            readout = MLP(hidden_size, hidden_size, output_size,
                          n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self, x: Tensor, u: Tensor, mask: Tensor,
                edge_index: Tensor, edge_weight: OptTensor = None,
                node_index: OptTensor = None, target_nodes: OptTensor = None):
        if target_nodes is None:
            target_nodes = slice(None)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            #print(f'l={l}', 'h:', tuple(h.shape), 'x:', tuple(x.shape), 'mask:', tuple(mask.shape), 'ei:', edge_index)
            h = self.encoder[l](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations


import argparse
import torch

from inc.diffus import *

try:
    from inc.test_ms import Tester  # type: ignore
except Exception:
    from inc.test import Tester  # type: ignore


def get_args():
    parser = argparse.ArgumentParser()

    # -------------------- common experiment args --------------------
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed')
    parser.add_argument('--data_dir', type=str, default='input', help='dataset folder')
    parser.add_argument('--output', type=str, default='output/spin.pt', help='output file name')
    parser.add_argument(
        '--device', type=str,
        default='cuda'  if torch.cuda.is_available() else 'cpu',
        help='torch device, e.g., "cuda", "cuda:0", or "cpu"'
    )
    parser.add_argument(
        '--obs_time', '--snapshot', dest='obs_time',
        type=str, default='',
        help='extra observed snapshot times, comma-separated, e.g., "5,7,9". '
             'Final time T will always be added automatically.'
    )

    # -------------------- diffusion parameter estimation --------------------
    parser.add_argument('--b_pI0', type=float, default=1e-3,
                        help='initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type=float, default=1e-3,
                        help='initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type=int, default=500,
                        help='optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type=float, default=3e-3,
                        help='learning rate in diffusion parameter estimation')

    # -------------------- SPIN model hyperparams --------------------
    parser.add_argument('--u_size', type=int, default=1, help='u feature size (kept as 1)')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--reweight', type=str, default='softmax',
                        choices=['softmax', 'l1', 'none'],
                        help='attention reweighting: softmax | l1 | none')
    parser.add_argument('--n_layers', type=int, default=4, help='number of SPIN layers')
    parser.add_argument('--eta', type=int, default=3, help='layers before enabling spatial mask-off')
    parser.add_argument('--message_layers', type=int, default=1, help='message MLP depth inside attention')

    parser.add_argument(
        '--no_temporal_self_attention', action='store_false',
        dest='temporal_self_attention',
        help='disable temporal self-attention (default: enabled)'
    )
    parser.set_defaults(temporal_self_attention=True)

    # -------------------- training hyperparams --------------------
    parser.add_argument('--lr', type=float, default=8e-4, help='Adam lr')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='Adam weight_decay')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='synthetic batch size per epoch')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    if args.reweight == 'none':
        args.reweight = None

    return args


def parse_obs_time(obs_time_str: str, T: int):
    """Parse comma-separated observed snapshot times and always include final time T."""
    times = []
    if obs_time_str:
        for part in str(obs_time_str).split(','):
            part = part.strip()
            if part == '':
                continue
            times.append(int(part))
    # keep within [0, T]
    times = [t for t in times if 0 <= t <= T]
    if T not in times:
        times.append(T)
    return sorted(set(times))


def build_obs_mask(n_samples: int, n_nodes: int, T: int, obs_time, device):
    """Return float mask with shape (samples, T+1, nodes, 1), 1=observed, 0=missing."""
    mask = torch.zeros((n_samples, T + 1, n_nodes, 1), dtype=torch.float32, device=device)
    if len(obs_time) > 0:
        mask[:, obs_time, :, :] = 1.0
    return mask


def spin_prep(y, edge_index, obs_time, args):
    """
    y: (samples, nodes, T+1)
    return:
      x:    (samples, T+1, nodes, 1)
      u:    (samples, T+1, u_size)
      mask: (samples, T+1, nodes, 1)
      ei:   edge_index
    """
    n_samples, n_nodes, Tp1 = y.size()
    T = Tp1 - 1

    x = y.float().transpose(1, 2).unsqueeze(dim=3)  # (samples, T+1, nodes, 1)
    u = torch.ones(n_samples, T + 1, args.u_size, dtype=torch.float32, device=args.device)
    mask = build_obs_mask(n_samples, n_nodes, T, obs_time, device=args.device)

    # NOTE: SPIN implementation here uses a single static edge_index shared across all steps.
    ei = edge_index
    return x, u, mask, ei


def spin_run(data, args):
    """Train SPIN on synthetic histories and infer missing diffusion history."""
    T = int(data.T.item())
    obs_time = parse_obs_time(args.obs_time, T)
    data.obs_ts = obs_time
    bpar = b_estim(data, args, obs_time=obs_time)

    n_nodes = int(data.num_nodes)
    n_out = int(data.y[:, -1].max().item() + 1)

    # -------------------- train --------------------
    model = SPINModel(
        input_size=1,
        u_size=args.u_size,
        n_nodes=n_nodes,
        hidden_size=args.hidden_size,
        output_size=1,
        temporal_self_attention=args.temporal_self_attention,
        reweight=args.reweight,
        n_layers=args.n_layers,
        eta=args.eta,
        message_layers=args.message_layers,
    ).to(args.device)

    I0 = int((data.y[:, 0] == SIR_STATES.I).long().sum().item())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    model.train()
    pbar = trange(1, args.epochs + 1)
    for epoch in pbar:
        opt.zero_grad()

        # synthetic history
        Y_true = diffus_gen(
            T=T,
            n_nodes=n_nodes,
            edge_index=data.edge_index,
            I0=I0,
            n_samples=args.batch_size,
            pI=bpar.pI,
            pR=bpar.pR,
        )  # (T+1, nodes, samples)

        x, u, mask, ei = spin_prep(Y_true.transpose(0, 2), data.edge_index, obs_time, args)
        z = model(x=x, u=u, mask=mask, edge_index=ei)[0]  # (samples, T+1, nodes, 1)
        y_true = Y_true.permute(2, 0, 1).unsqueeze(-1)  # (samples, T+1, nodes, 1)
        unobs = ~mask.bool()
        loss = (z - y_true).abs()
        loss = loss[unobs].mean() if unobs.any() else loss.mean()

        pbar.set_description(f'[epoch={epoch}] loss={loss.item():.4f}')
        loss.backward()
        opt.step()

    # -------------------- infer --------------------
    with torch.no_grad():
        model.eval()

        x, u, mask, ei = spin_prep(data.y.unsqueeze(0).clone(), data.edge_index, obs_time, args)
        z = model(x=x, u=u, mask=mask, edge_index=ei)[0]  # (1, T+1, nodes, 1)

        y_pred = z[0, :, :, 0].clamp(0, n_out - 1).transpose(0, 1).round().long()  # (nodes, T+1)
        for t in obs_time:
            y_pred[:, t] = data.y[:, t]

        return y_pred.detach().clone()


def main():
    args = get_args()

    def _run(data):
        return spin_run(data, args)

    tester = Tester(args.data_dir, args.device, _run)
    tester.test([args.dataset], seed=args.seed, rep=1)
    tester.save(args.output)


if __name__ == '__main__':
    main()
