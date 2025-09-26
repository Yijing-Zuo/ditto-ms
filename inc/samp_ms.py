# -*- coding: utf-8 -*-
"""
每一步先处理 R->I，再处理 I->S；
I->S:
按 zI 降序遍历,当某个时刻 t 有观测时，对相应节点的本步“反向转移”进行计算，
以确保 Y[t] 与观测一致；若与 SIR 反向可达性或容量约束冲突，则将该样本标记为 invalid；
若 compute_lik=True，则返回在 Qθ 下的 log-likelihood：

R->I：对 (y_{t+1}==R) 的位置累加 lR(不变)；
I->S：只对 **最终 msk_opt=True** 的位置累加 lI（与原版一致）；
对于invalid 样本的 lik 置为 -inf

原版:Y = samp_ms(q_net, y_T, zI, zR, n_samples=64, compute_lik=False)
现有版本:Y, lq = samp_ms(
    q_net, y_T, zI, zR, n_samples=64, compute_lik=True,
    obs_times=[3, 7],
    obs_states=torch.stack([y3_obs, y7_obs]),   # (2, n_nodes) Long
    obs_masks=torch.stack([m3, m7])              # (2, n_nodes) Bool
)
"""

import torch
try:
    from .diffus import SIR_STATES
except Exception:
    from types import SimpleNamespace
    SIR_STATES = SimpleNamespace(S=0, I=1, R=2)
def torch_log(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return torch.log(torch.clamp(x, min=eps))

@torch.no_grad()
def samp_ms(
    q_net,
    y,
    zI,
    zR,
    n_samples,
    compute_lik=False,
    obs_times=None,#list[int] / 1D LongTensor，观测时刻（取值{0..T}；允许 T 表示 y_T）
    obs_states=None,#(K, nodes) Long，obs_times 对应的观测状态
    obs_masks=None#(K, nodes) Bool，部分观测掩码；None 表示全 True
):
    """
      返回:若compute_lik=False -> Y: (T, nodes, samples) Long
      若compute_lik=True  -> (Y, lik)，其中 lik: (samples,) Float
    """

    device  = q_net.device
    T       = int(q_net.T)
    n_nodes = int(q_net.n_nodes)
    obs_map = {}
    if obs_times is not None:
        if not torch.is_tensor(obs_times):
            obs_times = torch.as_tensor(obs_times, dtype=torch.long, device=device)
        else:
            obs_times = obs_times.to(device=device, dtype=torch.long)

        assert obs_states is not None#obs_states 必须与 obs_times 同时提供
        if not torch.is_tensor(obs_states):
            obs_states = torch.as_tensor(obs_states, dtype=torch.long, device=device)
        else:
            obs_states = obs_states.to(device=device, dtype=torch.long)
        assert obs_states.dim() == 2 and obs_states.size(1) == n_nodes#obs_states 必须为 (K, nodes)
        assert obs_states.size(0) == obs_times.numel()#obs_states 第一维必须等于 obs_times 的长度
        if obs_masks is None:
            obs_masks = torch.ones_like(obs_states, dtype=torch.bool, device=device)
        else:
            if not torch.is_tensor(obs_masks):
                obs_masks = torch.as_tensor(obs_masks, dtype=torch.bool, device=device)
            else:
                obs_masks = obs_masks.to(device=device, dtype=torch.bool)
            assert obs_masks.shape == obs_states.shape#obs_masks 形状必须与 obs_states 一致

        for k, t in enumerate(obs_times.tolist()):
            y_obs_t = obs_states[k]
            m_obs_t = obs_masks[k]
            if t == T:
                obs_map[T] = (y_obs_t, m_obs_t)
            else:
                assert 0 <= t < T#obs_times 的取值必须位于 [0, T]
                obs_map[t] = (y_obs_t, m_obs_t)
    zI, uid = zI.sort(dim=1, descending=True)
    uid = uid.squeeze(dim=2)
    qI = torch.sigmoid(zI)
    xI = SIR_STATES.I - qI.expand(-1, -1, n_samples).bernoulli().long()
    lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI))
    qR = torch.sigmoid(zR)
    xR = SIR_STATES.R - qR.expand(-1, -1, n_samples).bernoulli().long()
    lR = torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR))
    y = y.to(device=device, dtype=torch.long).unsqueeze(1).expand(-1, n_samples)# (nodes, samples)
    Y = torch.empty(T, n_nodes, n_samples, dtype=torch.long, device=device)# (T, nodes, samples)

    if compute_lik:
        lik = q_net.zero
    invalid = torch.zeros(n_samples, dtype=torch.bool, device=device)
    if T in obs_map:
        y_obs_T, m_obs_T = obs_map[T]
        bad_T = m_obs_T.unsqueeze(1) & (y != y_obs_T.unsqueeze(1))
        if bad_T.any():
            invalid |= bad_T.any(dim=0)
    for t in range(T - 1, -1, -1):
        y_obs_t = None
        m_obs_t = None
        if t in obs_map:
            y_obs_t, m_obs_t = obs_map[t]
            # 反向可达性：
            #y_{t+1}=S -> y_t 必为 S，否则 invalid
            #y_{t+1}=I -> y_t 不可为 R，否则 invalid
            #y_{t+1}=R -> y_t ∈ {R,I,S} 均可达（先 R->I，再可能 I->S）
            bad_from_S = (y == SIR_STATES.S) & m_obs_t.unsqueeze(1) & (y_obs_t != SIR_STATES.S).unsqueeze(1)
            bad_from_I = (y == SIR_STATES.I) & m_obs_t.unsqueeze(1) & (y_obs_t == SIR_STATES.R).unsqueeze(1)
            bad_any = bad_from_S | bad_from_I
            if bad_any.any():
                invalid |= bad_any.any(dim=0)
        mR = (y == SIR_STATES.R)
        if y_obs_t is not None:
            mR_obs = mR & m_obs_t.unsqueeze(1)
            qR_t = qR[t].expand(-1, n_samples)
            # 观测想要 y_t=R -> 强制“不发生 R->I”，保持 R
            wantR = mR_obs & (y_obs_t.eq(SIR_STATES.R).unsqueeze(1))
            if wantR.any():
                xR[t][wantR] = SIR_STATES.R
                lR[t][wantR] = torch_log(1. - qR_t[wantR])
            # 观测想要 y_t ∈ {I,S} -> 强制“发生 R->I”，先变 I
            wantI_or_S = mR_obs & (~y_obs_t.eq(SIR_STATES.R).unsqueeze(1))
            if wantI_or_S.any():
                xR[t][wantI_or_S] = SIR_STATES.I
                lR[t][wantI_or_S] = torch_log(qR_t[wantI_or_S])
                # 若目标是 S，则稍后 I->S 阶段还会再强制一次（最终变 S）
        y = torch.where(mR, xR[t], y)
        if compute_lik:
            lik = lik + torch.where(mR, lR[t], q_net.zero).sum(dim=0)
        msk = (y == SIR_STATES.I)# (nodes, samples)
        rem = torch.where(msk, q_net.rem, q_net.n_inf)# (nodes, samples)  广播 rem 初值
        for i, u in enumerate(uid[t]):# u: 节点 id（0..n_nodes-1）
            if msk[u].max():# 该节点在任一样本为 I 才需要处理
                vid = q_net.neighbs[u.item()]# (deg_u,) Long，邻居索引列表
                # 原版 I->S 可行性：opt = (rem[u] > 1) & (min_{v∈N(u)} rem[v] > 1)
                opt = (rem[u] > 1) & (rem[vid].min(dim=0).values > 1)  # (samples,)
                msk_opt = msk[u] & opt                                    # (samples,)
                # 若 t 时刻有观测，对该节点进行强制（仅当此节点被观测）
                if (y_obs_t is not None) and bool(m_obs_t[u.item()]):
                    target = int(y_obs_t[u.item()].item())
                    if target == SIR_STATES.S:
                        # 目标为 S -> 强制“发生 I->S”
                        need_I = ~msk[u]
                        if need_I.any():
                            invalid |= need_I
                        # 覆盖本节点排序坐标 (t, i) 的采样与对数概率
                        xI[t, i] = SIR_STATES.S
			qi = qI[t, i, 0]
                        # qI[t, i] 形状为 (1,)，扩展到 (samples,)
                        lI[t, i].fill_(float(torch_log(qi)))
                        # 强制发生时仍需 opt 通过；否则该样本 invalid
                        bad_no_opt = msk[u] & (~opt)
                        if bad_no_opt.any():
                            invalid |= bad_no_opt
                    elif target == SIR_STATES.I:
                        # 目标为 I -> 强制“不发生 I->S”
                        need_I = ~msk[u]
                        if need_I.any():
                            invalid |= need_I
                        xI[t, i] = SIR_STATES.I
			qi = qI[t, i, 0]
                        lI[t, i].fill_(float(torch_log(1. - qi)))
                        # 不发生时，无需 opt（与原版一致：opt=False 时不会进入 toss）
                    else:
                        # 目标为 R：本步（I->S 阶段）无法直接达到 R,不可达
                        invalid |= msk[u]
                #应用 I->S 到 y（仅在 msk_opt 的样本上生效）
                y[u] = torch.where(msk_opt, xI[t, i], y[u])
                trs = (y[u] != SIR_STATES.I)# 是否发生了 I->S（= 正向新感染）

                rem[u]   = torch.where(msk[u], torch.where(trs, rem[u] - 1, q_net.n_inf[u].expand_as(rem[u])), rem[u])
                rem[vid] = torch.where(msk[u].unsqueeze(0),
                                       torch.where(trs.unsqueeze(0), rem[vid] - 1, q_net.n_inf[vid],expand_as(rem[vid])),
                                       rem[vid])
                #最终 msk[u] 仅保留“进入 toss 且 opt 通过”的样本（与原版一致）
                msk[u] = msk_opt
        Y[t] = y
        if compute_lik:
            lik = lik + torch.where(msk[uid[t]], lI[t], q_net.zero).sum(dim=0)  # (samples,)

    Y = Y.detach().clone()
    if compute_lik:
        neg_inf = torch.full_like(lik if torch.is_tensor(lik) else torch.zeros(n_samples, device=device), float('-inf'))
        if torch.is_tensor(lik):
            lik = torch.where(invalid, neg_inf, lik).detach().clone()
        else:
            lik = torch.where(invalid, neg_inf, torch.zeros(n_samples, device=device)).detach().clone()
        return Y, lik
    else:
        return Y
