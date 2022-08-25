import math
import numpy as np
import torch
from typing import Optional


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    
    # N就是T*H*W//16//16, 
    # HW/P=H*W//16//16
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    # flatten后的ms：B x N x 1
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        mk = mk.transpose(1, 2)
        # mk B x N x CK
        # B x N x CK @ B x CK x HW/P = B x N x HW/P
        # 得到了N x HW/P的矩阵，矩阵的每一个元素ij代表着mk^2:Ck x N的第i列和qe:Ck x HW/P的第j列的内积
        # 即\Sigma_c kci^2*ecj，因为有ij所以通过矩阵乘法，就直接实现了sum
        a_sq = (mk.pow(2) @ qe)
        # B x N x CK @ B x CK x HW/P = B x N x HW/P
        # 矩阵的每一个元素ij代表着mk^2:Ck x N的第i列和qe:Ck x HW/P的第j列的内积，再乘以与j相关的系数qe
        # 即\Sigma_c kc_i * ecj * qcj
        two_ab = 2 * (mk @ (qk * qe))
        # sum完 B x 1 x HW/P
        # B x CK x HW/P * B x CK x HW/P = B x CK x HW/P sum - >B x HW/P 
        # 即\Sigma_c qcj^2 * ecj，这里只有jj，所以需要手动sum一下
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        # 合理，这里是算qk和所有T个mk之间similarity，所以sum一下，broadcast
        # a2-2ab+b2
        similarity = (-a_sq+two_ab-b_sq) # 这里broadcast了，所以是B x N x HW/P
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)
    # shrinkage 是仅与i相关的，所以算完之后再乘上去就行了
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW
    # similarity: B x N x [HW/P]
    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        # maxes: [B x 1 x HW//P]
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        # x_exp: [B x THW//P x HW//P]
        x_exp = torch.exp(similarity - maxes)
        # sum: [B x 1 x HW//P]
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        # x_exp: [B x THW//P x HW//P]
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape # 这里的H和W是除以16之后的

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity) # 纯粹的矩阵乘法: B x CV x THW @ B x THW x HW = B x CV x HW
    mem = mem.view(B, CV, H, W)

    return mem
