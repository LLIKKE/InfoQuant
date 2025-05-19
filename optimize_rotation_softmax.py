import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from Activation_loader import CustomActivationDataset
from utils.hadamard_utils import random_hadamard_matrix
from utils.function_utils import optims
from datetime import datetime
import time
import os

dtype_mapping = {
    'float32': torch.float32,
    'float64': torch.float64,
    'int32': torch.int32,
    'int64': torch.int64,
}

def BDMM(hidden, Qs):
    B, N, D = hidden.shape
    num_blocks = len(Qs)
    d = D // num_blocks
    hidden = hidden.view(B, N, num_blocks, d)
    hidden = torch.stack(
        [hidden[:, :, i, :] @ Qs[i] for i in range(num_blocks)], dim=2
    )
    hidden = hidden.view(B, N, D)
    return hidden

def aug_massive_token(mas_idx, aug):
    mas_idx = mas_idx * aug
    mas_idx[mas_idx == 0] = 1
    return mas_idx


def Cost(rotation_u, T, dim, X_aug):
    XQ = torch.abs(rotation_u)
    score = F.softmax(XQ / T, dim=dim)
    if X_aug is None:
        return torch.norm((score * XQ))
    if len(rotation_u.shape) == 4:
        X_aug = X_aug.reshape(X_aug.shape[0], X_aug.shape[1], 1, -1)
    return torch.norm((score * XQ) * X_aug)


class Op_X(nn.Module):
    def __init__(self, temperature, block_diag, rotation_mean):
        super(Op_X, self).__init__()

        self.temperature = temperature
        self.block_diag = block_diag
        self.rotation_mean = rotation_mean

    def forward(self, X, X_aug, Q1):
        h, w = X.shape[-2:]
        rotation_X = BDMM(X, Q1)

        if self.block_diag > 0:
            rotation_X = rotation_X.view(-1, h, w // self.block_diag, self.block_diag)

        if self.rotation_mean:
            xu = torch.mean(rotation_X, dim=-1, keepdim=True)
            cost = Cost((rotation_X - xu).flatten(start_dim=2), T=self.temperature, dim=-1, X_aug=X_aug)
        else:
            cost = Cost((rotation_X).flatten(start_dim=2), T=self.temperature, dim=-1, X_aug=X_aug)

        return cost


class Op_stage1(nn.Module):
    def __init__(self, hidden_size, temperature=1., args=None):
        super(Op_stage1, self).__init__()

        self.aug = args.aug_token
        dtype = dtype_mapping[args.dtype]
        if args.block_diag < 0:
            b_diag = 1
            dim = hidden_size
        else:
            b_diag = args.block_diag
            dim = hidden_size // args.block_diag
        self.Q1 = nn.ParameterList([
            torch.nn.Parameter(random_hadamard_matrix(dim, args.device, args.seed).to(dtype)) for i in range(b_diag)
        ])

        self.Op_X = Op_X(temperature, hidden_size // args.block_diag, args.rotation_mean)

    def forward(self, X, massive_idx):

        X_aug = None
        if massive_idx is not None:
            X_aug = aug_massive_token(massive_idx, self.aug)

        Xcost = self.Op_X(X, X_aug, self.Q1)

        return Xcost

class Op_atten(nn.Module):
    def __init__(self, temperature, head_dim, num_kv_heads, num_key_value_groups):
        super(Op_atten, self).__init__()
        self.temperature = temperature
        self.head_dim = head_dim
        self.num_key_value_groups = num_key_value_groups
        self.num_kv_heads = num_kv_heads

    def forward(self, M, X_aug, Q2):
        bs, slen, w = M.shape
        if self.num_key_value_groups>1:
            A = []
            M_chunks = torch.chunk(M, self.num_kv_heads, dim=-1)
            for chunk in M_chunks:
                A_chunks = torch.chunk(chunk, self.num_key_value_groups, dim=-1)
                A.append(torch.cat(A_chunks, dim=-2))
            M = torch.cat(A, dim=-1)

        rotation_M = BDMM(M, Q2)

        if self.num_key_value_groups>1:
            A = []
            M_chunks = torch.chunk(rotation_M, self.num_kv_heads, dim=-1)
            for chunk in M_chunks:
                A_chunks = torch.chunk(chunk, self.num_key_value_groups, dim=-2)
                A.append(torch.cat(A_chunks, dim=-1))
            rotation_M = torch.cat(A, dim=-1)

        if self.head_dim > 0:
            rotation_M = rotation_M.reshape(-1, slen, w // self.head_dim, self.head_dim)
        else:
            rotation_M = rotation_M.reshape(-1, slen, w)

        xu = torch.mean(rotation_M, dim=-1, keepdim=True)

        return Cost(rotation_M - xu, T=self.temperature, dim=-1, X_aug=X_aug)


class Op_X_v(nn.Module):
    def __init__(self, temperature, head_dim):
        super(Op_X_v, self).__init__()
        self.temperature = temperature
        self.head_dim = head_dim

    def forward(self, Xv, X_aug, Q2):
        _, h, w = Xv.shape

        rotation_Xv = BDMM(Xv, Q2)

        if self.head_dim > 0:
            rotation_Xv = rotation_Xv.reshape(-1, h, w // self.head_dim, self.head_dim)
        else:
            rotation_Xv = rotation_Xv.reshape(-1, h, w)

        xu = torch.mean(rotation_Xv, dim=-1, keepdim=True)

        return Cost(rotation_Xv - xu, T=self.temperature, dim=-1, X_aug=X_aug)


class Op_stage2(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_kv_heads, temperature1=1., temperature2=1., args=None):
        super(Op_stage2, self).__init__()
        dtype = dtype_mapping[args.dtype]
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_attention_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.Q2 = nn.ParameterList([
            torch.nn.Parameter(random_hadamard_matrix(self.head_dim, args.device, args.seed).to(dtype)) for i in
            range(self.num_kv_heads)
        ])

        self.aug_MHA = args.aug_MHA
        self.aug = args.aug_token

        self.Op_X_v = Op_X_v(temperature2, self.head_dim)

        self.Op_atten = Op_atten(temperature1, self.head_dim, self.num_kv_heads, self.num_key_value_groups)


    def forward(self, Xv, M, Vmassive_idx, Mmassive_idx):

        V_aug = None
        M_aug = None
        if self.aug_MHA:
            if Vmassive_idx is not None:
                V_aug = aug_massive_token(Vmassive_idx, self.aug)
            if Mmassive_idx is not None:
                M_aug = aug_massive_token(Mmassive_idx, self.aug)
        Mcost = self.Op_atten(M, M_aug, self.Q2)

        Xvcost = self.Op_X_v(Xv, V_aug, self.Q2)

        loss1 = Mcost.detach()
        loss2 = Xvcost.detach()

        return Mcost + Xvcost * (loss1 / loss2)


class Op_X_p(nn.Module):
    def __init__(self, temperature, block_diag, rotation_mean):
        super(Op_X_p, self).__init__()
        self.temperature = temperature
        self.block_diag = block_diag
        self.rotation_mean = rotation_mean

    def forward(self, Xp, X_aug, Q4):
        h, w = Xp.shape[-2:]
        rotation_Xp = BDMM(Xp, Q4)

        if self.block_diag > 0:
            rotation_Xp = rotation_Xp.view(-1, h, w // self.block_diag, self.block_diag)

        if self.rotation_mean:
            xu = torch.mean(rotation_Xp, dim=-1, keepdim=True)
            cost = Cost((rotation_Xp - xu).flatten(start_dim=2), T=self.temperature, dim=-1, X_aug=X_aug)
        else:
            cost = Cost((rotation_Xp).flatten(start_dim=2), T=self.temperature, dim=-1, X_aug=X_aug)
        return cost


class Op_stage3(nn.Module):
    def __init__(self, hidden_size, temperature3=1., args=None):
        super(Op_stage3, self).__init__()
        dtype = dtype_mapping[args.dtype]

        self.aug = args.aug_token
        if args.block_diag < 0:
            b_diag = 1
            dim = hidden_size
        else:
            b_diag = args.block_diag
            dim = hidden_size // args.block_diag
        self.Q4 = nn.ParameterList([
            torch.nn.Parameter(random_hadamard_matrix(dim, args.device, args.seed).to(dtype)) for i in range(b_diag)
        ])

        self.Op_X_p = Op_X_p(temperature3, hidden_size // args.block_diag, args.rotation_mean)
    def forward(self, Xp, massive_idx):

        X_aug = None
        if massive_idx is not None:
            X_aug = aug_massive_token(massive_idx, self.aug)
        Xpcost = self.Op_X_p(Xp, X_aug, self.Q4)
        return Xpcost


def optimize_rotation_matrix(Activation, layer_idx, epochs, T, lrs, batch_size, config, args, logger=None):
    output = f"{args.exp_dir}/Q/"
    logger.info("get matrix layer {}".format(layer_idx))
    if os.path.exists(output + f'{args.model_name}_layer_{layer_idx}_Q.pth'):
        return None

    stage1 = Op_stage1(config.hidden_size, temperature=T, args=args).to(args.device)
    stage2 = Op_stage2(hidden_size=config.hidden_size,
                       num_attention_heads=config.num_attention_heads,
                       num_kv_heads=config.num_key_value_heads,
                       temperature1=T, temperature2=T,
                       args=args).to(args.device)
    stage3 = Op_stage3(config.hidden_size, temperature3=T, args=args).to(args.device)

    stages = [stage1, stage2, stage3]

    optim = optims(stages, epochs, lrs, args=args)

    Activation_set = CustomActivationDataset(Activation, layer_idx, args)
    dataloader = DataLoader(Activation_set, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in range(epochs):
        sloss1, sloss2, sloss3 = 0, 0, 0,
        start_tick = time.time()
        for batch_idx, (X, P, V, M, X_aug, P_aug, V_aug, M_aug) in enumerate(dataloader):
            X, P, V, M = X.to(args.device), P.to(args.device), V.to(args.device), M.to(args.device)
            X_aug, P_aug, V_aug, M_aug = X_aug.to(args.device), P_aug.to(args.device), V_aug.to(args.device), M_aug.to(
                args.device)

            optim.zero_grad()

            loss1 = stages[0](X, X_aug)
            loss2 = stages[1](V, M, V_aug, M_aug)
            loss3 = stages[2](P, P_aug)

            optim.backward([loss1, loss2, loss3])
            optim.step()

            sloss1 += loss1.detach()
            sloss2 += loss2.detach()
            sloss3 += loss3.detach()

        optim.lr_step()
        logger.info(
            f"layer {layer_idx} iter {epoch} time {time.time() - start_tick:.3f}s, loss1: {sloss1.item() / len(dataloader):.4f}, loss2: {sloss2.item() / len(dataloader):.4f}, loss3: {sloss3.item() / len(dataloader):.4f}")
    Q = {
        "Q1": stage1.Q1,
        "Q2": stage2.Q2,
        "Q4": stage3.Q4,
    }

    if not os.path.exists(output + f'{args.model_name}_layer_{layer_idx}_Q.pth'):
        os.makedirs(output, exist_ok=True)
        torch.save(Q, output + f'{args.model_name}_layer_{layer_idx}_Q.pth')

    return Q

