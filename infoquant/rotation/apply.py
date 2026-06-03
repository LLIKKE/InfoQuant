# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import matplotlib.pyplot as plt
import torch
import tqdm
import torch.nn as nn
from infoquant.utils import monkeypatch
from infoquant.quantization import quantizers as quant_utils
from infoquant.utils import runtime as utils
from infoquant.rotation.hadamard import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
    get_hadK,
    hadamard_matrix
)
from infoquant.utils.runtime import HadamardTransform
import numpy as np


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor, device) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1_ = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1_).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, R1, device) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.v_proj, layer.self_attn.q_proj, layer.self_attn.k_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        R1_ = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1_.T).to(device="cpu", dtype=dtype)



def rotate_attention_output(layer, R1, device) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1_ = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(R1_.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(R1_.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1_ = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1_.T).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    R1_ = R1.to(device=device, dtype=torch.float64)
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(R1_.T, W_).to(device="cpu", dtype=torch.float16)

    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output

    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(R1_.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor, device) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1_ = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1_.T).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, num_key_value_groups, Q2, device):
    Wv_proj = layer.self_attn.v_proj
    Wo_proj = layer.self_attn.o_proj

    dtype = Wv_proj.weight.data.dtype

    W_v = Wv_proj.weight.data.to(device=device, dtype=torch.float64)
    W_o = Wo_proj.weight.data.to(device=device, dtype=torch.float64)

    R2 = torch.block_diag(*Q2)

    R2_inv = []
    for Q in Q2:
        R2_inv.extend([Q.T,]*num_key_value_groups)
    R2_inv = torch.block_diag(*R2_inv)

    R2_ = R2.to(device=device, dtype=torch.float64)
    R2_inv_ = R2_inv.to(device=device, dtype=torch.float64)

    Wv_proj.weight.data = torch.matmul(R2_.T, W_v).to(device="cpu", dtype=dtype)
    Wo_proj.weight.data = torch.matmul(W_o, R2_inv_.T).to(device="cpu", dtype=dtype)


def block_diag(Q, device, dtype=torch.float64):
    Q['Q1'] = Q['Q1'].to(device=device)
    Q['Q2'] = Q['Q2'].to(device=device)
    Q['Q4'] = Q['Q4'].to(device=device)
    if isinstance(Q['Q1'], nn.ParameterList):
        Q1 = torch.block_diag(*Q['Q1'])
    else:
        Q1 = Q['Q1']
    if isinstance(Q['Q2'], nn.ParameterList):
        Q2 = torch.block_diag(*Q['Q2'])
    else:
        Q2 = Q['Q2']
    if isinstance(Q['Q4'], nn.ParameterList):

        Q4 = torch.block_diag(*Q['Q4'])
    else:
        Q4 = Q['Q4']
    Q['Q1']=Q['Q1'].cpu()
    Q['Q2'] = Q['Q2'].cpu()
    Q['Q4'] = Q['Q4'].cpu()
    torch.cuda.empty_cache()
    return Q1.to(device).to(dtype), Q2.to(device).to(dtype), Q4.to(device).to(dtype)


def rotate_model(model, device,args,logger=None):
    if logger is not None:
        logger.info("Rotate model...")
        
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    rotated_matrix_path = args.rotated_matrix_path if args.rotated_matrix_path is not None else args.exp_dir
    Q = torch.load(f"{rotated_matrix_path}/Q/" + f'{args.model_name}_layer_{0}_Q.pth')
    hadK1 = hadamard_matrix(hidden_size // len(Q['Q1']), device).to(torch.float64)
    hadK = torch.block_diag(*([hadK1,]*len(Q['Q1'])))
    rotate_head(model, hadK.T, device)  # R1.T W^T->WR1
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]

    def residual_linear(A, B, bs, inv=True):
        AB = nn.ModuleList([nn.Linear(hidden_size // bs, hidden_size // bs, bias=False) for _ in range(bs)])
        for idx, layer in enumerate(AB):
            if inv:
                layer.weight.data = (A[idx].T @ B[idx]).T.to(torch.float16)
            else:
                layer.weight.data = (A[idx] @ B[idx]).T.to(torch.float16)
        return AB

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):

        Q = torch.load(f"{rotated_matrix_path}/Q/"+ f'{args.model_name}_layer_{idx}_Q.pth')
        Q1, _, Q4 = block_diag(Q, device)
        Q1_inv, _, Q4_inv = Q1.T, _, Q4.T
        if idx == 0:
            rotate_embeddings(model, Q1, device)

        layers[idx].R1R4 = residual_linear(Q["Q1"], Q["Q4"], len(Q['Q1']))
        rotate_attention_inputs(layers[idx], Q1_inv, device)
        rotate_attention_output(layers[idx], Q4, device)
        rotate_mlp_input(layers[idx], Q4_inv, device)

        if idx != len(layers) - 1:
            Q_next = torch.load(f"{rotated_matrix_path}/Q/" + f'{args.model_name}_layer_{idx+1}_Q.pth')
            Q_n1, _, _ = block_diag(Q_next, device)
            rotate_mlp_output(layers[idx], Q_n1, device)
            layers[idx].R4R6 = residual_linear(Q["Q4"], Q_next["Q1"], len(Q['Q1']))
        else:
            rotate_mlp_output(layers[idx], hadK, device)
            layers[idx].R4R6 = residual_linear(Q["Q4"].to(torch.float64).to(device), [hadK1,]*len(Q['Q1']), len(Q['Q1']))
        rotate_ov_proj(layers[idx], num_key_value_groups,
                       Q2=Q['Q2'], device=device)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
        module,
        function_name,
        *args,
        **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
