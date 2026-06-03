# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import argparse
import transformers
import os
from datetime import datetime
import logging
from termcolor import colored
import pprint
import shutil


def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["arc_challenge",
                 "arc_easy",
                 "boolq",
                 "hellaswag",
                 "lambada_openai",
                 "openbookqa",
                 "piqa",
                 "social_iqa",
                 "winogrande", ])

    parser.add_argument(
        "--zero_shot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eval zero shot.",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )
    parser.add_argument(
        "--input_model", type=str,default="meta-llama/Meta-Llama-3-8B", help="model path"
    )

    parser.add_argument(
        "--cache", type=str, default=None, help="auto log_dir+model_name"
    )

    parser.add_argument(
        "--log_dir", type=str, default="./", help="log path"
    )

    parser.add_argument(
        "--exp_name", type=str, default="rotated_matrix",
    )

    parser.add_argument(
        "--rotated_matrix_path", type=str, default=None, help="rotation rotated matrix"
    )


    parser.add_argument(
        "--clip_parameter", type=str, default=None, help="clip_parameter"
    )

    parser.add_argument(
        "--lac_path", type=str, default=None, help="lac"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=32, help="num_hidden_layers"
    )

    parser.add_argument('--device', type=int, default=0, help='GPU')


    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ and optimize Q",
    )
    parser.add_argument('--model_max_length', type=int, default=2048, help='Max sequence length for model (default:2048)')

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate the moodel.",
    )

    # optimize rotation matrix
    parser.add_argument('--block_diag', type=int, default=2, help='block diag into smaller multiples, no V and M')
    parser.add_argument('--aug_token', type=int, default=30, help='token augmentation')
    parser.add_argument(
        "--aug_MHA",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="aug_MHA",
    )
    parser.add_argument('--aug_start', type=int, default=2, help='start aug token from aug_start layer')
    parser.add_argument(
        "--rotation_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rotated activation zero-mean",
    )
    parser.add_argument('--lrs', nargs='+', type=int, default=[2, 2, 2], help='lr list for stage1,2,3')
    parser.add_argument('--lr_scheduler', type=float, default=0.01, help='lr scheduler')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=int, default=2, help='temperature for softmax')
    parser.add_argument('--dtype', type=str, default='float32', help='''data dtype''')


    # Learn activation clip Arguments
    parser.add_argument(
        "--save_Safetensors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save_Safetensors",
    )

    parser.add_argument(
        "--lac",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn activation clip",
    )

    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="bf16 of float16",
    )

    parser.add_argument('--cali_epochs', type=int, default=5)
    parser.add_argument('--cali_bsz', type=int, default=4)
    parser.add_argument('--cali_lr', type=float, default=0.02)
    parser.add_argument('--eta_m', type=int, default=10)

    parser.add_argument(
        "--deactive_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="deactive_amp",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="deactive_amp",
    )

    parser.add_argument(
        "--residual_bits", type=int, default=16, help="residual_bits"
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=0.9,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )

    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )

    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )

    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=0.9,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )

    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=0.9,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )

    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )

    args, unknown = parser.parse_known_args()

    assert args.k_pre_rope is False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown


def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def process_args_ptq():

    args, unknown_args = parser_gen()
    args.device = f"cuda:{args.device}"
    args.model_name = args.input_model.split("/")[-1]
    args.bsz = args.batch_size
    log = f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    args.cache = os.path.join(args.log_dir, args.model_name)
    args.exp_dir = os.path.join(args.log_dir, args.model_name,f"W{args.w_bits}A{args.a_bits}KV{args.k_bits}",args.exp_name, log)
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = create_logger(args.exp_dir)
    #copy_py_files_to_log_dir('./', args.exp_dir + "/code/")
    return args, logger


def copy_py_files_to_log_dir(source_dir, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".py"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(log_dir, os.path.relpath(source_file, source_dir))
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(source_file, target_file)
