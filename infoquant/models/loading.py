# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import torch.nn as nn
import transformers

from infoquant.calibration import datasets as data_utils
from infoquant.models import fuse_norm as fuse_norm_utils
from infoquant.models.llama_patched import LlamaForCausalLM
from infoquant.quantization import gptq as gptq_utils
from infoquant.quantization import quantizers as quant_utils
from infoquant.rotation import apply as rotation_utils
from infoquant.rotation import hadamard as hadamard_utils
from infoquant.utils import runtime as utils


def load_rotated_model(args, config, device):
    transformers.set_seed(args.seed)

    # Rotate the weights
    if args.num_hidden_layers>0:
        config.num_hidden_layers = args.num_hidden_layers
    if args.rotate:
        process_word_embeddings = False
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False
            process_word_embeddings = True
        dtype = torch.float16
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.input_model,
            config=config,
            torch_dtype=dtype,
        )
        model.eval()
        model.seqlen = args.model_max_length
        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, device, args)

        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)

        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        for name in qlayers:
            if "down_proj" in name:
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(args.input_model,
                                                              torch_dtype='auto',
                                                              config=config,
                                                              low_cpu_mem_usage=True)
        fuse_norm_utils.fuse_layer_norms(model)
        model.seqlen = args.model_max_length
        model.eval()

    if args.w_bits < 16:

        if args.w_rtn:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, device, args)

        else:  # GPTQ Weight Quantization

            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.input_model,
                seqlen=args.model_max_length,
                eval_mode=False,
            )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, device, args)

    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            # print(name)
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            lac = args.lac
            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision

                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                    lac=lac
                )

            if "o_proj" in name:
                layer_groupsize = head_dim
            if "R1R4" in name:

                if "0" in name or "1" in name or "2" in name:
                    layer_input_bits = 16
                    lac = False
                else:
                    layer_input_bits = args.residual_bits
            if "R4R6" in name:

                if "0" in name or "1" in name or "2" in name:
                    layer_input_bits = 16
                    lac = False
                else:
                    layer_input_bits = args.residual_bits
            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision

                if args.int8_down_proj:
                    layer_input_bits = 8

                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                lac=lac
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
                "lac": args.lac
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model

def rotated_model(args, model, device, config,logger=None, rotation=False):
    transformers.set_seed(args.seed)

    # Rotate the weights

    if args.num_hidden_layers>0:
        config.num_hidden_layers = args.num_hidden_layers

    if rotation:
        model.eval()
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, device, args,logger)

        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)

        had_K, K = hadamard_utils.get_hadK(config.intermediate_size)
        for name in qlayers:
            if "down_proj" in name:
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(args.input_model,
                                                              torch_dtype=torch.float16,
                                                              config=config,
                                                              low_cpu_mem_usage=True)
        model.seqlen = args.model_max_length
        model.eval()
    return model

def set_quant_state(args, layer,layer_idx, device, w_bits, a_bits, v_bits, k_bits, config,attention_mask=None,position_idx=None,inps=None):

    if w_bits < 16:
        if args.w_rtn:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd_layer(layer, device, args)
        else:  # GPTQ Weight Quantization
            if inps is None:
                raise NotImplementedError("GPTQ need inps")
            quantizers = gptq_utils.gptq_fwrd_layer(layer,layer_idx, inps,attention_mask,position_idx,device, args)

    if a_bits < 16 or v_bits < 16:
        qlayers = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                layer, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio
            lac = args.lac
            num_heads = config.num_attention_heads
            model_dim = config.hidden_size
            head_dim = model_dim // num_heads
            group_num = -1
            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                group_num = config.num_key_value_heads
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                    lac=lac,
                    group_num=group_num
                )

            if "o_proj" in name:
                layer_groupsize = head_dim
                group_num = config.num_attention_heads
            if "R1R4" in name or "R4R6" in name:
                layer_input_bits = args.residual_bits
                if layer_input_bits>=16:
                    lac = False
            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision

                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize
            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                lac=lac,
                group_num=group_num
            )

    if k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            group_num = config.num_attention_heads
            k_quant_config = {
                "k_bits": k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
                "lac": args.lac,
                "group_num": group_num
            }

            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                config=config,
                **k_quant_config,
            )
    return layer
