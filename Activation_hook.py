import os
import pickle
import torch
import transformers
from utils import  data_utils, quant_utils,fuse_norm_utils
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm , LlamaForCausalLM


def hook_Activation(layer, layer_idx, fp_inps, attention_mask, position_ids, args, logger):
    logger.info("hook activation...")

    Activations = dict()
    handles = []

    fulls = quant_utils.find_qlayers(layer, layers=[LlamaRMSNorm, ])

    def add_batch(name, layer_idx, output=True):
        def tmp(_, inp, out):
            if layer_idx not in Activations:
                Activations[layer_idx] = {}
            if name not in Activations[layer_idx]:
                Activations[layer_idx][name] = []
            if output:
                Activations[layer_idx][name].append(out.data.detach().cpu())
            else:
                Activations[layer_idx][name].append(inp[0].data.detach().cpu())
            assert len(out.data.shape) == 3

        return tmp

    for name, module in fulls.items():
        # we don't save RMSNorm output before lm_head
        if "layer" in name:
            handles.append(module.register_forward_hook(add_batch(name, layer_idx, True)))

    fulls = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear, ])
    for name, module in fulls.items():
        if "v_proj" in name:
            handles.append(module.register_forward_hook(add_batch(name, layer_idx, True)))
        elif "o_proj" in name:
            handles.append(module.register_forward_hook(add_batch(name, layer_idx, False)))

    with torch.no_grad():
        for j in range(args.nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[
                0].cpu()

    for h in handles:
        h.remove()

    return Activations, fp_inps

if __name__ == '__main__':
    pass