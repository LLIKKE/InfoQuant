import os
import time
import gc
import functools
from contextlib import nullcontext
import torch.nn.functional as F
import torch
import torch.nn as nn
import transformers


import datetime
from logging import Logger

from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import load_rotated_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from transformers import AutoTokenizer

from termcolor import colored
import pprint
import warnings
from Activation_hook import hook_Activation
from optimize_rotation_softmax import optimize_rotation_matrix
from outlier_token_mask import Mask_token
from utils import fuse_norm_utils

def Peak_Suppression_Orthogonal_Transformation(model, dataloader, dev, config, args, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    dtype = torch.float16
    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    # same input of first layer for fp model and quant model
    fp_inps = inps  # take output of fp model as input
    num_train_layer = len(layers)

    for i in range(num_train_layer):
        logger.info("optimize layer {}".format(i))
        layer = layers[i].to(dev)

        with torch.no_grad():
            Activation, fp_outs = hook_Activation(layer, i, fp_inps, attention_mask=attention_mask,
                                                  position_ids=position_ids, args=args, logger=logger)
        fp_inps = fp_outs
        layer = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        if args.aug_token > 0 and args.aug_start <= i:
            Mask_token(Activation, layer_idx=i, K=50, samples=args.eta_m, args=args, logger=logger)

        rotation_matrix = optimize_rotation_matrix(Activation, i, args.epochs, args.T, args.lrs, args.batch_size,
                                                   config, args, logger)

        torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def train() -> None:
    args, logger = process_args_ptq()

    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)

    config = transformers.AutoConfig.from_pretrained(
        args.input_model
    )
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    logger.info(f"rotating model: {args.input_model}")

    model = transformers.LlamaForCausalLM.from_pretrained(args.input_model,
                                                          torch_dtype=torch.float16,
                                                          config=config,
                                                          low_cpu_mem_usage=True).cpu()
    logger.info(f"fuse layer norm weight")
    fuse_norm_utils.fuse_layer_norms(model)
    model.seqlen = args.model_max_length
    model.config.use_cache = False

    if os.path.exists(f"{args.cache}/calidata_{args.nsamples}.pth"):
        trainloader = torch.load(f"{args.cache}/calidata_{args.nsamples}.pth")
        logger.info(
            f"load calibration data from cache : {args.cache}/trainloader_{len(trainloader)}.pth, nsamples: {len(trainloader)}")
    else:
        trainloader = data_utils.get_wikitext2(
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.input_model,
            seqlen=args.model_max_length,
            eval_mode=False
        )
        torch.save(trainloader, f"{args.cache}/calidata_{args.nsamples}.pth")
        logger.info(
            f"load calibration data from cache : {args.cache}/calidata_{len(trainloader)}.pth, nsamples: {len(trainloader)}")

    Peak_Suppression_Orthogonal_Transformation(model, trainloader, args.device, config, args, logger)
    model = model.cpu()
    del model

    torch.cuda.empty_cache()
    model = load_rotated_model(args, config, args.device)
    model.config.use_cache = False
    if os.path.exists(f"{args.cache}/wikitext.pth"):
        testloader = torch.load(f"{args.cache}/wikitext.pth")
        logger.info(
            f"load wikitext test from cache : {args.cache}/wikitext.pth")
    else:
        testloader = data_utils.get_wikitext2(
            seed=args.seed,
            seqlen=args.model_max_length,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        torch.save(testloader, f"{args.cache}/wikitext.pth")
        logger.info(
            f"load wikitext test from cache : {args.cache}/wikitext.pth")

    dataset_ppl = eval_utils.evaluator(model, testloader, args.device, args)
    logger.info(f"ppl: {dataset_ppl}")


if __name__ == "__main__":
    train()
