import os
import time
import gc
import functools
from contextlib import nullcontext
import torch.nn.functional as F
import torch
import torch.nn as nn
import transformers

from utils.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, \
    check_params_grad, load_clip_parameters

import datetime

import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import rotated_model, set_quant_state
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from transformers import AutoTokenizer

from termcolor import colored
import pprint
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def Lac(args, model, dataloader, dev, config, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.bfloat16 if isinstance(model, transformers.Qwen2ForCausalLM) else torch.float16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

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
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()
    inps = inps.cpu()
    # same input of first layer for fp model and quant model
    fp_inps = inps.cpu()  # take output of fp model as input
    fp_outs = torch.zeros_like(inps).cpu()  # take output of fp model as input

    loss_func = torch.nn.MSELoss()
    # start training
    clip_parameters = {}
    num_train_layer = len(layers)

    for i in range(num_train_layer):
        logger.info(f"Quant and lac layer: {i}")
        dtype_dict = {}

        layer = layers[i].to(dev)

        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(fp_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids)[0].cpu()

        layer = set_quant_state(args, layers[i], i, dev, w_bits=args.w_bits, a_bits=args.a_bits, k_bits=args.k_bits,
                                v_bits=args.v_bits, config=config, attention_mask=attention_mask,
                                position_idx=position_ids, inps=fp_inps).to(dev)

        if not args.lac or args.cali_epochs <= 0 or (args.a_bits>=16 and args.k_bits>=16 and args.v_bits>=16):
            fp_inps, fp_outs = fp_outs.cpu(), fp_inps.cpu()

            for name, param in layer.named_parameters():
                param.requires_grad = False
            end_type = torch.bfloat16 if args.bf16 else torch.float16
            layers[i] = layer.to(end_type).to("cpu")
            torch.cuda.empty_cache()
            continue

        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []

        if args.lac:
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.cali_lr})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cali_epochs * (
                args.nsamples // args.cali_bsz), eta_min=args.cali_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main

        for epoch in range(args.cali_epochs):
            mse = 0
            start_tick = time.time()

            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    quant_out = layer(fp_inps[index:index + args.cali_bsz, ].to(dev), attention_mask=attention_mask_batch,
                                      position_ids=position_ids)[0]
                    loss = loss_func(fp_outs[index:index + args.cali_bsz, ].to(dev), quant_out)

                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

            logger.info(
                f"layer {i} lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, loss: {mse:.8f}")

        fp_inps, fp_outs = fp_outs.cpu(), fp_inps.cpu()

        for name, param in layer.named_parameters():
            param.requires_grad = False

        end_type = torch.bfloat16 if args.bf16 else torch.float16
        layers[i] = layer.to(end_type).to("cpu")

        clip_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(clip_parameters, os.path.join(args.exp_dir, f"clip_parameters.pth"))

        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model


def train() -> None:
    args, logger = process_args_ptq()

    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)

    config = transformers.AutoConfig.from_pretrained(
        args.input_model, token=None
    )

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        config=config,
        torch_dtype=dtype,
        token=None,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model.seqlen = args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=None,
    )

    model.config.use_cache = False

    logger.info(f"rotating model: {args.input_model}")

    model = rotated_model(args, model, args.device, config, logger, args.rotate)

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

    logger.info(f"lac model: {args.input_model}")
    model = Lac(args, model, trainloader, args.device, config, logger)

    if args.clip_parameter is not None:
        load_clip_parameters(args,model,args.clip_parameter)
        model = model.to(torch.bfloat16 if args.bf16 else torch.float16)

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

    torch.cuda.empty_cache()

    if args.zero_shot:
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM
        import lm_eval
        task_manager = TaskManager()

        model.to(args.device)
        tasks = task_manager.match_tasks(args.tasks)
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", max_batch_size=256, trust_remote_code=True)
        results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size="auto", max_batch_size=256)
        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in
                       results['results'].items()}
        mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in
                    results['results'].items()}
        mean_std_val = round(sum(std_vals.values()) / len(std_vals.values()), 4)
        metric_vals['acc_avg'] = mean_acc_val
        results['results']['AVERAGE'] = {
            "acc,none": mean_acc_val,
            "acc_stderr,none": mean_std_val
        }
        logger.info("\n" + make_table(results))
        print(metric_vals)


if __name__ == "__main__":
    train()
