import os

import torch

from infoquant.calibration import datasets


def load_or_create_calibration_loader(args, logger):
    os.makedirs(args.cache, exist_ok=True)
    cache_path = os.path.join(args.cache, f"calidata_{args.nsamples}.pth")
    if os.path.exists(cache_path):
        trainloader = torch.load(cache_path)
        logger.info(
            f"load calibration data from cache : {cache_path}, nsamples: {len(trainloader)}"
        )
        return trainloader

    trainloader = datasets.get_wikitext2(
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.input_model,
        seqlen=args.model_max_length,
        eval_mode=False,
    )
    torch.save(trainloader, cache_path)
    logger.info(
        f"save calibration data to cache : {cache_path}, nsamples: {len(trainloader)}"
    )
    return trainloader


def load_or_create_wikitext_eval(args, tokenizer, logger):
    os.makedirs(args.cache, exist_ok=True)
    cache_path = os.path.join(args.cache, "wikitext.pth")
    if os.path.exists(cache_path):
        testloader = torch.load(cache_path)
        logger.info(f"load wikitext test from cache : {cache_path}")
        return testloader

    testloader = datasets.get_wikitext2(
        seed=args.seed,
        seqlen=args.model_max_length,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    torch.save(testloader, cache_path)
    logger.info(f"save wikitext test to cache : {cache_path}")
    return testloader
