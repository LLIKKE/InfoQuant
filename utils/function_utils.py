import math
import numpy as np
import torch
from train_utils.optimizer import SGDG
from scipy.linalg import qr
from collections import OrderedDict
import os

def load_clip_parameters(args, model, path=None):
    if path is None:
        clip_parameters = torch.load(os.path.join(args.exp_dir, f"clip_parameters.pth"))
    else:
        clip_parameters = torch.load(os.path.join(path, f"clip_parameters.pth"))
    layers = model.model.layers

    for i in range(len(layers)):
        clip_param = clip_parameters[i]
        layers[i].load_state_dict(clip_param, strict=False)
    return model

def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a*a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_random_orthg(size):
    H = np.random.randn(size, size)
    Q, R = qr(H)
    Q_modified = Q @ np.diag(np.sign(np.diag(R)))
    return torch.from_numpy(Q_modified)


def get_init_weight(dim, ):
    return get_random_orthg(dim)


def get_inverse(matrix):
    dtype = matrix.dtype
    return matrix.double().inverse().to(dtype)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return iter(params)


def get_paras_dict_by_name(model, required_names, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                destination[prefix + name] = param.detach()
    return destination


def check_params_grad(model):
    for name, param in model.named_parameters():
        print(name, ':{}'.format(param.requires_grad))
    return


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f}%")


def set_require_grad_all(model, requires_grad):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return



class optims():
    def __init__(self, stages, epochs, learning_rates, args=None):
        self.optimizers = []
        self.schedulers = []
        for i in range(len(stages)):
            self.optimizers.append(
                SGDG(stages[i].parameters(), lr=learning_rates[i], stiefel=True, device=args.device)
            )
            self.schedulers.append(
                torch.optim.lr_scheduler.LinearLR(self.optimizers[i], start_factor=1.0, end_factor=args.lr_scheduler,
                                                  total_iters=epochs))

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def backward(self, losses):
        for loss in losses:
            loss.backward()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def lr_step(self):
        for scheduler in self.schedulers:
            scheduler.step()
