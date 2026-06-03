import torch
import os

def find_outlier(value, k):
    p = value

    mean = torch.mean(p)
    std = torch.std(p)
    indx = (p < mean - k * std) | (p > mean + k * std)
    return torch.any(indx, dim=-1)


def Mask_token(Activations, layer_idx, K, samples, args, logger=None):
    output = f"{args.cache}/K/"

    logger.info(f"Computing mask tokens for {layer_idx} layer")

    select_K = {}
    for layer_idx in Activations.keys():
        select_K[layer_idx] = {}
        for key2 in Activations[layer_idx].keys():
            select_K[layer_idx][key2] = tuple()

            number1 = []
            number2 = []
            for k in range(K):
                add_matrix = find_outlier(Activations[layer_idx][key2][0], k)
                start_massive = 0
                for i in range(samples):
                    position = find_outlier(Activations[layer_idx][key2][i], k)
                    start_massive += torch.sum(position)
                    add_matrix = add_matrix | position
                outlier = torch.sum(add_matrix)
                start_massive = start_massive / samples
                if outlier == 0:
                    break
                number1.append(start_massive)
                number2.append(outlier)

            select_K[layer_idx][key2] = (number1, number2)

    if not os.path.exists(output + f'{args.model_name}_layer_{layer_idx}_K.pth'):
        os.makedirs(output, exist_ok=True)
        torch.save(select_K,output + f'{args.model_name}_layer_{layer_idx}_K.pth')
