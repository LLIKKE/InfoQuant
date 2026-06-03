import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
from infoquant.rotation.hadamard import random_hadamard_matrix

dtype_mapping = {
    'float32': torch.float32,
    'float64': torch.float64,
    'int32': np.int32,
    'int64': np.int64,
}
def load_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")



def find_stable_start(ncradio, outlier, nthreshold, othreshold, window_size, token_radio=0.2, offset_k=4):
    assert len(ncradio) == len(outlier)
    idx_n = -1
    idx_o = -1
    for i in range(len(ncradio) - window_size + 1):
        n_window = ncradio[i:i + window_size]
        o_window = outlier[i:i + window_size]
        if np.mean(n_window) > np.mean(ncradio) and np.mean(o_window) < np.mean(outlier) and outlier[
            i] < token_radio * np.max(outlier):
            diff = np.abs(np.diff(n_window))
            if np.any(diff < nthreshold):
                idx_n = i
            diff = np.abs(np.diff(o_window))
            if np.any(diff < othreshold):
                idx_o = i
            if idx_o != -1 and idx_n != -1:
                return min(idx_n, idx_o) + offset_k
    if idx_o == -1 and idx_n == -1:
        return len(ncradio) - 2 + offset_k
    else:
        k = min(idx_n, idx_o) if (idx_n != -1 and idx_o != -1) else max(idx_n, idx_o)
    return k + offset_k


def find_outlier(value, k, per_token):
    p = value
    if per_token:
        mean = np.mean(p, axis=-1, keepdims=True)
        std = np.std(p, axis=-1, keepdims=True)
    else:
        mean = np.mean(p)
        std = np.std(p.astype(np.float64))
    indx = (p < mean - k * std) | (p > mean + k * std)
    return indx


def aug_token(value, k):

    indx = find_outlier(value, k, False)
    idx = np.any(indx, axis=-1)
    return idx


class CustomActivationDataset(Dataset):
    def __init__(self, Activation, layer_idx, args=None):

        if args is None:
            self.dtype = dtype_mapping['float32']
        else:
            self.dtype = dtype_mapping[args.dtype]
        self.layer_idx = layer_idx

        if layer_idx >= args.aug_start:
            self.aug_token = args.aug_token
        else:
            self.aug_token = -1

        self.activation = Activation[layer_idx]


        self.keys = {
            "X": f'input_layernorm',
            "V": f'self_attn.v_proj',
            "M": f'self_attn.o_proj',
            "P": f'post_attention_layernorm',
        }

        self.nsamples = len(self.activation[self.keys['X']])

        if self.aug_token > 1:
            self._compute_aug_weights_processes(args)

        self.pad = torch.ones([args.model_max_length, 1])

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        def safe_load(data, key, idx):
            return data[key][idx].squeeze().to(dtype=self.dtype)

        # 加载基础数据
        X = safe_load(self.activation, self.keys['X'], idx)
        P = safe_load(self.activation, self.keys['P'], idx)
        V = safe_load(self.activation, self.keys['V'], idx)
        M = safe_load(self.activation, self.keys['M'], idx)
        

        if self.aug_token > 1:
            X_aug = self.aug_weight[self.keys['X']][idx]
            P_aug = self.aug_weight[self.keys['P']][idx]
            V_aug = self.aug_weight[self.keys['V']][idx]
            M_aug = self.aug_weight[self.keys['M']][idx]
        else:

            X_aug = P_aug = V_aug = M_aug = self.pad

        return X, P, V, M, X_aug, P_aug, V_aug, M_aug

    def _compute_aug_weights_processes(self, args, nthreshold=0.1, othreshold=100, window_size=3):
        self.aug_weight = {}
        processes = []
        manager = multiprocessing.Manager()
        queue = manager.Queue()  # 使用Queue来传递结果
        path = f"{args.cache}/K/"+f'{args.model_name}_layer_{self.layer_idx}_K.pth'
        K_list = torch.load(path)[self.layer_idx]

        position = 0

        for key in self.activation.keys():
            process = multiprocessing.Process(target=self._compute_aug_weights,
                                              args=(args, key, K_list, position, nthreshold, othreshold, window_size,
                                                  queue))
            position += 1
            process.daemon = True
            processes.append(process)
            process.start()
        for process in processes:
            process.join()

        while not queue.empty():
            key, weights = queue.get()
            self.aug_weight[key] = weights

    def _compute_aug_weights(self, args, key, K_list, position, nthreshold=0.1, othreshold=100, window_size=3,
                             queue=None):
        weights = []
        start_massive = np.array(K_list[key][0])
        outlier = np.array(K_list[key][1])
        ncradio = (outlier - start_massive) / outlier
        k=len(ncradio)
        for i in range(1, len(ncradio)):
            if abs(ncradio[i] - ncradio[i - 1])<0.02 and outlier[i] < 0.4 * 2048:
                k=i
                break

        for i in tqdm(range(self.nsamples), desc=f"Ada k={k}: {key}", position=position):
            weights.append(torch.from_numpy(aug_token(self.activation[key][i].numpy(), k).reshape(-1, 1).astype(np.int32)))
        if queue is not None:
            queue.put((key, weights))
