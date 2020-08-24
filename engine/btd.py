import numpy as np
import numpy.linalg as LA
import torch
import tensorly
from tensorly.decomposition import partial_tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tqdm import trange

tensorly.set_backend("pytorch")

def pci(cfg, tensor, groups, bottle_dim, remainders=[]):
    # principle component iteration
    # input:
    #       kernel tensor - tensor s x t x p (in_channels x out_channels x spatial);
    #       group number - groups
    # output: [tensor_1,tensor_2, ..., tensor_G]
    # initialize tensor_i <- 0 for i = 1, ..., G
    min_step = cfg.DECOMPOSITION.MIN_STEP
    max_iter = cfg.DECOMPOSITION.MAX_ITER

    s, t = bottle_dim
    rank = (int(s / groups), int(t / groups))

    btd_tensors = [torch.zeros_like(tensor) for _ in range(groups)]
    error_last = 100
    for _ in trange(int(max_iter)):
        btd_cores_and_factors = []
        # for i = 1 to G do
        for i in range(groups):
            # tensor_res <- tensor - sum(tensor_1, ..., tensor_i-1, tensor_i+1, ..., tensor_G)
            tensor_res = tensor - (sum(btd_tensors) - btd_tensors[i])
            # tensor_i <- HOOI(tensor_res)
            btd_i = partial_tucker(tensor_res, modes=[0, 1], rank=rank)
            btd_cores_and_factors.append(btd_i)
            tensor_i = tucker_to_tensor(btd_i)
            btd_tensors[i] = tensor_i
        # tensor_res <- tensor - sum(tensor_1, tensor_2, ..., tensor_G)
        tensor_res = tensor - sum(btd_tensors)
        error = torch.norm(tensor_res).item() / torch.norm(tensor).item()
        delta = error_last - error
        # until tensor_res ceases to decrease or maximum iterations reached
        if abs(delta) <= min_step:
            break
        error_last = error
    if remainders:
        btd_cores_and_factors = [
            (c - rc, (f0 - rf0, f1 - rf1))
            for (c, (f0, f1)), (rf0, rc, rf1) in zip(btd_cores_and_factors, remainders)
        ]
    return btd_cores_and_factors, tensor_res