#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from typing import Any

device = None


# initializes GPU
def init_gpu(use_gpu: bool = True, gpu_id: int = 0, verbose: bool = True) -> None:
    """Initializes torch device."""
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        if verbose:
            print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU not detected. Defaulting to CPU.")


# save model
def save(net: torch.nn, optimizer: torch.optim, save_path: str) -> None:
    state = {"state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, save_path)


# restore model
def restore(restore_path: str) -> Any:
    return torch.load(restore_path)


# save model
def save_model(net: torch.nn, save_path: str) -> None:
    state = {"state_dict": net.state_dict()}
    torch.save(state, save_path)


# sin activation function
class sin_act(torch.nn.Module):
    def __init__(self):
        super(sin_act, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# sin activation function
class Rsin(torch.nn.Module):
    def __init__(self):
        super(Rsin, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), torch.sin(x))

        return torch.sin(x)
