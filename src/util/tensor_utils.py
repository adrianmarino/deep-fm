import torch


def tensor_eq(a, b):
    return torch.all(a.eq(b))
