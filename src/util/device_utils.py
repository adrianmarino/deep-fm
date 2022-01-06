import os

import torch


def set_device_name(name):
    os.environ['DEVICE'] = name


def get_device_name():
    return os.environ['DEVICE'] if 'DEVICE' in os.environ else None


def get_device(i=0):
    name = get_device_name()

    if 'gpu' in name:
        name = f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else 'cpu'

    return torch.device(name)


def set_device_memory(device_name, process_memory_fraction=0.5):
    if 'gpu' in device_name:
        torch.cuda.set_per_process_memory_fraction(
            process_memory_fraction,
            get_device()
        )
        torch.cuda.empty_cache()
