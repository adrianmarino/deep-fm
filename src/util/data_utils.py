from torch.utils.data import random_split


def train_val_test_split(X, train_percent=0.7, val_percent=0.15):
    train_length = int(X.shape[0] * train_percent)
    val_length = int(X.shape[0] * val_percent)
    test_length = X.shape[0] - train_length - val_length

    return random_split(X, (train_length, val_length, test_length))


def train_val_split(X, train_percent=0.7):
    train_length = int(X.shape[0] * train_percent)
    val_length = X.shape[0] - train_length

    return random_split(X, (train_length, val_length))
