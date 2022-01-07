# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

from callbacks.early_stop_callback import EarlyStop

sys.path.append('./src')

import warnings

warnings.filterwarnings("ignore")

import logging
from bunch import Bunch
from torch.utils.data import DataLoader

from callbacks.output import Logger
from callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from callbacks.validation import Validation
from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from logger import initialize_logger
from modules import DeepFM
from util.data_utils import train_val_split
from util.device_utils import set_device_name, set_device_memory, get_device

from sklearn.metrics import roc_auc_score

import click

from torch.nn import BCELoss
from torch.optim import Adam


# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def cv_train(ps):
    train_set, val_set = train_val_split(ps.dataset, ps.train_percent)

    model = DeepFM(
        ps.features_n_values,
        ps.embedding_size,
        ps.units_per_layer,
        ps.dropout
    ).to(ps.device)

    logging.info('Start training...')
    model.fit(
        data_loader=DataLoader(train_set, ps.batch_size, num_workers=ps.num_workers),
        loss_fn=BCELoss(),
        epochs=ps.epochs,
        optimizer=Adam(
            params=model.parameters(),
            lr=ps.lr,
            weight_decay=ps.weight_decay
        ),
        callbacks=[
            Validation(
                data_loader=DataLoader(val_set, ps.batch_size, num_workers=ps.num_workers),
                metrics={
                    'val_loss': lambda y_pred, y_true: BCELoss()(y_pred, y_true).item(),
                    'val_auc': lambda y_pred, y_true: roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                },
                each_n_epochs=1
            ),
            Logger(['time', 'epoch', 'train_loss', 'val_loss', 'val_auc', 'patience', 'lr']),
            ReduceLROnPlateau(metric='val_auc', mode='max', factor=ps.lr_factor, patience=ps.lr_patience),
            EarlyStop(metric='val_auc', mode='max', patience=3)
        ]
    )


def load_dataset(name):
    if '1m' == name:
        dataset_path = '../datasets/ml-1m/ratings.dat'
        dataset = MovieLens1MDataset(dataset_path=dataset_path)
    else:
        dataset_path = '../datasets/ml-20m/ratings.csv'
        dataset = MovieLens20MDataset(dataset_path=dataset_path)

    logging.info('{} dataset loaded! Shape: {}'.format(dataset_path, dataset.shape))

    return dataset


# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@click.command()
@click.option(
    '--device',
    default='gpu',
    help='Device used to functions and optimize model. Values: gpu, cpu.'
)
@click.option(
    '--cuda-process-memory-fraction',
    default=0.5,
    help='Setup max memory user per CUDA process. Percentage expressed between 0 and 1.'
)
@click.option(
    '--dataset',
    default='1m',
    help='select movie lens dataset type. Values: 1m(default), 20m.'
)
def main(device, cuda_process_memory_fraction, dataset):
    initialize_logger()
    set_device_name(device)
    set_device_memory(device, cuda_process_memory_fraction)

    ds = load_dataset(dataset)

    cv_train(Bunch({
        'lr': 0.001,
        'lr_factor': 0.1,
        'lr_patience': 1,
        'weight_decay': 1e-6,
        'epochs': 50,
        'embedding_size': 100,
        'units_per_layer': [300, 300],
        'dropout': 0.8,
        'batch_size': 4000,
        'train_percent': 0.7,
        'num_workers': 12,
        'features_n_values': ds.field_dims,
        'dataset': ds,
        'device': get_device()
    }))


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
