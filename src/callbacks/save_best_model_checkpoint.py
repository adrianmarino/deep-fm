import logging
from datetime import datetime

import torch
from pytorch_common.callbacks import Callback

from callbacks.metric_improve_mixin import MetricImproveMixin
from utils.os_utils import create_path


class SaveBestModel(MetricImproveMixin, Callback):

    def __init__(self, metric, path='./weights', mode='min', experiment_name='model', time_pattern='%Y-%m-%d_%H-%M-%S'):
        self._init('checkpoint', metric, mode)
        self.__path = create_path(path)
        self.__time_pattern = time_pattern
        self.__experiment_name = experiment_name

    def _on_improvement(self, ctx, mode):
        file_path = self.__get_file_path(ctx)
        logging.info("Save best model: {}".format(file_path))
        torch.save(ctx.model.state_dict(), file_path)
        ctx['best_model_path'] = file_path

    def __get_file_path(self, ctx):
        return '{}/{}--{}--epoch_{}--{}_{}.pt'.format(
            self.__path,
            datetime.now().strftime(self.__time_pattern),
            self.__experiment_name,
            ctx.epoch,
            self._metric_name,
            self._metric(ctx)
        )
