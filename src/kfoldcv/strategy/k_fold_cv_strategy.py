from abc import ABCMeta
from abc import abstractmethod


class KFoldCVStrategy(metaclass=ABCMeta):
    @abstractmethod
    def perform(self, train_fold_fn, dataset, params, folds):
        pass
