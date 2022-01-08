import numpy as np
from sklearn.model_selection import StratifiedKFold

from kfoldcv.strategy.non_parallel_k_fold_cv_strategy import NonParallelKFoldCVStrategy


class KFoldCV:
    def __init__(
            self,
            model_train_fn,
            get_y_values_fn,
            strategy=NonParallelKFoldCVStrategy(),
            k_fold=5,
            random_state=42,
            shuffle=True
    ):
        self.__model_train_fn = model_train_fn
        self.__get_y_values_fn = get_y_values_fn
        self.__strategy = strategy
        self.__folder = StratifiedKFold(
            n_splits=k_fold,
            shuffle=shuffle,
            random_state=random_state
        )

    def train(self, dataset, params):
        np.random.seed(params['seed'])

        folds = self.__folder.split(dataset, self.__get_y_values_fn(dataset))

        scores = self.__strategy.perform(
            self.__model_train_fn,
            dataset,
            params,
            folds
        )

        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'scores': scores
        }
