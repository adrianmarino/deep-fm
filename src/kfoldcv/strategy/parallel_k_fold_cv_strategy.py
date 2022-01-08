from multiprocessing import Pool

from kfoldcv.strategy.k_fold_cv_strategy import KFoldCVStrategy


class ParallelKFoldCVStrategy(KFoldCVStrategy):
    def __init__(self, n_processes):
        self.__n_processes = n_processes

    def perform(self, train_fold_fn, dataset, params, folds):
        with Pool(processes=self.__n_processes) as pool:
            _params = [(dataset, train_idx, val_idx, params, fold) for fold, (train_idx, val_idx) in enumerate(folds)]
            return pool.starmap(train_fold_fn, _params)
