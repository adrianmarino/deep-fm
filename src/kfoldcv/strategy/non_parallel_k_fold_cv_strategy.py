from kfoldcv.strategy.k_fold_cv_strategy import KFoldCVStrategy


class NonParallelKFoldCVStrategy(KFoldCVStrategy):
    def perform(self, train_fold_fn, dataset, params, folds):
        return [train_fold_fn(dataset, train_idx, val_idx, params, fold) for fold, (train_idx, val_idx) in
                enumerate(folds)]
