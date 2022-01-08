from torch.optim import lr_scheduler

from callbacks import Callback


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced. See ReduceLROnPlateau.
    """
    def __init__(self, mode='min', factor=0.1, patience=10, metric='val_loss'):
        """
        :param mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        :param factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        :param patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        :param metric: metric used to control learning rate reduction.
        """
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.metric = metric

    def on_init(self, ctx):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            ctx.optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience
        )

    def on_after_train(self, ctx):
        if self.metric in ctx and ctx[self.metric] is not None:
            self.scheduler.step(ctx[self.metric])

        for param_group in ctx.optimizer.param_groups:
            ctx['lr'] = param_group['lr']
