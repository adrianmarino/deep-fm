from torch.optim import lr_scheduler

from callbacks import Callback


class ReduceLROnPlateau(Callback):
    def __init__(self, mode='min', factor=0.1, patience=10, metric='val_loss'):
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
