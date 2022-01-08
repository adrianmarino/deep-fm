from bunch import Bunch

from callbacks import Callback
from modules.fn import Fn
from modules.mixin.common_mixin import CommonMixin
from util.stopwatch import Stopwatch


class FitMixin(CommonMixin):
    def fit(
            self,
            data_loader,
            loss_fn,
            epochs,
            optimizer,
            callbacks=[],
            verbose=1
    ):
        """
        Train a model.
        :param data_loader: data_loader with train set.
        :param loss_fn: function to minimize.
        :param epochs: number of epochs to train model.
        :param optimizer: optimizer used to adjust model.
        :param callbacks: callback collection. See Callback.
        :param verbose: show/hide logs.
        """
        ctx = Bunch({
            'verbose': verbose,
            'epochs': epochs,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'device': self.device,
            'model': self,
            'stopwatch': Stopwatch()
        })

        Callback.invoke_on_init(ctx, callbacks)
        for epoch in range(epochs):
            ctx.stopwatch.reset()
            ctx['epoch'] = epoch + 1
            ctx['train_loss'] = Fn.train(
                self,
                data_loader,
                loss_fn,
                optimizer,
                ctx.device
            )
            ctx['time'] = ctx.stopwatch.to_str()
            Callback.invoke_on_after_train(ctx, callbacks)

            if 'early_stop' in ctx and ctx.early_stop is True:
                break
