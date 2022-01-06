from bunch import Bunch

from callbacks import Callback
from modules.fn import Fn
from modules.mixin.common_mixin import CommonMixin


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
        ctx = Bunch({
            'verbose': verbose,
            'epochs': epochs,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'device': self.device,
            'model': self
        })

        Callback.invoke_on_init(ctx, callbacks)
        for epoch in range(epochs):
            ctx['epoch'] = epoch + 1
            ctx['train_loss'], ctx['time'] = Fn.train(self, data_loader, loss_fn, optimizer, ctx.device)
            Callback.invoke_on_after_train(ctx, callbacks)
