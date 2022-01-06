from abc import abstractmethod

from bunch import Bunch

from callbacks import Callback


class OutputCallback(Callback):
    def __init__(self, each_n_epochs=50): self.each_n_epochs = each_n_epochs

    def is_plot_time(self, ctx): return ctx.epoch % self.each_n_epochs == 0

    def _is_last_epoch(self, ctx): return ctx.epochs == ctx.epoch

    def _is_first_epoch(self, ctx): return ctx.epoch == 1

    def can_plot(self, ctx):
        return (self._is_last_epoch(ctx) or (
                not self._is_first_epoch(ctx) and self.is_plot_time(ctx))) and ctx.verbose > 0

    def on_after_train(self, ctx):
        if self.can_plot(ctx):
            self.on_show(ctx)

    @abstractmethod
    def on_show(self, ctx):
        pass
