from callbacks import Callback
from modules.fn import Fn
from util.stopwatch import Stopwatch


class Validation(Callback):
    def __init__(self, data_loader, metrics, each_n_epochs=1):
        self.data_loader = data_loader
        self.metrics = metrics
        self.each_n_epochs = each_n_epochs

    def on_after_train(self, ctx):
        if ctx.epoch % self.each_n_epochs == 0:
            stopwatch = Stopwatch()
            y_pred, y_true = Fn.validation(ctx.model, self.data_loader, ctx.device)
            ctx['time'] = stopwatch.to_str()
            for (name, fn) in self.metrics.items():
                ctx[name] = fn(y_pred, y_true)
        else:
            for (name, fn) in self.metrics.items():
                ctx[name] = None
