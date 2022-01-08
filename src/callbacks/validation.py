from callbacks import Callback
from modules.fn import Fn


class Validation(Callback):
    """
    This callback allows to validation model performance into a validation using one or more specified validation metrics.
    """
    def __init__(self, data_loader, metrics, each_n_epochs=1):
        """
        Validation callback constrcutor.
        :param data_loader: validation set data_loader.
        :param metrics: a dict with each metric name and calculation function. i.e.:

            metrics={
                'val_loss': lambda y_pred, y_true: BCELoss()(y_pred, y_true).item(),
                'val_auc':  lambda y_pred, y_true: roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
            }

        :param each_n_epochs: number of epochs to wait to calculate validation metrics.
        """
        self.data_loader = data_loader
        self.metrics = metrics
        self.each_n_epochs = each_n_epochs

    def on_after_train(self, ctx):
        if ctx.epoch % self.each_n_epochs == 0:
            y_pred, y_true = Fn.validation(ctx.model, self.data_loader, ctx.device)
            ctx['time'] = ctx.stopwatch.to_str()
            for (name, fn) in self.metrics.items():
                ctx[name] = fn(y_pred, y_true)
        else:
            for (name, fn) in self.metrics.items():
                ctx[name] = None
