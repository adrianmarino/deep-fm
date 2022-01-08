from callbacks import Callback


class EarlyStop(Callback):
    """
    Stop training when model has stopped improving a specified metric.
    """
    def __init__(self, metric, mode='min', patience=10):
        """
        :param metric (str): Metric used to check model performance improving.
        :param mode (str): One of `min`, `max`. In `min` mode check that metric go down after each epoch.
        :param patience (int): Number of epochs with no metric improvement.
        """
        self.__metric = metric
        self.__mode = mode
        self.__patience = patience

    def on_init(self, ctx):
        ctx['patience'] = 0

    def last_metric_name(self):
        return 'last_{}'.format(self.__metric)

    def update_last_metric(self, ctx):
        ctx[self.last_metric_name()] = ctx[self.__metric]

    def get_last_metric(self, ctx):
        return ctx[self.last_metric_name()]

    def has_metric(self, ctx):
        return self.__metric in ctx

    def has_last_metric(self, ctx):
        return self.last_metric_name() in ctx

    def on_after_train(self, ctx):
        if self.has_metric(ctx):
            if self.has_last_metric(ctx):
                if self.__mode == 'min':
                    ctx['patience'] = 0 if ctx[self.__metric] < self.get_last_metric(ctx) else ctx.patience + 1
                else:
                    ctx['patience'] = 0 if ctx[self.__metric] > self.get_last_metric(ctx) else ctx.patience + 1

                ctx['early_stop'] = (ctx['patience'] == self.__patience)

            self.update_last_metric(ctx)
