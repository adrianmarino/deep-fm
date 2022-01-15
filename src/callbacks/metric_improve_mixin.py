class MetricImproveMixin:
    def _init(self, last_metric_prefix, metric, mode='min'):
        """
        :param metric (str): Metric used to check model performance improving.
        :param mode (str): One of `min`, `max`. In `min` mode check that metric go down after each epoch.
        """
        self.__last_metric_prefix = last_metric_prefix
        self._metric_name = metric
        self._previous_metric_name = 'previous_{}'.format(self._metric_name)
        self._mode = mode

    def __update_last_metric(self, ctx):
        if self.__has_metric(ctx):
            ctx[self._previous_metric_name] = self._metric(ctx)

    def _previous_metric(self, ctx):
        return ctx[self._previous_metric_name]

    def _metric(self, ctx):
        return ctx[self._metric_name]

    def __has_metric(self, ctx):
        return self._metric_name in ctx

    def __has_last_metric(self, ctx):
        return self._previous_metric_name in ctx

    def on_after_train(self, ctx):
        if self.__has_metric(ctx):
            if self.__has_last_metric(ctx):
                if self._mode == 'min' and ctx[self._metric_name] < self._previous_metric(ctx):
                    self._on_improvement(ctx, self._mode)
                elif self._mode == 'max' and ctx[self._metric_name] > self._previous_metric(ctx):
                    self._on_improvement(ctx, self._mode)

            self.__update_last_metric(ctx)

    def _on_improvement(self, ctx, mode):
        pass
