import logging

from callbacks.output.output_callback import OutputCallback


class Logger(OutputCallback):
    def __init__(self, metrics=['epoch', 'train_loss'], each_n_epochs=1):
        super().__init__(each_n_epochs)
        self.metrics = metrics

    def on_show(self, ctx):
        logging.info({m: ctx[m] for m in self.metrics if m in ctx and ctx[m] is not None})
