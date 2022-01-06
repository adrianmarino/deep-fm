import logging

from callbacks.output.output_callback import OutputCallback


class Logger(OutputCallback):
    def __init__(self, metrics=['epoch', 'train_loss'], each_n_epochs=5):
        super().__init__(each_n_epochs)
        self.metrics = metrics

    def on_show(self, ctx):
        logging.info({name: value for (name, value) in ctx.items() if name in self.metrics and value is not None})
