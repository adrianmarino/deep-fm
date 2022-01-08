import logging

from callbacks.output.output_callback import OutputCallback


class Logger(OutputCallback):
    """
    Logs context properties. Ingeneral is used to log performance metrics every n epochs. i.e.:

        metrics=['time', 'epoch', 'train_loss', 'val_loss', 'val_auc', 'patience', 'lr']

    """
    def __init__(self, metrics=['epoch', 'train_loss'], each_n_epochs=1):
        super().__init__(each_n_epochs)
        self.metrics = metrics

    def on_show(self, ctx):
        logging.info({m: ctx[m] for m in self.metrics if m in ctx and ctx[m] is not None})
