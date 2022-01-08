from abc import ABCMeta
from abc import abstractmethod


class Callback(metaclass=ABCMeta):
    """
    A callback allows to add extra behavior to modules. a.e.:
        - Add Metrics validation.
        - log/plot metrics after ech epoch.
        - Reduce learning rate then model stop improving.
        - etc...

    You could create you own callback. A callback has two events:
        - on_init: Invoked before start training.
        - on_after_train: Invoked after each epoch.

    A callback receive a context as argument, that allows access model data like:
        - a model reference.
        - epochs, current epoch
        - optimizer.
        - loss_fn
        - device
        - stopwatch: Give you current training time from beginning.

    Note: a callback can introduce new context data tih is a normal behavior.
    """
    @staticmethod
    def invoke_on_init(ctx, callbacks):
        [it.on_init(ctx) for it in callbacks]
        return ctx

    @staticmethod
    def invoke_on_after_train(ctx, callbacks):
        [it.on_after_train(ctx) for it in callbacks]
        return ctx

    def on_init(self, ctx):
        pass

    @abstractmethod
    def on_after_train(self, ctx):
        pass
