from abc import ABCMeta
from abc import abstractmethod


class Callback(metaclass=ABCMeta):

    @staticmethod
    def invoke_on_init(ctx, callbacks):
        [it.on_init(ctx) for it in callbacks]
        return ctx

    @staticmethod
    def invoke_on_after_train_batch(ctx, callbacks):
        [it.on_after_train_batch(ctx) for it in callbacks]
        return ctx

    @staticmethod
    def invoke_on_after_train(ctx, callbacks):
        [it.on_after_train(ctx) for it in callbacks]
        return ctx

    def on_init(self, ctx):
        pass

    def on_after_train_batch(self, ctx):
        pass

    @abstractmethod
    def on_after_train(self, ctx):
        pass
