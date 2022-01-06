class CommonMixin:
    @property
    def params(self):
        return {k: v.data for k, v in list(self.named_parameters())}

    @property
    def device(self):
        return next(self.parameters()).device
