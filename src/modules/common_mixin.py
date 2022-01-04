class CommonMixin:
    @property
    def params(self):
        return {k: v.data for k, v in list(self.named_parameters())}
