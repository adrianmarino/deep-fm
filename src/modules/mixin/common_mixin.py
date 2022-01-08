class CommonMixin:
    @property
    def params(self):
        """
        Get all model parameters
        :return: parameters dict
        """
        return {k: v.data for k, v in list(self.named_parameters())}

    @property
    def device(self):
        """
        Get device assigned to model
        :return: device
        """
        return next(self.parameters()).device
