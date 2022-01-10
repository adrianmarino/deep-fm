from pytorch_common.modules import CommonMixin
from torch import sum
from torch.nn import Module


class EmbeddingFactorizationMachine(Module, CommonMixin):
    def forward(self, x):
        square_of_sum = sum(x, dim=1) ** 2
        sum_of_squares = sum(x ** 2, dim=1)
        return 0.5 * sum((square_of_sum - sum_of_squares), dim=1, keepdim=True)
