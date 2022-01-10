from torch import sum, zeros
from torch.nn import Parameter, Module
from torch.nn.init import normal_

from pytorch_common.modules import CommonMixin
from modules.multi_feature_embedding import MultiFeatureEmbedding


class CategoricalFeaturesLineal(Module, CommonMixin):
    def __init__(self, features_n_values: list[int], n_outputs=1):
        super().__init__()
        self.embedding = MultiFeatureEmbedding(
            features_n_values=features_n_values,
            embedding_size=n_outputs
        )
        self.bias = Parameter(zeros((n_outputs,)))
        normal_(self.bias.data)

    def forward(self, x): return sum(self.embedding(x), dim=1) + self.bias
