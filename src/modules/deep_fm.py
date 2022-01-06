from torch import sigmoid
from torch.nn import Module

from modules.categorial_features_linear import CategoricalFeaturesLineal
from modules.embedding_factorization_machine import EmbeddingFactorizationMachine
from modules.mixin.fit_mixin import FitMixin
from modules.mlp import MultiLayerPerceptron
from modules.multi_feature_embedding import MultiFeatureEmbedding


class DeepFM(Module, FitMixin):
    def __init__(self, features_n_values: list[int], embedding_size, units_per_layer, dropout):
        super().__init__()
        self.lineal = CategoricalFeaturesLineal(features_n_values)
        self.fm = EmbeddingFactorizationMachine()
        self.embedding = MultiFeatureEmbedding(features_n_values, embedding_size)
        self.__init_mlp(features_n_values, embedding_size, units_per_layer, dropout)

    def __init_mlp(self, features_n_values, embedding_size, units_per_layer, dropout):
        # Embedding output must be flattened to pass through mlp...
        self.mlp_input_dim = len(features_n_values) * embedding_size
        self.mlp = MultiLayerPerceptron(self.mlp_input_dim, units_per_layer, dropout)

    def __flatten_emb(self, emb): return emb.view(-1, self.mlp_input_dim)

    def forward(self, x):
        x_emb = self.embedding(x)

        output = self.lineal(x) + self.fm(x_emb) + self.mlp(self.__flatten_emb(x_emb))

        return sigmoid(output.squeeze(1))
