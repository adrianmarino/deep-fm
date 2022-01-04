import numpy as np
from torch.nn import Module, Embedding
from torch.nn.init import xavier_uniform_

from modules.common_mixin import CommonMixin


class MultiFeatureEmbedding(Module, CommonMixin):

    def __init__(self, features_n_values: list[int], embedding_size: int):
        """
            This layer allows creates only one embedding for multiples feature variables. Because is not necessary
            create an embedding layer for each input feature allowing save memory.

        :param features_n_values: A list that contains count of all possible values for each categorical feature.
                             i.e.: [3, 2, 5], to create an embedding for 3 categorical features with 3, 2, and 5
                            possible values respectively.

        :param embedding_size: Embedding vector len. Same for all features.
        """
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=sum(features_n_values),
            embedding_dim=embedding_size
        )
        xavier_uniform_(self.embedding.weight.data)

        self.feat_emb_offset = np.concatenate((np.zeros(1), np.cumsum(features_n_values)[:-1]), axis=0)

    def _sum_emb_offset(self, x): return x + x.new_tensor(self.feat_emb_offset).unsqueeze(0)

    def forward(self, x):
        """
            Replace input values for embedding vectors. Find into embedding lookup table indexing by input values
            and get the associated embedding vector.

        :param batches: A list of batches. Each batch contains a list of feature vectors.
        :return: Same batches list but this contains embeddings instead of categorical input values.
        """
        return self.embedding(self._sum_emb_offset(x))

    @property
    def vectors(self): return self.params['embedding.weight']
