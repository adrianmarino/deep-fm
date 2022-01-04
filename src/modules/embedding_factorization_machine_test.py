import unittest

import torch
from torch import FloatTensor, Tensor

from modules.embedding_factorization_machine import EmbeddingFactorizationMachine
from tensor_utils import tensor_eq


def tensor_round(tensor, n_digits=2):
    return torch.round(tensor * 10 ** n_digits) / (10 ** n_digits)


class EmbeddingFactorizationMachineTest(unittest.TestCase):
    def test_forward_one_feature_vector(self):
        # Prepare
        embeddings = FloatTensor([
            [  # Feature vector
                [0.1, 0.2],  # Feature 1 embedding vector
                [0.2, 0.3]  # Feature 2 embedding vector
            ]
        ])
        layer = EmbeddingFactorizationMachine()

        # Perform
        y = layer(embeddings)

        # Asserts
        self.assertEqual(y.shape, torch.Size((1, 1)))
        self.assertTrue(tensor_eq(tensor_round(y), Tensor([[0.08]])))
