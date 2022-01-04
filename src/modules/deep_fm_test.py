import unittest

import torch

from modules.deep_fm import DeepFM


class DeepFMTest(unittest.TestCase):
    def test_forward_one_batch_of_size_two(self):
        # Prepare...
        model = DeepFM(
            features_n_values=[2, 3],
            embedding_size=4,
            units_per_layer=[20, 20, 20],
            dropout=0.2
        )
        X = torch.LongTensor([
            [0, 0],
            [1, 2]
        ])

        # Perform...
        y = model(X)

        # Asserts...
        self.assertEqual(y.shape[0], 2)
        [self.assertTrue(0 <= o <= 1) for o in y]
