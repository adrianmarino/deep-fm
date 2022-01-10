import unittest

from torch import LongTensor, Size

from modules.categorial_features_linear import CategoricalFeaturesLineal


class CategoricalFeaturesLinealTest(unittest.TestCase):
    def test_create_lineal_for_two_categorical_features(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        output = 1

        # Perform...
        layer = CategoricalFeaturesLineal(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            output
        )

        # Asserts...
        weights_shape = layer.params['embedding.embedding.weight'].shape
        bias_shape = layer.params['bias'].shape
        self.assertEqual(weights_shape[0], cat_feature_1_possible_values + cat_feature_2_possible_values)
        self.assertEqual(weights_shape[1], output)
        self.assertEqual(bias_shape[0], output)

    def test_forward_two_feature_vectors(self):
        # Prepare...
        cat_feature_1_possible_values = 2
        cat_feature_2_possible_values = 5
        n_output = 1

        layer = CategoricalFeaturesLineal(
            [cat_feature_1_possible_values, cat_feature_2_possible_values],
            n_output
        )
        X = LongTensor([
            [0, 0],  # Features vector 1
            [1, 2]  # Features vector 2
        ])

        # Perform...
        y = layer(X)

        # Asserts...
        self.assertEqual(y.shape, Size([y.shape[0], n_output]))
