from torch.nn import Module, Linear, BatchNorm1d, ReLU, Dropout, Sequential

from modules.common_mixin import CommonMixin


class MultiLayerPerceptron(Module, CommonMixin):

    def __init__(self, input_units, units_per_layer, dropout, n_outputs=1):
        super().__init__()
        layers = []

        for units in units_per_layer:
            layers.extend([
                Linear(input_units, units),
                BatchNorm1d(units),
                ReLU(),
                Dropout(p=dropout)
            ])
            input_units = units

        layers.append(Linear(input_units, n_outputs))

        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
