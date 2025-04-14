import torch,math
import numpy as np
import torch.nn as nn

actications = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "softmax": nn.Softmax(dim=-1),
}

class MLP(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_features, num_hidden_layer,activation="relu"):
        super().__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_features = hidden_features
        self.num_hidden_layer = num_hidden_layer
        self.activation = actications[activation]

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feature, hidden_features))
        self.layers.append(self.activation)
        for i in range(num_hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(hidden_features, out_feature))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    
class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=10):
        super().__init__()

        self.in_features = in_features

        self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class NeuralOcto(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers, num_frequencies=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers

        self.pos_encoding = PosEncodingNeRF(in_features, num_frequencies)

        self.mlp = MLP(self.pos_encoding.out_dim, out_features, hidden_features, num_hidden_layers)

    def forward(self, coords):
        coords_pos_enc = self.pos_encoding(coords)
        return self.mlp(coords_pos_enc)