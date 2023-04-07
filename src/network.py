import torch
import torch.nn as nn


# run on GPU if available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class TorchNet(nn.Module):
    """
    PyTorch neural network. Network layers are defined in __init__ and forward
    pass implemented in forward.

    Args:
        in_features: number of features in input layer
        hidden_dim: number of features in hidden dimension
        out_features: number of features in output layer
    """

    def __init__(self, in_features, hidden_dim, out_features):
        super(TorchNet, self).__init__()

        self.layer1 = nn.Linear(in_features, hidden_dim, device=device)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, out_features, device=device)
        # self.relu2 = nn.ReLU()

        self.layers = [self.layer1, self.relu1, self.layer2]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
