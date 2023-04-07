import torch
import os
from pathlib import Path

from src.model import init_model
from src.algo1 import algo1

from src.network import TorchNet
from src.constants import *


def main():
    if not Path("models/").exists():
        os.mkdir(Path("models/"))

    # model, gradients = init_model(200)
    # torch.save(model.state_dict(), Path("models", "model.dat"))

    try:
        model = TorchNet(in_features, hidden_dim, out_features)
        model.load_state_dict(torch.load(Path("models", "model.dat")))

        dLdw2 = model.layer2.weight.grad
        dLdb2 = model.layer2.bias.grad
        dLdw1 = model.layer1.weight.grad
        dLdb1 = model.layer1.bias.grad
        gradients = [dLdw1, dLdb1, dLdw2, dLdb2]
    except FileNotFoundError:
        model, gradients = init_model(200)
        torch.save(model.state_dict(), Path("models", "model.dat"))

    algo1(gradients)


if __name__ == "__main__":
    main()
