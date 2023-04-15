import torch
import os
import numpy
import pandas as pd
from pathlib import Path

from src.model import init_model
from src.algo1 import algo1

from src.network import TorchNet
from src.constants import *


def main(retrain=False):
    if not Path("models/").exists():
        os.mkdir(Path("models/"))

    # Try loading a saved instance of the model instead of retraining
    try:
        # Cause the saved model to not be found when a new model is requested
        if retrain:
            raise FileNotFoundError

        # Load model (obsolete?)
        model = TorchNet(in_features, hidden_dim, out_features)
        model.load_state_dict(torch.load(Path("models", "model.dat")))

        # Extract the gradients from the gradient file
        df = pd.read_pickle('models/gradients.pkl')
        gradients = []
        for index, row in df.iterrows():
            gradients.append(torch.tensor(row[0]))

    except FileNotFoundError:
        # Train a new network and save its weights
        model, gradients = init_model(200)
        torch.save(model.state_dict(), Path("models", "model.dat"))
        grad = pd.DataFrame(numpy.array([x.tolist() for x in gradients]))
        grad.to_pickle('models/gradients.pkl')

    # First part of the reconstruction algorithm
    g_mc = algo1(gradients)
    print("Printing output of Algorithm 1:")
    print(g_mc.shape)
    print(g_mc)


if __name__ == "__main__":
    main(retrain=False)
