import torch


from src.model import init_model
from src.algo1 import algo1


def main():
    model, gradients = init_model(200)

    algo1(gradients)

    # maybe put something here to save the trained model such that it can be loaded when running
    # and doesn't need to be initialized every time


if __name__ == "__main__":
    main()
