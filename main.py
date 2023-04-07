import torch


from src.model import init_model


def main():
    model, gradients = init_model(50)

    # maybe put something here to save the trained model such that it can be loaded when running
    # and doesn't need to be initialized every time


if __name__ == "__main__":
    main()
