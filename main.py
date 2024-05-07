import random
from tictoe import play, play_vs_ai
from train import fit, load, plot_weights
from utils import *


if __name__ == "__main__":
    train = input("fit model? y or n: ")

    while train not in ("y", "n"):
        train = input("Play? y or n: ")

        if train == "n":
            break

    if train == "y":
        weights = fit(lr=0.0025, epochs=5000)
    else:
        weights = load(r"./model_weights.pkl")

    play(verbose=True)
    play(weights, verbose=True)

    play(weights, verbose=True, aagent_vs_agent=True)

    play_vs_ai(weights)
