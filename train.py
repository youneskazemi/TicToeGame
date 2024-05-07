import numpy as np
import matplotlib.pyplot as plt
from player import Player
from utils import *


def train_model(player1, player2, lr=0.001, weights=np.ones(7), epochs=10000):
    weights_history = [weights.copy()]
    print("Start Training ...")
    for i in range(epochs):
        turn = np.random.choice([0, 1])
        board = np.empty((3, 3), dtype=str)
        while True:
            pre_features = player1.get_features(board)
            pre_approx = player1.evalApproximation(pre_features, weights)
            if turn == 0:
                player1.make_move(board, weights)
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break
                player2.make_move(board, weights)
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break
            else:
                player2.make_move(board, weights)
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break
                player1.make_move(board, weights)
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break
            succ_features = player1.get_features(board)
            succ_approx = player1.evalApproximation(succ_features, weights)
            weight_update(
                weights=weights,
                learning_rate=lr,
                train_val=succ_approx,
                approx=pre_approx,
                features=pre_features,
            )
        if outcome[0] == 1 and outcome[1].marker == "O":
            result = -1
        elif outcome[0] == 0:
            result = 0
        elif outcome[0] == 1 and outcome[1].marker == "X":
            result = 1
        curr_features = player1.get_features(board)
        curr_approx = player1.evalApproximation(curr_features, weights)
        weight_update(
            weights=weights,
            learning_rate=lr,
            train_val=result,
            approx=curr_approx,
            features=pre_features,
        )

        if i % 1000 == 0:
            print(f"[INFO] : {i + epochs // 10} / {epochs} PROCESSED")

        weights_history.append(weights.copy())

    return weights, weights_history


def plot_weights(weights_history):
    num_weights = len(weights_history[0])
    plt.figure(figsize=(10, 6))
    for i in range(num_weights):
        plt.plot(
            range(len(weights_history)),
            [weights[i] for weights in weights_history],
            label=f"Weight {i+1}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Weight Value")
    plt.title("Weight Changes Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def fit(
    weights=np.ones(7), weights_filename="model_weights.pkl", lr=0.001, epochs=10000
):
    player1 = Player("X")
    player2 = Player("O")
    trained_weights, weights_history = train_model(
        player1=player1, player2=player2, lr=lr, weights=weights, epochs=epochs
    )
    save_weights(trained_weights, weights_filename)
    print(f"Weights Trained !")
    plot_weights(weights_history)
    return trained_weights


def load(weights_filename):
    try:
        weights = load_weights(weights_filename)
        print(f"Weights loaded from {weights_filename}")
        return weights
    except FileNotFoundError:
        raise "Could not found weights file!"
