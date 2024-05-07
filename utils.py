import numpy as np
import pickle


def print_board(board):
    for row in board:
        print(" | ".join([str(cell).center(3) for cell in row]))
        print("-" * (5 * len(row) - 1))


def weight_update(weights, learning_rate, train_val, approx, features):
    for i in range(len(weights)):
        weights[i] += learning_rate * (train_val - approx) * features[i]


def finished(board, player):
    for i in range(board.shape[0]):
        if np.all(board[i, :] == player.marker):
            return 1, player

    for j in range(board.shape[0]):
        if np.all(board[:, j] == player.marker):
            return 1, player

    if (
        board[0][0] == player.marker
        and board[1][1] == player.marker
        and board[2][2] == player.marker
    ):
        return 1, player
    elif (
        board[0][2] == player.marker
        and board[1][1] == player.marker
        and board[2][0] == player.marker
    ):
        return 1, player

    if "" in board:
        return -1, player
    else:
        return 0, player


def save_weights(weights, filename):
    with open(filename, "wb") as f:
        pickle.dump(weights, f)


def load_weights(filename):
    with open(filename, "rb") as f:
        weights = pickle.load(f)
        return weights

