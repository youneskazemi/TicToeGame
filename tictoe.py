import random
from utils import *
from player import Player

player1 = Player("X")
player2 = Player("O")


def play(weights=None, verbose=False, aagent_vs_agent=False):

    if aagent_vs_agent:
        print("Trained Model Vs Trained Model")
    else:
        if weights is None:
            print("Random Player Vs Random Player")
        else:
            print("Trained Model Vs Random Player")
    count_win = 0
    count_loss = 0
    count_draw = 0
    for i in range(1000):
        board = np.empty((3, 3), dtype=str)
        start = random.randint(0, 1)
        while True:
            if start == 1:
                (
                    player1.random_move(board)
                    if weights is None
                    else player1.make_move(board, weights)
                )
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break
                (
                    player2.random_move(board)
                    if aagent_vs_agent is False
                    else player2.make_move(board, weights)
                )
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break
            else:
                (
                    player2.random_move(board)
                    if aagent_vs_agent is False
                    else player2.make_move(board, weights)
                )
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break
                (
                    player1.random_move(board)
                    if weights is None
                    else player1.make_move(board, weights)
                )
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break
        if outcome[0] == 1 and outcome[1].marker == "O":
            count_loss += 1
        elif outcome[0] == 0:
            count_draw += 1
        elif outcome[0] == 1 and outcome[1].marker == "X":
            count_win += 1
    win_rate = count_win / (count_win + count_draw + count_loss)

    if verbose:
        print("Wins: " + str(count_win))
        print("Draws: " + str(count_draw))
        print("Loss: " + str(count_loss))
        print(f"Win Rate : {win_rate * 100}%")

    return count_win, count_draw, count_loss, win_rate


def play_vs_ai(weights):
    player1 = Player("X")

    player2 = Player("O")

    while True:
        board = np.empty((3, 3), dtype=str)
        # start = random.randint(0, 1)
        start = 0
        x = input("Play? y or n: ")

        while x not in ("y", "n"):
            x = input("Play? y or n: ")

        if x == "n":
            break

        while True:
            if start == 0:
                print("Bot Move")

                player1.make_move(board, weights)
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break

                print_board(board)
                print("Human Move")

                player2.human_move(board)
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break

            else:
                print_board(board)
                print("Human Move")

                player2.human_move(board)
                outcome = finished(board, player2)
                if outcome[0] != -1:
                    break
                print("Bot Move")

                player1.make_move(board, weights)
                print(player1.get_features(board))
                outcome = finished(board, player1)
                if outcome[0] != -1:
                    break
        if outcome[0] == 1 and outcome[1].marker == "O":
            print("Human won")

        elif outcome[0] == 0:
            print("Draw")

        elif outcome[0] == 1 and outcome[1].marker == "X":
            print("Bot won.")
        print_board(board)
