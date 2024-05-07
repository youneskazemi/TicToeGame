import numpy as np

markers = np.array(["X", "O"])


class Player:
    def __init__(self, marker):
        self.marker = marker
        self.enemy_marker = markers[np.where(markers != self.marker)][0]

    def random_move(self, board):
        while True:
            idx = np.random.choice(board.shape[0], 2)
            if board[idx[0], idx[1]] == "":
                board[idx[0], idx[1]] = self.marker
                break

    def human_move(self, board):
        x = int(input("x: "))
        y = int(input("y: "))
        while (
            (board[int(y) - 1][int(x) - 1] == "")
            and not (x < 4 and x > 0)
            and not (y < 4 and y > 0)
        ):
            print("Invalid Inputs!")
            x = int(input("x: "))
            y = int(input("y: "))

        board[int(x) - 1][int(y) - 1] = self.marker

    def make_move(self, board, weights):
        boards = []
        for i in range(board.shape[0]):
            for j in range(board.shape[0]):
                if board[i][j] == "":
                    new_board = board.copy()
                    new_board[i][j] = self.marker
                    boards.append((new_board, (i, j)))

        val = float("-inf")
        for i in range(len(boards)):
            features = self.get_features(boards[i][0])
            curr_val = self.evalApproximation(features, weights)
            if val < curr_val:
                val = curr_val
                best_move = boards[i][1]
        board[best_move[0]][best_move[1]] = self.marker

    def get_features(self, board):
        """
        0-bias
        1-col-row,diag two own and one empty
        2-col-row-diag two enemy and one empty
        3-own center
        4-number of own corners
        5-col-row,diag one own and two empty
        6-col-row,diag three own
        """

        features = [1, 0, 0, 0, 0, 0, 0]

        # Check center field
        if board[1, 1] == self.marker:
            features[3] += 1

        # Check corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for corner in corners:
            if board[corner[0], corner[1]] == self.marker:
                features[4] += 1

        # Check rows, columns, and diagonals
        for i in range(3):
            own_row = np.count_nonzero(board[i] == self.marker)
            empty_row = np.count_nonzero(board[i] == "")

            # Rows
            if own_row == 2 and empty_row == 1:
                features[1] += 1
            elif own_row == 1 and empty_row == 2:
                features[5] += 1
            elif own_row == 3:
                features[6] += 1

            # Columns
            own_column = np.count_nonzero(board[:, i] == self.marker)
            empty_column = np.count_nonzero(board[:, i] == "")
            if own_column == 2 and empty_column == 1:
                features[1] += 1
            elif own_column == 1 and empty_column == 2:
                features[5] += 1
            elif own_column == 3:
                features[6] += 1

        # Diagonals
        own_diagonal = np.count_nonzero(np.diag(board) == self.marker)
        empty_diagonal = np.count_nonzero(np.diag(board) == "")
        if own_diagonal == 2 and empty_diagonal == 1:
            features[1] += 1
        elif own_diagonal == 1 and empty_diagonal == 2:
            features[5] += 1
        elif own_diagonal == 3:
            features[6] += 1

        anti_diagonal = np.count_nonzero(np.diag(np.fliplr(board)) == self.marker)
        empty_anti_diagonal = np.count_nonzero(np.diag(np.fliplr(board)) == "")
        if anti_diagonal == 2 and empty_anti_diagonal == 1:
            features[1] += 1
        elif anti_diagonal == 1 and empty_anti_diagonal == 2:
            features[5] += 1
        elif anti_diagonal == 3:
            features[6] += 1

        return features

    def evalApproximation(self, features, weights):
        return weights.T @ features
