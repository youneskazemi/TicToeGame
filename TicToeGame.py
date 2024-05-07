import pygame
import numpy as np
from utils import load_weights, finished
from player import Player

pygame.init()

player1 = Player("O")
player2 = Player("X")

WIDTH, HEIGHT = 300, 350
LINE_WIDTH = 3
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

player1_wins = 0
player2_wins = 0

weights_filename = "model_weights.pkl"
try:
    weights = load_weights(weights_filename)
    print(f"Weights loaded from {weights_filename}")
except FileNotFoundError:
    print("Error: Trained weights not found!")
    pygame.quit()
    quit()

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")


def draw_grid():
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(
            win, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH
        )
        pygame.draw.line(
            win,
            BLACK,
            (i * SQUARE_SIZE, 0),
            (i * SQUARE_SIZE, HEIGHT - 65),
            LINE_WIDTH,
        )


def draw_marks(board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == player2.marker:
                pygame.draw.line(
                    win,
                    RED,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE),
                    ((col + 1) * SQUARE_SIZE, (row + 1) * SQUARE_SIZE),
                    LINE_WIDTH,
                )
                pygame.draw.line(
                    win,
                    RED,
                    ((col + 1) * SQUARE_SIZE, row * SQUARE_SIZE),
                    (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE),
                    LINE_WIDTH,
                )
            elif board[row][col] == player1.marker:
                pygame.draw.circle(
                    win,
                    BLUE,
                    (
                        int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                        int(row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    ),
                    SQUARE_SIZE // 2 - LINE_WIDTH,
                    LINE_WIDTH,
                )


def get_mouse_pos(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


def is_valid_move(board, row, col):
    return board[row][col] == ""


def display_game_over(winner=None):
    global player1_wins, player2_wins
    font = pygame.font.Font(None, 40)
    if winner is None:
        text = font.render("Draw!", True, BLACK)

    elif winner.marker == player2.marker:
        text = font.render(f"{player2.marker} wins!", True, RED)
        player2_wins += 1
    elif winner.marker == player1.marker:
        text = font.render(f"{player1.marker} win!", True, BLUE)
        player1_wins += 1

    win.blit(
        text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2)
    )
    pygame.display.update()
    pygame.time.delay(3000)


def play_again_prompt():
    font = pygame.font.Font(None, 30)
    text = font.render("Play again? (Y/N)", True, BLACK)
    win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 50))
    pygame.display.update()


def play_with_agent_prompt():
    font = pygame.font.Font(None, 30)
    text = font.render("Play versus agent? (Y/N)", True, WHITE)
    win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 50))
    pygame.display.update()


def draw_scoreboard():
    font = pygame.font.Font(None, 30)
    text_player1 = font.render(f"Player 1 wins: {player1_wins}", True, BLUE)
    text_player2 = font.render(f"Player 2 wins: {player2_wins}", True, RED)
    win.blit(text_player1, (10, HEIGHT - 60))
    win.blit(text_player2, (10, HEIGHT - 30))


def play_game():
    global player1_wins, player2_wins
    agent = False
    play_with_agent_prompt()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    agent = True
                    break
                elif event.key == pygame.K_n:
                    break
        else:
            continue
        break

    turns = True
    while True:
        board = np.empty((3, 3), dtype=str)
        board.fill("")
        player_turn = player2.marker if turns else player1.marker
        # player_turn = player2.marker

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and player_turn == player1.marker
                ):
                    row, col = get_mouse_pos(pygame.mouse.get_pos())
                    if is_valid_move(board, row, col):
                        board[row][col] = player1.marker
                        player_turn = player2.marker
                elif (
                    player_turn == player2.marker
                    and event.type == pygame.MOUSEBUTTONDOWN
                ):
                    row, col = get_mouse_pos(pygame.mouse.get_pos())
                    if is_valid_move(board, row, col):
                        board[row][col] = player2.marker
                        player_turn = player1.marker
                elif agent and player_turn == player2.marker:
                    player2.make_move(board, weights)
                    player_turn = player1.marker

            win.fill(WHITE)
            draw_grid()
            draw_marks(board)
            draw_scoreboard()
            pygame.display.update()
            stats_player1, winner_player1 = finished(board, player1)
            stats_player2, winner_player2 = finished(board, player2)

            if stats_player1 == 1:
                display_game_over(winner_player1)
                play_again_prompt()
                break
            elif stats_player2 == 1:
                display_game_over(winner_player2)
                play_again_prompt()
                break
            elif stats_player1 == 0 or stats_player2 == 0:
                display_game_over()
                play_again_prompt()
                break

        turns = not turns

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        break
                    elif event.key == pygame.K_n:
                        pygame.quit()
                        quit()
            else:
                continue
            break


play_game()
