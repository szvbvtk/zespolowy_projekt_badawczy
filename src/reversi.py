import numpy as np
from mcts import State
from numba import jit
from numba import int8


class Reversi(State):
    N = M = 8

    # SYMBOLS = ["\u25cb", "+", "\u25cf"]
    SYMBOLS = ["\u25cf", "+", "\u25cb"]

    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
        else:
            self.board = np.zeros((Reversi.M, Reversi.N), dtype=np.int8)
            mid_row = Reversi.M // 2
            mid_col = Reversi.N // 2

            self.board[mid_row - 1, mid_col - 1] = -1  # biale
            self.board[mid_row, mid_col] = -1  # biale
            self.board[mid_row - 1, mid_col] = 1  # czarne
            self.board[mid_row, mid_col - 1] = 1  # czarne

            # self.board[0, 3] = -1
            # self.board[0, 4] = -1
            # self.board[0, 5] = -1
            # self.board[0, 6] = -1
            # self.board[0, 7] = -1
            # self.board[1, 3] = -1
            # self.board[1, 4] = -1
            # self.board[1, 5] = -1
            # self.board[1, 6] = -1
            # self.board[1, 7] = -1
            # self.board[2, 4] = -1
            # self.board[2, 6] = -1
            # self.board[2, 7] = -1
            # self.board[3, 5] = -1
            # self.board[3, 6] = -1
            # self.board[3, 7] = -1
            # self.board[4, 6] = -1
            # self.board[4, 7] = -1
            # self.board[5, 7] = -1
            # self.board[4, 0] = -1
            # self.board[5, 0] = -1
            # self.board[5, 1] = -1
            # self.board[6, 0] = -1
            # self.board[7, 0] = -1
            # self.board[7, 1] = -1
            # self.board[7, 2] = -1
            # self.board[7, 3] = -1
            # self.board[7, 4] = -1

            # self.board[3, 3] = 1
            # self.board[4, 4] = 1
            # self.board[3, 4] = 1
            # self.board[4, 3] = 1
            # self.board[2, 3] = 1
            # self.board[2, 5] = 1
            # self.board[4, 5] = 1
            # self.board[4, 2] = 1
            # self.board[5, 2] = 1
            # self.board[5, 3] = 1
            # self.board[5, 4] = 1
            # self.board[6, 2] = 1

    @staticmethod
    def class_repr():
        return f"{Reversi.__name__}_{Reversi.M}x{Reversi.N}"

    def __str__(self):
        s = ""

        for row_idx in range(Reversi.M):
            s += f"{row_idx + 1}| "
            for col_idx in range(Reversi.N):
                s += f"{Reversi.SYMBOLS[self.board[row_idx, col_idx] + 1]} "

            s += "\n"

        s += "   "
        for col_idx in range(Reversi.N):
            s += f"{chr(ord('A') + col_idx)} "

        s += "\n"
        s += f"Ruch: {'Czarny' if self.turn == 1 else 'Biały'}\n"

        return s

    def take_action_job(self, action_index):
        if action_index == Reversi.M * Reversi.N:
            self.turn *= -1
            return True

        row = action_index // Reversi.N
        col = action_index % Reversi.N

        pawns_indices = self.get_pawns_to_flip(action_index)
        if not pawns_indices[0]:
            return False

        self.board[row, col] = self.turn

        self.board[pawns_indices[0], pawns_indices[1]] = self.turn

        # if self.has_legal_actions(-self.turn):
        #     self.turn *= -1

        self.turn *= -1

        return True

    def compute_outcome_job(self):
        if self.has_legal_actions(self.turn):
            return None

        if self.has_legal_actions(-self.turn):
            return None

        player_1_points = np.sum(self.board == 1)
        player_minus_1_points = np.sum(self.board == -1)

        if player_1_points > player_minus_1_points:
            return 1
        elif player_minus_1_points > player_1_points:
            return -1
        else:
            return 0

    def has_legal_actions(self, turn):
        for action_index in range(Reversi.M * Reversi.N):
            if self.get_pawns_to_flip(action_index, turn)[0]:
                return True
        return False

    def get_all_legal_actions(self, turn):
        legal_actions = []
        for action_index in range(Reversi.M * Reversi.N):
            if self.get_pawns_to_flip(action_index, turn)[0]:
                legal_actions.append(action_index)
        return legal_actions

    def take_random_action_playout(self):
        legal_actions = self.get_all_legal_actions(self.turn)

        if legal_actions:
            random_action_index = np.random.choice(legal_actions)
            child = self.take_action(random_action_index)
            return child
        else:
            child = self.take_action(Reversi.M * Reversi.N)
            return child

        # random_action_index = np.random.choice(legal_actions)

        # child = self.take_action(random_action_index)

        # return child

    def get_board(self):
        return self.board

    def get_extra_info(self):
        no_white_pawns = np.sum(self.board == -1)
        no_black_pawns = np.sum(self.board == 1)

        return np.array([no_white_pawns, no_black_pawns], dtype=np.uint8)

    @staticmethod
    def action_name_to_index(action_name):
        if action_name == "-":
            return Reversi.M * Reversi.N

        col = action_name[0].upper()
        row = int(action_name[1]) - 1
        i = row
        j = ord(col) - ord("A")
        return i * Reversi.N + j

    @staticmethod
    def action_index_to_name(action_index):
        # pusta akcja
        if action_index == Reversi.M * Reversi.N:
            return "-"

        row = action_index // Reversi.N
        col = action_index % Reversi.N

        return f"{chr(ord('A') + col)}{row + 1}"

    @staticmethod
    def get_board_shape():
        return (Reversi.M, Reversi.N)

    @staticmethod
    def get_extra_info_memory():
        return 2

    @staticmethod
    def get_max_actions():
        return Reversi.M * Reversi.N + 1

    def get_pawns_to_flip(self, action_index, turn=None):
        start_row = action_index // Reversi.N
        start_col = action_index % Reversi.N
        pawns_to_flip_coords = [[], []]  # wiersze, kolumny

        if self.board[start_row, start_col] != 0:
            return pawns_to_flip_coords

        player = self.turn if turn is None else turn
        opponent = -player

        for horizontal in (-1, 0, 1):
            for vertical in (-1, 0, 1):
                if horizontal == 0 and vertical == 0:
                    continue

                next_row = start_row + vertical
                next_col = start_col + horizontal

                pawns_in_direction = [[], []]
                while 0 <= next_row < Reversi.M and 0 <= next_col < Reversi.N:
                    if self.board[next_row, next_col] == opponent:
                        pawns_in_direction[0].append(next_row)
                        pawns_in_direction[1].append(next_col)
                        next_row += vertical
                        next_col += horizontal
                    elif self.board[next_row, next_col] == player:
                        pawns_to_flip_coords[0].extend(pawns_in_direction[0])
                        pawns_to_flip_coords[1].extend(pawns_in_direction[1])
                        break
                    else:
                        break

        return pawns_to_flip_coords
