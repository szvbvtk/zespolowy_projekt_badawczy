import numpy as np
from mcts import State
from numba import jit
from numba import int8


class Reversi(State):
    N = M = 6  

    SYMBOLS = ["\u25cf", "+", "\u25cb"]

    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
        else:
            self.board = np.zeros((Reversi.M, Reversi.N), dtype=np.int8)
            mid_row = Reversi.M // 2
            mid_col = Reversi.N // 2

            self.board[mid_row - 1, mid_col - 1] = -1  # białe
            self.board[mid_row, mid_col] = -1   
            self.board[mid_row - 1, mid_col] = 1       # czarne
            self.board[mid_row, mid_col - 1] = 1


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
        row = action_index // Reversi.N
        col = action_index % Reversi.N

        pawns_indices = self.get_pawns_to_flip(action_index)
        if not pawns_indices[0]:
            return False

        self.board[row, col] = self.turn
        # for row_idx, col_idx in pawns_indices:
        #     self.board[row_idx, col_idx] = self.turn
        self.board[pawns_indices[0], pawns_indices[1]] = self.turn

        if self.has_legal_actions(-self.turn):
            self.turn *= -1

        return True

    # zrobione - częsciowo
    def compute_outcome_job(self):
        # czy gracz może wykonać ruch
        if self.has_legal_actions(self.turn):
            return None

        # jeśli nie może to sprawdzamy czy chociaż przeciwnik może, najwyżej tura obecnego gracza przepadnie
        # if self.last_action_index is None:
        #     return None

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

    # zrobione
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

    @staticmethod
    @jit(int8(int8, int8, int8, int8, int8, int8[:, :]), nopython=True, cache=True)
    def compute_outcome_job_numba_jit(M, N, turn, last_i, last_j, board):
        """Called by ``compute_outcome_job`` for faster outcomes."""
        last_token = -turn
        i, j = last_i, last_j
        # N-S
        total = 0
        for k in range(1, 6):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if i + k >= M or board[i + k, j] != last_token:
                break
            total += 1
        if total == 4:
            return last_token
        # E-W
        total = 0
        for k in range(1, 6):
            if j + k >= N or board[i, j + k] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if j - k < 0 or board[i, j - k] != last_token:
                break
            total += 1
        if total == 4:
            return last_token
        # NE-SW
        total = 0
        for k in range(1, 6):
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
                break
            total += 1
        if total == 4:
            return last_token
        # NW-SE
        total = 0
        for k in range(1, 6):
            if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1
        if total == 4:
            return last_token
        return 0

    def take_random_action_playout(self):
        """
        Picks a uniformly random action from actions available in this state and returns the result of calling ``take_action`` with the action index as argument.

        Returns:
            child (State):
                result of ``take_action`` call for the random action.
        """
        legal_actions = self.get_all_legal_actions(self.turn)

        random_action_index = np.random.choice(legal_actions)

        child = self.take_action(random_action_index)

        return child


    # zrobione
    def get_board(self):
        return self.board

    # zrobione
    def get_extra_info(self):
        # W reversi nie ma dodatkowych informacji
        return None

    # zrobione
    @staticmethod
    def action_name_to_index(action_name):
        col = action_name[0].upper()
        row = int(action_name[1]) - 1
        i = row
        j = ord(col) - ord("A")
        return i * Reversi.N + j

    # zrobione
    @staticmethod
    def action_index_to_name(action_index):
        row = action_index // Reversi.N
        col = action_index % Reversi.N

        return f"{chr(ord('A') + col)}{row + 1}"

    # zrobione
    @staticmethod
    def get_board_shape():
        return (Reversi.M, Reversi.N)

    # zrobione
    @staticmethod
    def get_extra_info_memory():
        return 0

    # zrobione
    @staticmethod
    def get_max_actions():
        return Reversi.M * Reversi.N

    def get_pawns_to_flip(self, action_index, turn=None):
        start_row = action_index // Reversi.N
        start_col = action_index % Reversi.N
        pawns_to_flip_coords = [[], []]  # wiersze, kolumny

        # jeśli pole jest zajęte
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

                pawns_in_direction = [[], []]  # wiersze, kolumny
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
