import numpy as np
from mcts import State
from numba import jit
from numba import int8

class Reversi(State):  
    N = M = 8 # rozmiar planszy Reversi (8x8)
    
    SYMBOLS = ["\u25CF", "+", "\u25CB"] 
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
        else:
            self.board = np.zeros((Reversi.M, Reversi.N), dtype=np.int8)
            self.board[[3,4],[3,4]] = -1 # białe
            self.board[[3,4],[4,3]] = 1 # czarne

            # super().__init__(parent) ustawia self.turn = 1 (czarne zaczynają)
    
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
        """
        Places a stone onto the crossing of the board indicated by the action_index (row: ``action_index // Reversi.N``, column: ``action_index % Reversi.N``) 
        and returns ``True`` if the action is legal (crossing was not occupied).
        Otherwise, does no changes and returns ``False``.

        Args:
            action_index (int): 
                index of crossing where to place a stone.
        
        Returns:
            action_legal (bool):
                boolean flag indicating if the specified action was legal and performed.
        """        
        row = action_index // Reversi.N
        col = action_index % Reversi.N

        pawns_indices = self.get_pawns_to_flip(action_index)
        if not pawns_indices:
            return False

        self.board[row, col] = self.turn
        for row_idx, col_idx in pawns_indices:
            self.board[row_idx, col_idx] = self.turn

        self.turn *= -1

        return True
    
    def compute_outcome_job(self):
        """        
        Computes and returns the game outcome for this state in compliance with rules of Reversi game: 
        {-1, 1} denoting a win for the minimizing or maximizing player, respectively, if he connected exactly 5 his stones (6 or more do not count for a win); 
        0 denoting a tie, when the board is filled and no line of 5 exists; 
        ``None`` when the game is ongoing.
        
        Returns:
            outcome ({-1, 0, 1} or ``None``)
                game outcome for this state.
        """        
        i = self.last_action_index // Reversi.N
        j = self.last_action_index % Reversi.N
        if True: # a bit faster outcome via numba
            numba_outcome = Reversi.compute_outcome_job_numba_jit(Reversi.M, Reversi.N, self.turn, i, j, self.board)
            if numba_outcome != 0:
                return numba_outcome 
        else:
            last_token = -self.turn        
            # N-S
            total = 0
            for k in range(1, 6):
                if i -  k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Reversi.M or self.board[i + k, j] != last_token:
                    break            
                total += 1
            if total == 4:
                return last_token                        
            # E-W
            total = 0
            for k in range(1, 6):
                if j + k >= Reversi.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break            
                total += 1
            if total == 4:
                return last_token            
            # NE-SW
            total = 0
            for k in range(1, 6):
                if i - k < 0 or j + k >= Reversi.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Reversi.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1            
            if total == 4:
                return last_token            
            # NW-SE
            total = 0
            for k in range(1, 6):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Reversi.M or j + k >= Reversi.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1            
            if total == 4:
                return last_token                                    
        if np.sum(self.board == 0) == 0: # draw
            return 0
        return None        

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
        indexes = np.where(np.ravel(self.board) == 0)[0]
        action_index = np.random.choice(indexes) 
        child = self.take_action(action_index)
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
        """        
        Returns an action's index (numbering from 0) based on its name. E.g., name ``"B4"`` for 15 x 15 Reversi maps to index ``18``.
        
        Args:
            action_name (str):
                name of an action.
        Returns:
            action_index (int):
                index corresponding to the given name.   
        """        
        col = action_name[0].upper()
        row = int(action_name[1]) - 1
        i = row
        j = ord(col) - ord('A')
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
    
    # zrobione - pomyślec nad tym czy nie lepiej przechowywac row coords i col coords żeby w take action od razu przekazac liste dwoch list X i Y - żeby nie trzeba bylo zmieniac pionków w petli
    def get_pawns_to_flip(self, action_index):
        start_row = action_index // Reversi.N
        start_col = action_index % Reversi.N
        pawns_to_flip = []

        # jeśli pole jest zajęte
        if self.board[start_row, start_col] != 0:
            return pawns_to_flip

        player = self.turn
        opponent = -player

        for horizontal in (-1, 0, 1):
            for vertical in (-1, 0, 1):
                if horizontal == 0 and vertical == 0:
                    continue

                next_row = start_row + vertical
                next_col = start_col + horizontal

                pawns_in_direction = []
                while 0 <= next_row < Reversi.M and 0 <= next_col < Reversi.N:
                    if self.board[next_row, next_col] == opponent:
                        pawns_in_direction.append((next_row, next_col))
                        next_row += vertical
                        next_col += horizontal
                    elif self.board[next_row, next_col] == player:
                        pawns_to_flip.extend(pawns_in_direction)
                        break
                    else:
                        break

        return pawns_to_flip
