import numpy as np

SYMBOLS = ["\u25cf", "+", "\u25cb"]

def __str__(self):
    s = ""

    for row_idx in range(8):
        s += f"{row_idx + 1}| "
        for col_idx in range(8):
            s += f"{SYMBOLS[board[row_idx, col_idx] + 1]} "

        s += "\n"

    s += "   "
    for col_idx in range(8):
        s += f"{chr(ord('A') + col_idx)} "

    s += "\n"

    return s

board = np.zeros((8, 8), dtype=np.int8)

# 1| + + + ● ● ● ● ●
# 2| + + + ● ● ● ● ●
# 3| + + + ○ ● ○ ● ●
# 4| + + + ○ ○ ● ● ●
# 5| ● + ○ ○ ○ ○ ● ●
# 6| ● ● ○ ○ ○ + + ●
# 7| ● + ○ + + + + +
# 8| ● ● ● ● ● + + +
#    A B C D E F G H

board[0, 3] = -1
board[0, 4] = -1
board[0, 5] = -1
board[0, 6] = -1
board[0, 7] = -1
board[1, 3] = -1
board[1, 4] = -1
board[1, 5] = -1
board[1, 6] = -1
board[1, 7] = -1
board[2, 4] = -1
board[2, 6] = -1
board[2, 7] = -1
board[3, 5] = -1
board[3, 6] = -1
board[3, 7] = -1
board[4, 6] = -1
board[4, 7] = -1
board[5, 7] = -1
board[4, 0] = -1
board[5, 0] = -1
board[5, 1] = -1
board[6, 0] = -1
board[7, 0] = -1
board[7, 1] = -1
board[7, 2] = -1
board[7, 3] = -1
board[7, 4] = -1

board[3, 3] = 1
board[4, 4] = 1
board[3, 4] = 1
board[4, 3] = 1
board[2, 3] = 1
board[2, 5] = 1
board[4, 5] = 1
board[4, 2] = 1
board[5, 2] = 1
board[5, 3] = 1
board[5, 4] = 1
board[6, 2] = 1


str_board = __str__(board)
print(str_board)
